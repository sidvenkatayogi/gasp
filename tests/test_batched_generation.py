import pytest
import torch

from gasp.models import Rollouts, TinyTokenizer, load_model


@pytest.mark.parametrize("device,precision", [("cpu", "fp32"), pytest.param("cuda", "bf16",
    marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA"))])
def test_mixed_length_greedy_batch_matches_separate_generation(config, device, precision):
    cfg = config.with_changes(device=device, precision=precision)
    model, tok = load_model(cfg)
    # Exercise ordinary Hugging Face generation, not the binary smoke branch.
    rollout = Rollouts(tok, cfg.with_changes(tiny=False))
    prompts = [[2, 3, 4], [2, 3], [2, 4, 3, 2]]
    expected = [rollout.generate(model, p, 1, 6, "separate", temperature=0)[0].tokens
                for p in prompts]
    actual = rollout.generate_many(model, prompts, 1, 6, "batched", temperature=0)
    assert [group[0].tokens for group in actual] == expected
    assert [group[0].prompt for group in actual] == prompts
    assert rollout.counts["batched"] == rollout.counts["separate"]
    assert model.training


def test_sampling_groups_eos_padding_and_settings(config):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def generate(self, input_ids, attention_mask, **kwargs):
            assert input_ids.tolist() == [[2, 3], [0, 4]]
            assert attention_mask.tolist() == [[1, 1], [0, 1]]
            assert kwargs["num_return_sequences"] == 2
            assert kwargs["do_sample"] and kwargs["temperature"] == .7
            assert kwargs["top_k"] == config.top_k and kwargs["top_p"] == 1.
            assert kwargs["max_new_tokens"] == 3
            assert not self.training
            return torch.tensor([[2, 3, 4, 1, 0], [2, 3, 5, 6, 1],
                                 [0, 4, 7, 1, 0], [0, 4, 8, 9, 10]])

    model = Model()
    rollout = Rollouts(TinyTokenizer(), config.with_changes(tiny=False))
    groups = rollout.generate_many(model, [[2, 3], [4]], 2, 3, "filter", temperature=.7)
    assert [[s.tokens for s in group] for group in groups] == [
        [[4, 1], [5, 6, 1]], [[7, 1], [8, 9, 10]]]
    assert [[s.prompt for s in group] for group in groups] == [[[2, 3], [2, 3]], [[4], [4]]]
    assert rollout.counts["filter"] == {"sequences": 4, "generated_tokens": 10,
                                        "prompt_tokens": 6, "allocated_tokens": 12}
    assert model.training


def test_batch_failure_restores_model_mode(config):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def generate(self, **kwargs):
            raise RuntimeError("generation failed")

    model = Model()
    rollout = Rollouts(TinyTokenizer(), config.with_changes(tiny=False))
    with pytest.raises(RuntimeError, match="generation failed"):
        rollout.generate_many(model, [[2, 3], [4]], 4, 3, "filter")
    assert model.training and rollout.counts == {}
