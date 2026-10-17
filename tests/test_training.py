import copy
import json
import random
import numpy as np
import pytest
import torch

from gasp.models import Rollouts, model_digest, rng_state, token_logps
from gasp.losses import advantages, clipped_loss
from gasp.verification import correct, extract_answer, polluter_reward


@pytest.mark.parametrize("condition",["unified","split"])
def test_role_updates_and_frozen_snapshot(factory, condition):
    t=factory(condition)
    assert (t.roles.agent is t.roles.polluter) == (condition=="unified")
    initial_agent=model_digest(t.roles.agent)
    initial_polluter=model_digest(t.roles.polluter)
    assert initial_agent==initial_polluter
    e=t.step()
    assert e["grad_norm"]>0
    assert model_digest(t.roles.agent)!=initial_agent
    assert (model_digest(t.roles.polluter)!=initial_polluter) == (condition=="unified")
    frozen=model_digest(t.roles.opponent)
    assert frozen==initial_polluter
    t.step()
    assert model_digest(t.roles.opponent)==frozen
    before_a, before_p=model_digest(t.roles.agent),model_digest(t.roles.polluter)
    # Entry to polluter block snapshots the current agent. Polluter real rewards
    # come from sampled binary agent outputs, not mocked verifier outcomes.
    e=t.step()
    assert t.roles.snapshot_role=="agent"
    assert model_digest(t.roles.opponent)==before_a
    assert e["grad_norm"]>0
    assert model_digest(t.roles.polluter)!=before_p
    for c in e["contexts"]:
        assert c["rewards"]==[1-v for v in c["agent_correctness"]]
    assert (model_digest(t.roles.agent)!=before_a) == (condition=="unified")
    assert all(p.grad is None for p in t.roles.opponent.parameters())
    if condition=="split":
        assert t.roles.snapshot is None and t.roles.opponent is t.roles.agent
        assert all(p.requires_grad for p in t.roles.agent.parameters())
    else:
        assert all(not p.requires_grad for p in t.roles.snapshot.parameters())
    t.step()
    assert model_digest(t.roles.opponent)==before_a
    last_p=model_digest(t.roles.polluter)
    t.step()
    assert model_digest(t.roles.opponent)==last_p


def test_old_logps_detached_and_guidance_conditioning(factory):
    t=factory("split")
    entry=t.step()["contexts"][0]
    p=t.tokenizer.decode(entry["privileged_prompt_ids"])
    d=t.tokenizer.decode(entry["guidance_conditioning_ids"])
    assert "repair" in p and "clean" in p and "polluted" in p
    assert "agent" in d and "clean" not in d and "polluted" in d
    # Trace the actual model identity used by generator; loss computation remains real.
    calls=[]
    original=t.rollouts.generate
    def record(model,*args,**kwargs):
        result=original(model,*args,**kwargs)
        calls.append((model,args[3],result))
        return result
    t.rollouts.generate=record
    t.step()
    assert [m for m,purpose,_ in calls if purpose=="repair_generation"]==[t.roles.agent]
    assert all(not s.old_logps.requires_grad for _,_,ss in calls for s in ss if s.old_logps is not None)


def test_non_smoke_autoregressive_generation_and_gradients(factory):
    t=factory("split")
    # Exercise model.generate and unrestricted token log probabilities, which
    # the binary smoke sampler bypasses. The weights remain tiny and offline.
    c=t.cfg.with_changes(tiny=False)
    samples=Rollouts(t.tokenizer,c).generate(t.roles.agent,[5,3],2,4,"general",old=True)
    for sample in samples:
        assert 1 <= len(sample.tokens) <= 4
        assert not sample.old_logps.requires_grad
        logps=token_logps(t.roles.agent,sample.prompt,sample.tokens,c)
        torch.testing.assert_close(logps.detach(),sample.old_logps)
        clipped_loss(logps,sample.old_logps,1.).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in t.roles.agent.parameters())
    assert all(p.grad is None for p in t.roles.polluter.parameters())


def assert_tree_equal(a,b):
    if torch.is_tensor(a):
        assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):
        assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: assert_tree_equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): assert_tree_equal(x,y)
    else:
        assert a==b


@pytest.mark.parametrize("condition",["unified","split"])
@pytest.mark.parametrize("cut",[1,2,3])
def test_resume_exact_inside_and_between_blocks(factory,condition,cut):
    uninterrupted=factory(condition)
    for _ in range(6): uninterrupted.step()
    expected_roles=copy.deepcopy(uninterrupted.roles.state_dict())
    expected_opt=copy.deepcopy(uninterrupted.optimizer.state_dict())
    expected_sched=copy.deepcopy(uninterrupted.scheduler.state_dict())
    expected_rng=rng_state()
    interrupted=factory(condition)
    for _ in range(cut): interrupted.step()
    path=interrupted.save()
    resumed=factory(condition)
    resumed.resume(path)
    assert resumed.block_position==cut%2
    for _ in range(6-cut): resumed.step()
    assert_tree_equal(expected_roles,resumed.roles.state_dict())
    assert_tree_equal(expected_opt,resumed.optimizer.state_dict())
    assert_tree_equal(expected_sched,resumed.scheduler.state_dict())
    assert_tree_equal(expected_rng,rng_state())
    assert resumed.role_counts==uninterrupted.role_counts
    assert resumed.cursor==uninterrupted.cursor and resumed.order==uninterrupted.order
    assert resumed.data_rng.getstate()==uninterrupted.data_rng.getstate()
    assert resumed.rollouts.counts==uninterrupted.rollouts.counts
    assert resumed.loss_tokens==uninterrupted.loss_tokens


def test_binary_rewards_and_zero_variance():
    assert polluter_reward(0)==1 and polluter_reward(1)==0
    with pytest.raises(ValueError):polluter_reward(.5)
    assert torch.equal(advantages([0,0]),torch.zeros(2))
    assert torch.equal(advantages([1,1]),torch.zeros(2))
    torch.testing.assert_close(advantages([0,1]),torch.tensor([-1.,1.]))


def test_loss_detaches_old_probabilities_and_reward():
    new=torch.tensor([-.7,-.8],requires_grad=True)
    old=torch.tensor([-.9,-1.],requires_grad=True)
    reward=torch.tensor(1.,requires_grad=True)
    clipped_loss(new,old,reward).backward()
    assert old.grad is None and reward.grad is None and new.grad is not None


def test_verifier_only_final_answer():
    assert correct("<think>9</think><answer>1,200</answer>","1200")==1
    assert correct("I considered 9 but have not finished",9)==0
    assert extract_answer(r"Then \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert correct("#### 2", "2")==1
    assert extract_answer(r"Our answer is $\boxed 2$.") == "2"
    assert extract_answer(r"It follows that $x^2+y^2 = \boxed 9$.") == "9"
    assert extract_answer(r"Then \boxed {\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_answer(r"Unfinished \boxed 12") is None
    assert extract_answer(r"Unfinished \boxed{12") is None


def test_guidance_schedule(config):
    c=config.with_changes(role_steps=100,block_steps=5)
    assert c.guidance_weight(0)==.07 and c.guidance_weight(79)==.07
    assert c.guidance_weight(90)>c.guidance_weight(98)>c.guidance_weight(99)==0


@pytest.mark.parametrize("condition", ["unified", "split"])
def test_optimizer_moments_follow_parameter_ownership(factory, condition):
    t=factory(condition)
    a=next(t.roles.agent.parameters()); p=next(t.roles.polluter.parameters())
    params=t.optimizer.param_groups[0]["params"]
    assert len(params)==len({id(x) for x in params})
    t.step(); t.step()
    assert t.optimizer.state[a]["step"]==2
    if condition=="split":
        assert p not in t.optimizer.state
    agent_moment=t.optimizer.state[a]["exp_avg"].clone()
    t.step()
    assert t.optimizer.state[p]["step"]==(3 if condition=="unified" else 1)
    if condition=="split":
        assert torch.equal(agent_moment, t.optimizer.state[a]["exp_avg"])
        assert t.optimizer.state[a]["step"]==2
    assert t.scheduler.last_epoch==3


@pytest.mark.parametrize("condition",["unified","split"])
def test_lora_pilot_updates_and_resume(config,tmp_path,condition):
    pytest.importorskip("peft")
    from gasp.data import make_smoke_data
    from gasp.trainer import Trainer
    c=config.with_changes(tuning="lora",lora_rank=4,gradient_checkpointing=True,
                          condition=condition,block_steps=1,role_steps=2)
    make_smoke_data(tmp_path/"data",c)
    def make(out):
        return Trainer(c,tmp_path/"data/train.jsonl",tmp_path/"data/eval.jsonl",tmp_path/out)
    t=make("a")
    base={n:p.detach().clone() for n,p in t.roles.agent.named_parameters() if not p.requires_grad}
    initial=model_digest(t.roles.agent)
    t.step();t.step()
    assert model_digest(t.roles.agent)!=initial
    for n,p in t.roles.agent.named_parameters():
        if n in base:assert torch.equal(p,base[n])
    path=t.save()
    t.step();expected=model_digest(t.roles.agent)
    resumed=make("b");resumed.resume(path);resumed.step()
    assert model_digest(resumed.roles.agent)==expected
