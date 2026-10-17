import torch


def advantages(rewards, epsilon=1e-8):
    rewards = torch.as_tensor(rewards, dtype=torch.float32).detach()
    return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + epsilon)


def clipped_loss(new_logps, old_logps, advantage, clip=.2):
    if new_logps.shape != old_logps.shape:
        raise ValueError("Old/new log probability shape mismatch")
    ratio = (new_logps - old_logps.detach()).exp()
    adv = torch.as_tensor(advantage, device=ratio.device).detach()
    return -torch.minimum(ratio * adv, ratio.clamp(1-clip, 1+clip) * adv).mean()
