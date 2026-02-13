import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu(h)
        std = torch.exp(self.log_std).clamp(1e-3, 2.0)
        v = self.v(h).squeeze(-1)
        return mu, std, v


class PPO:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, clip=0.2, vf_coef=0.5, ent_coef=0.01):
        self.gamma = gamma
        self.clip = clip
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

        self.model = PolicyNet(obs_dim, act_dim)
        self.opt = optim.Adam(self.model.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs_vec):
        x = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
        mu, std, v = self.model(x)
        dist = torch.distributions.Normal(mu, std)
        a = dist.sample()
        logp = dist.log_prob(a).sum(-1)
        return a.squeeze(0).numpy(), float(logp.item()), float(v.item())

    def update(self, batch, epochs=4, bs=256):
        obs = torch.tensor(batch["obs"], dtype=torch.float32)
        act = torch.tensor(batch["act"], dtype=torch.float32)
        old_logp = torch.tensor(batch["logp"], dtype=torch.float32)
        ret = torch.tensor(batch["ret"], dtype=torch.float32)
        adv = torch.tensor(batch["adv"], dtype=torch.float32)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = obs.size(0)
        idx = torch.randperm(n)

        for _ in range(epochs):
            for i in range(0, n, bs):
                j = idx[i:i + bs]
                mu, std, v = self.model(obs[j])
                dist = torch.distributions.Normal(mu, std)

                logp = dist.log_prob(act[j]).sum(-1)
                ratio = torch.exp(logp - old_logp[j])

                surr1 = ratio * adv[j]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv[j]
                pi_loss = -torch.min(surr1, surr2).mean()

                v_loss = (ret[j] - v).pow(2).mean()
                ent = dist.entropy().sum(-1).mean()

                loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
