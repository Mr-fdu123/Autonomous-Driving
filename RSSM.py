import torch
import torch.nn as nn
import torch.nn.functional as F

# 高斯采样
def sample_gussian(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

class RSSM(nn.Module):
    def __init__(self, obs_dim, actions_dim, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        # GRU
        self.gru = nn.GRUCell(actions_dim + latent_dim, hidden_dim)
        # prior: p(z_t | h_t)
        self.prior = nn.Linear(hidden_dim, 2*latent_dim)
        # posterior: q(z_t | h_t,o_t)
        self.post = nn.Linear(hidden_dim + obs_dim, 2*latent_dim)
        # decorder (h_t, z_t) -> o_t
        self.decorder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, obs_dim)
        )

    def kl_loss(self, mu_p, mu_q, logvar_p, logvar_q):
        kl = -0.5 * torch.sum(
                1 + logvar_q - logvar_p
                - ((mu_q - mu_p) ** 2 + logvar_q.exp()) / logvar_p.exp()
            )
        return kl

    def forward(self, obs, actions):
        # obs (B, T, obs) a (B, T, actions)
        B, T, _ = obs.shape
        # 初始化h, z
        h = torch.zeros(B, self.hidden_dim, device=obs.device)
        z = torch.zeros(B, self.latent_dim, device=obs.device)

        o_recons = []
        kls = []

        for t in range(T):
            a_t = actions[:, t, :]
            o_t = obs[:, t, :]
            # update GRU
            h = self.gru(torch.cat([a_t, z], dim=-1), h)
            # prior
            prior_out = self.prior(h)
            mu_p, logvar_p = torch.chunk(prior_out, 2, dim=-1) # (B, latent)
            # posterior
            post_out = self.post(torch.cat([h, o_t], dim=-1))
            mu_q, logvar_q = torch.chunk(post_out, 2, dim=-1)
            # sample  z_t
            z = sample_gussian(mu_p, logvar_p)
            # recon o_t
            o_recon = self.decorder(torch.cat([h, z], dim=-1)) # (B, obs)

            o_recons.append(o_recon)
            kls.append(self.kl_loss(mu_p, mu_q, logvar_p, logvar_q))

        recons = torch.stack(o_recons, dim=1) # (B, T, obs)
        return recons, kls

def rssm_loss(recons, obs, kls):
    recon_loss = F.mse_loss(recons, obs)
    kl_loss = torch.mean(torch.stack(kls))
    return recon_loss + kl_loss

# training
model = RSSM(obs_dim=10, actions_dim=4).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(1000):
    obs = torch.randn(32, 20, 10).cuda()
    actions = torch.randn(32, 20, 4).cuda()

    recon, kls = model(obs, actions)
    loss = rssm_loss(obs, recon, kls)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(loss.item())



            


