import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# time embedding
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        time: (B,)
        return: (B, dim)
        """
        half_dim = self.dim // 2
        device = time.device
        embedding = math.log(10000) / (half_dim - 1)
        embedding = torch.exp(torch.arange(half_dim, device=device) * -embedding)
        embedding = time[:, None].float() * embedding[None, :] # (B, dim/2)

        embedding = torch.cat([embedding.cos(), embedding.sin()], dim=-1) # (B, dim)
        return embedding

# ResBlock
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_emb = nn.Linear(time_emb_dim, out_channels)
        self.act = nn.SiLU()

        if in_channels == out_channels:
            self.res_conv = nn.Identity()
        else:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, time_embedding):
        # x (B, C, H, W)  time (B, D)
        # x -> conv1 -> +time -> conv2 -> +res
        h = self.act(self.conv1(x)) # (B, out, H, W)
        time_embedding = self.time_emb(time_embedding)[:, :, None, None] # (B, out, 1, 1)
        h = h + time_embedding
        h = self.act(self.conv2(h))
        h = h + self.res_conv(x)
        return h
    
# Down Up
class DownSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)
    
class UpSample(nn.Module):
    def __init__(self, channels):
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)
    
# U-net backbone
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, time_emb_dim=128):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # down
        self.down1 = ResidualBlock(base_channels, base_channels, time_emb_dim)
        self.downsample1 = DownSample(base_channels)

        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.downsample2 = DownSample(base_channels * 2)

        # bottleneck
        self.mid1 = ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.mid2 = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        # up
        self.upsample1 = UpSample(base_channels * 4)
        self.up1 = ResidualBlock(base_channels * 4 + base_channels * 2, base_channels * 2, time_emb_dim)

        self.upsample2 = UpSample(base_channels * 2)
        self.up2 = ResidualBlock(base_channels * 2 + base_channels, base_channels, time_emb_dim)

        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x, time):
        t_emb = self.time_embedding(time)

        x = self.init_conv(x)

        # encoder
        x1 = self.down1(x, t_emb)
        x2 = self.downsample1(x1)

        x3 = self.down2(x2, t_emb)
        x4 = self.downsample2(x3)

        # bottleneck
        x4 = self.mid1(x4, t_emb)
        x4 = self.mid2(x4, t_emb)

        # decoder
        x = self.upsample1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.up1(x, t_emb)

        x = self.upsample2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up2(x, t_emb)

        return self.final_conv(x)
    
def linear_beta_schedule(timesteps):
    beta_start = 1e-4
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def diffusion_loss(diffusion, model, x_start, t):
    eps = torch.randn_like(x_start)
    x_noisy = diffusion.q_sample(x_start, t)
    eps_pred = model(x_noisy, t)
    loss = F.mse_loss(eps, eps_pred)
    return loss

# diffusion 
class Diffusion:
    def __init__(self, timesteps=1000, device='cpu'):
        self.timesteps = timesteps
        self.device = device

        self.betas = linear_beta_schedule(timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def extract(self, a, t):
        # a_t
        B = t.shape[0]
        out = a.gather(-1, t.cpu()).to(t.device) # (B,)
        out = out.reshape(B, 1, 1, 1)
        return out

    def q_sample(self, x_start, t):
        # x_0 -> x_T 
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t)
        eps = torch.randn_like(x_start)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * eps
    
    @torch.no_grad()
    def ddim_sample_step(self, model, x_t, t, t_prev, eta=0.0):
        """
        model: predicts noise epsilon_theta(x_t, t)
        x_t:   (B, C, H, W)
        t:     (B,) current timestep
        t_prev:(B,) previous timestep
        eta:   0 => deterministic DDIM
        """
        alpha_bar_t = self.extract(self.alphas_cumprod, t)
        alpha_bar_prev = self.extract(self.alphas_cumprod, t_prev)
        
        eps_pred = model(x_t, t)
        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
        sigma_t = (
        eta
        * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t))
        * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
        )
        dir_xt = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma_t**2, min=0.0)) * eps_pred
        noise = torch.randn_like(x_t)
        x_prev = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma_t * noise
        return x_prev, x0_pred
    
    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        alphas_cumprod,
        image_size=28,
        channels=1,
        batch_size=16,
        num_sampling_steps=50,
        eta=0.0,
        device="cuda"
    ):
        total_timesteps = alphas_cumprod.shape[0]

        # 例如从 999, 978, 958, ... 到 0
        step_indices = torch.linspace(
            total_timesteps - 1, 0, num_sampling_steps, device=device
        ).long()

        x = torch.randn(batch_size, channels, image_size, image_size, device=device)

        for i in range(len(step_indices) - 1):
            t = step_indices[i].repeat(batch_size) # (B,)
            t_prev = step_indices[i + 1].repeat(batch_size)

            x, _ = self.ddim_sample_step(
                model=model,
                x_t=x,
                t=t,
                t_prev=t_prev,
                alphas_cumprod=alphas_cumprod,
                eta=eta
            )

        # 最后一步直接到 t=0
        t = step_indices[-1].repeat(batch_size)
        alpha_bar_t = self.extract(alphas_cumprod, t)
        eps_pred = model(x, t)
        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

        return x0_pred

# training
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x:x*2.0-1.0)
])

data = datasets.MNIST(root='./data', train=True, transform=trans, download=True)
dataloader = DataLoader(dataset=data, batch_size=64, shuffle=True)

model = SimpleUNet(in_channels=1, base_channels=64, time_emb_dim=128)
diffusion = Diffusion(timesteps=500, device=device)
epochs = 10
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

for epoch in range(epochs):
    model.train()
    for step, (img, _) in enumerate(dataloader):
        B, C, H, W = img.shape
        t = torch.randint(0, diffusion.timesteps, (B,), device=device)
        img = img.to(device)

        x = diffusion.q_sample(img, t)
        loss = diffusion_loss(diffusion=diffusion_loss, model=model, x_start=x, t=t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"epoch {epoch} step {step} loss {loss.item():.4f}")


model.eval()
samples = diffusion.sample(model, image_size=28, batch_size=16, channels=1)
samples = (samples + 1) / 2
samples = samples.clamp(0, 1)