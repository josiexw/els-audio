import os, math, random, time
import numpy as np
import soundfile as sf
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "data/esc50/train"
OUT_DIR  = "checkpoints/ddpm"
BATCH_SIZE = 1
EPOCHS = 300
LR = 1e-4
LR_HALF_EVERY_EPOCHS = 50
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
LOG_EVERY = 500
SEED = 0
TIMESTEPS = 1000
BETA_SCHEDULE  = "cosine"
SAMPLER = "ddim"
DDIM_STEPS = 20
DDIM_ETA = 0.0
BASE_CHS = (64, 128, 256)
DROPOUT = 0.0
T_DIM = 256
TARGET_SR = 44100
N_FFT = 1024
HOP = 256
TARGET_SECONDS = 1.0
TARGET_FRAMES = int(TARGET_SECONDS * TARGET_SR / HOP) + 1
GAIN_DB = 6.0
CENTER_STFT = True
NUM_SAMPLES_TO_SAVE = 1
SAVE_EVERY_EPOCH = True
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
USE_AMP = True
SET_CHANNELS_LAST = torch.cuda.is_available()
PADDING_MODE = "zero"  # zero or circular

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
print(f"Using device: {DEVICE}")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

def time_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device).float() / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, device='cpu')
    ac = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return torch.clip(betas, 1e-7, 0.999)

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, timesteps)

def make_ddpm_params(timesteps, schedule):
    betas = cosine_beta_schedule(timesteps) if schedule == "cosine" else linear_beta_schedule(timesteps)
    alphas = 1.0 - betas
    ac = torch.cumprod(alphas, dim=0)
    ac_prev = torch.cat([torch.tensor([1.0]), ac[:-1]], dim=0)
    sac = torch.sqrt(ac)
    s1mac = torch.sqrt(1.0 - ac)
    sra = torch.sqrt(1.0 / alphas)
    post_var = betas * (1.0 - ac_prev) / (1.0 - ac)
    return {
        "betas": betas, "alphas": alphas, "alphas_cumprod": ac,
        "sqrt_alphas_cumprod": sac, "sqrt_one_minus_alphas_cumprod": s1mac,
        "sqrt_recip_alphas": sra, "posterior_variance": post_var
    }

def load_wav_mono(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim == 1:
        x = x[None, :]
    else:
        x = x.T
    if x.shape[0] > 1:
        x = x.mean(axis=0, keepdims=True)
    return torch.from_numpy(x), sr

class WavToSTFT2D(Dataset):
    def __init__(self, root, n_fft=N_FFT, hop=HOP, target_frames=TARGET_FRAMES, gain_db=GAIN_DB, center=CENTER_STFT):
        paths = []
        for r, _, fs in os.walk(root):
            for f in fs:
                if f.lower().endswith(".wav"):
                    paths.append(os.path.join(r, f))
        self.paths = sorted(paths)
        assert len(self.paths) > 0, f"No wav files found under {root}"
        self.n_fft = n_fft
        self.hop = hop
        self.win = torch.hann_window(n_fft)
        self.target_frames = target_frames
        self.gain_db = gain_db
        self.center = center

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        wav, sr = load_wav_mono(self.paths[idx])
        if self.gain_db and self.gain_db > 0:
            g = 10 ** (random.uniform(-self.gain_db, self.gain_db) / 20.0)
            wav = wav * g
        x = wav[0]
        rms = x.pow(2).mean().sqrt().clamp(min=1e-6)
        x = x * (0.2 / rms)
        x = x.clamp(-1.0, 1.0)
        Xc = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop, win_length=self.n_fft,
            window=self.win, return_complex=True, center=self.center
        )
        if self.target_frames is not None:
            T = Xc.shape[-1]
            if T >= self.target_frames:
                s = (T - self.target_frames) // 2
                Xc = Xc[:, s:s+self.target_frames]
            else:
                pad = self.target_frames - T
                Xc = F.pad(Xc, (0, pad))
        Xi = torch.stack([Xc.real, Xc.imag], dim=0).to(torch.float32)
        return Xi

class Conv2dPad(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, padding_mode="zero"):
        super().__init__()
        self.padding_mode = padding_mode
        self.k = k
        self.s = s
        pad = (k - 1) // 2
        self.pad = pad
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=0 if padding_mode == "circular" else pad)
    def forward(self, x):
        if self.padding_mode == "circular":
            x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="circular")
        return self.conv(x)

class ResBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, dropout=0.0, padding_mode="zero"):
        super().__init__()
        self.c1 = Conv2dPad(in_ch, out_ch, 3, 1, padding_mode)
        self.a1 = nn.SiLU()
        self.tp = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, out_ch))
        self.d  = nn.Dropout(dropout)
        self.c2 = Conv2dPad(out_ch, out_ch, 3, 1, padding_mode)
        self.a2 = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, t_emb):
        h = self.a1(self.c1(x))
        h = h + self.tp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.a2(self.c2(self.d(h)))
        return h + self.skip(x)

class Down2D(nn.Module):
    def __init__(self, ch_in, ch_out, t_dim, dropout=0.0, padding_mode="zero"):
        super().__init__()
        self.rb1 = ResBlock2D(ch_in, ch_out, t_dim, dropout, padding_mode)
        self.rb2 = ResBlock2D(ch_out, ch_out, t_dim, dropout, padding_mode)
        self.pool = Conv2dPad(ch_out, ch_out, 4, 2, padding_mode)
    def forward(self, x, t):
        x = self.rb1(x, t)
        x = self.rb2(x, t)
        return self.pool(x), x

class Up2D(nn.Module):
    def __init__(self, ch_in, ch_out, skip_ch, t_dim, dropout=0.0, padding_mode="zero"):
        super().__init__()
        self.up  = nn.ConvTranspose2d(ch_in, ch_out, 4, stride=2, padding=1)
        self.rb1 = ResBlock2D(ch_out + skip_ch, ch_out, t_dim, dropout, padding_mode)
        self.rb2 = ResBlock2D(ch_out, ch_out, t_dim, dropout, padding_mode)
    def forward(self, x, skip, t):
        x = self.up(x)
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            x = F.pad(x, (0, skip.size(-1)-x.size(-1), 0, skip.size(-2)-x.size(-2)))
        x = torch.cat([x, skip], dim=1)
        x = self.rb1(x, t)
        x = self.rb2(x, t)
        return x

class UNet2DConvOnly(nn.Module):
    def __init__(self, in_ch=2, base_chs=BASE_CHS, t_dim=T_DIM, dropout=DROPOUT, padding_mode=PADDING_MODE):
        super().__init__()
        self.tdim = t_dim
        self.tmlp = nn.Sequential(nn.Linear(t_dim, t_dim*4), nn.SiLU(), nn.Linear(t_dim*4, t_dim))
        chs = list(base_chs)
        self.in_conv = Conv2dPad(in_ch, chs[0], 3, 1, padding_mode)
        self.downs = nn.ModuleList([Down2D(chs[i], chs[i+1], t_dim, dropout, padding_mode) for i in range(len(chs)-1)])
        self.mid1 = ResBlock2D(chs[-1], chs[-1], t_dim, dropout, padding_mode)
        self.mid2 = ResBlock2D(chs[-1], chs[-1], t_dim, dropout, padding_mode)
        rev = list(reversed(chs))
        skip_chs = chs[1:]
        rev_skip = list(reversed(skip_chs))
        self.ups = nn.ModuleList([Up2D(rev[i], rev[i+1], rev_skip[i], t_dim, dropout, padding_mode) for i in range(len(rev)-1)])
        self.out_a = nn.SiLU()
        self.out_c = Conv2dPad(chs[0], in_ch, 3, 1, padding_mode)
    def forward(self, x, t):
        te = self.tmlp(time_embedding(t, self.tdim).to(x.device))
        x = self.in_conv(x)
        skips = []
        for d in self.downs:
            x, s = d(x, te); skips.append(s)
        x = self.mid1(x, te); x = self.mid2(x, te)
        for u in self.ups:
            s = skips.pop(); x = u(x, s, te)
        return self.out_c(self.out_a(x))

@torch.no_grad()
def sample_ddpm_2d(model, params, shape, device):
    model.eval()
    betas = params["betas"].to(device)
    sra   = params["sqrt_recip_alphas"].to(device)
    pvar  = params["posterior_variance"].to(device)
    T = betas.shape[0]
    x = torch.randn(shape, device=device)
    for i in reversed(range(T)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        eps = model(x, t)
        mean = sra[t].view(-1,1,1,1) * (x - betas[t].sqrt().view(-1,1,1,1) * eps)
        if i > 0:
            x = mean + torch.sqrt(pvar[t]).view(-1,1,1,1) * torch.randn_like(x)
        else:
            x = mean
    return x

@torch.no_grad()
def sample_ddim_2d(model, params, shape, device, eta=0.0, steps=20):
    model.eval()
    sac = params["sqrt_alphas_cumprod"].to(device)
    s1  = params["sqrt_one_minus_alphas_cumprod"].to(device)
    T = params["betas"].shape[0]
    ts = torch.linspace(T - 1, 0, steps, dtype=torch.long, device=device)
    x = torch.randn(shape, device=device)
    for i in range(steps):
        t = ts[i]
        eps = model(x, t.repeat(shape[0]))
        x0 = (x - s1[t].view(1,1,1,1) * eps) / sac[t].view(1,1,1,1)
        if i == steps - 1:
            x = x0
        else:
            t_prev = ts[i + 1]
            a_t, a_prev   = sac[t], sac[t_prev]
            s1_t, s1_prev = s1[t], s1[t_prev]
            sigma = eta * torch.sqrt((s1_prev**2 - ((a_prev / a_t) * s1_t)**2).clamp(min=0.0))
            eps_coeff = torch.sqrt((s1_prev**2 - sigma**2).clamp(min=0.0))
            x = a_prev.view(1,1,1,1) * x0 + eps_coeff.view(1,1,1,1) * eps + sigma.view(1,1,1,1) * torch.randn_like(x)
    return x

@torch.no_grad()
def istft_from_ri(ri, n_fft=N_FFT, hop=HOP, center=CENTER_STFT):
    device = ri.device
    win = torch.hann_window(n_fft, device=device)
    Xc = torch.complex(ri[:, 0], ri[:, 1])
    outs = []
    for i in range(Xc.size(0)):
        wav = torch.istft(
            Xc[i], n_fft=n_fft, hop_length=hop, win_length=n_fft,
            window=win, center=center
        )
        rms = wav.pow(2).mean().sqrt().clamp(min=1e-4)
        wav = (wav / (3 * rms)).clamp(-1.0, 1.0)
        outs.append(wav.unsqueeze(0))
    return torch.cat(outs, dim=0)

def save_wavs(batch_wav, sr, prefix, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    batch_wav = batch_wav.detach().cpu().numpy()
    for i, w in enumerate(batch_wav):
        sf.write(os.path.join(out_dir, f"{prefix}_{i}.wav"), w, sr)

def train():
    ds = WavToSTFT2D(DATA_DIR)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                    num_workers=0, pin_memory=False)
    sample_x = ds[0]
    in_ch, Fbins, Tspec = sample_x.shape
    model = UNet2DConvOnly(in_ch=in_ch).to(DEVICE)
    if SET_CHANNELS_LAST and DEVICE == "cuda":
        model = model.to(memory_format=torch.channels_last)
    params = make_ddpm_params(TIMESTEPS, BETA_SCHEDULE)
    sac = params["sqrt_alphas_cumprod"].to(DEVICE)
    s1  = params["sqrt_one_minus_alphas_cumprod"].to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9,0.999), weight_decay=WEIGHT_DECAY)
    gamma = 0.5 ** (1.0 / LR_HALF_EVERY_EPOCHS)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    scaler = torch.amp.GradScaler(enabled=(USE_AMP and DEVICE=="cuda"))
    global_step = 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        running = 0.0
        for X in pbar:
            X = X.to(DEVICE)
            if SET_CHANNELS_LAST and DEVICE == "cuda":
                X = X.to(memory_format=torch.channels_last)
            bsz = X.size(0)
            t   = torch.randint(0, TIMESTEPS, (bsz,), device=DEVICE).long()
            noise = torch.randn_like(X)
            xt = sac[t].view(-1,1,1,1)*X + s1[t].view(-1,1,1,1)*noise
            with torch.autocast(device_type=("cuda" if DEVICE=="cuda" else "cpu"), dtype=torch.float16 if DEVICE=="cuda" else torch.bfloat16, enabled=USE_AMP):
                pred = model(xt, t)
                loss = F.mse_loss(pred, noise)
            opt.zero_grad(set_to_none=True)
            if DEVICE=="cuda" and USE_AMP:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            running += loss.item()
            global_step += 1
            if global_step % LOG_EVERY == 0:
                avg = running / LOG_EVERY
                pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")
                running = 0.0
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "gstep": global_step,
                     "cfg": {"in_ch": in_ch, "F": Fbins, "T": Tspec}},
                    os.path.join(OUT_DIR, f"ckpt_e{epoch}_g{global_step}.pt")
                )
                with torch.no_grad():
                    shape = (NUM_SAMPLES_TO_SAVE, in_ch, Fbins, Tspec)
                    if SAMPLER == "ddim":
                        Y = sample_ddim_2d(model, params, shape, DEVICE, eta=DDIM_ETA, steps=DDIM_STEPS)
                    else:
                        Y = sample_ddpm_2d(model, params, shape, DEVICE)
                    wavs = istft_from_ri(Y)
                    save_wavs(wavs, TARGET_SR, prefix=f"preview_e{epoch}_g{global_step}")
        sched.step()
        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "gstep": global_step,
             "cfg": {"in_ch": in_ch, "F": Fbins, "T": Tspec}},
            os.path.join(OUT_DIR, "last.pt")
        )
        if SAVE_EVERY_EPOCH:
            with torch.no_grad():
                shape = (NUM_SAMPLES_TO_SAVE, in_ch, Fbins, Tspec)
                if SAMPLER == "ddim":
                    Y = sample_ddim_2d(model, params, shape, DEVICE, eta=DDIM_ETA, steps=DDIM_STEPS)
                else:
                    Y = sample_ddpm_2d(model, params, shape, DEVICE)
                wavs = istft_from_ri(Y)
                save_wavs(wavs, TARGET_SR, prefix=f"preview_e{epoch}")

if __name__ == "__main__":
    train()
