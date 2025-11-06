import os, math, random
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

CKPT_PATH = "checkpoints/ddpm/last.pt"
TEST_DIR  = "data/esc50/test"
OUT_DIR   = "assets/ddpm"
FIG_DIR   = "assets/figures"

TIMESTEPS = 1000
BETA_SCHEDULE = "cosine"
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
CENTER_STFT = True
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
SET_CHANNELS_LAST = torch.cuda.is_available()
PAD_MODE = "zero"
SEED = 0
INPUT_T_START = 950

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

def time_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device).float() / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1: emb = F.pad(emb, (0,1))
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
    if sr != TARGET_SR:
        raise RuntimeError(f"Expected {TARGET_SR} Hz, got {sr} Hz: {path}")
    return torch.from_numpy(x), sr

def wav_to_ri_stft(wav, n_fft=N_FFT, hop=HOP, target_frames=TARGET_FRAMES, center=CENTER_STFT):
    x = wav[0]
    rms = x.pow(2).mean().sqrt().clamp(min=1e-6)
    x = x * (0.2 / rms)
    x = x.clamp(-1.0, 1.0)
    win = torch.hann_window(n_fft)
    Xc = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, return_complex=True, center=center)
    if target_frames is not None:
        T = Xc.shape[-1]
        if T >= target_frames:
            s = (T - target_frames) // 2
            Xc = Xc[:, s:s+target_frames]
        else:
            pad = target_frames - T
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
    def __init__(self, in_ch=2, base_chs=BASE_CHS, t_dim=T_DIM, dropout=DROPOUT, padding_mode=PAD_MODE):
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
def sample_ddim_with_init(model, params, shape, device, eta=0.0, steps=20, x_init=None):
    model.eval()
    sac = params["sqrt_alphas_cumprod"].to(device)
    s1  = params["sqrt_one_minus_alphas_cumprod"].to(device)
    T = params["betas"].shape[0]
    ts = torch.linspace(T - 1, 0, steps, device=device).long()
    if x_init is None:
        x = torch.randn(shape, device=device)
        x0_rand = x.clone()
    else:
        x = x_init.clone().to(device)
        x0_rand = x_init.clone().to(device)
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
    return x0_rand, x

@torch.no_grad()
def ddim_refine_from_init(model, params, x_init, device, t_start=950, steps=20, eta=0.0):
    model.eval()
    sac = params["sqrt_alphas_cumprod"].to(device)
    s1  = params["sqrt_one_minus_alphas_cumprod"].to(device)
    t_start = int(max(1, min(TIMESTEPS-1, t_start)))
    ts = torch.linspace(t_start, 0, steps, device=device).long()
    x = x_init.clone().to(device)
    for i in range(steps):
        t = ts[i]
        eps = model(x, t.repeat(x.size(0)))
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
def forward_noise(Xi, params, t_scalar):
    sac = params["sqrt_alphas_cumprod"].to(Xi.device)
    s1  = params["sqrt_one_minus_alphas_cumprod"].to(Xi.device)
    bsz = Xi.size(0)
    t = torch.full((bsz,), int(t_scalar), device=Xi.device, dtype=torch.long)
    noise = torch.randn_like(Xi)
    xt = sac[t].view(-1,1,1,1)*Xi + s1[t].view(-1,1,1,1)*noise
    return xt

@torch.no_grad()
def istft_from_ri(ri, n_fft=N_FFT, hop=HOP, center=CENTER_STFT):
    device = ri.device
    win = torch.hann_window(n_fft, device=device)
    Xc = torch.complex(ri[:, 0], ri[:, 1])
    outs = []
    for i in range(Xc.size(0)):
        wav = torch.istft(Xc[i], n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, center=center)
        rms = wav.pow(2).mean().sqrt().clamp(min=1e-4)
        wav = (wav / (3 * rms)).clamp(-1.0, 1.0)
        outs.append(wav.unsqueeze(0))
    return torch.cat(outs, dim=0)

def save_wavs(batch_wav, sr, prefix, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    batch_wav = batch_wav.detach().cpu().numpy()
    for i, w in enumerate(batch_wav):
        sf.write(os.path.join(out_dir, f"{prefix}_{i}.wav"), w, sr)

def first_test_wav(root=TEST_DIR):
    cands = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(".wav"):
                cands.append(os.path.join(r, f))
    if not cands:
        raise RuntimeError(f"No wav files under {root}")
    cands.sort()
    return cands[7]

def _match_len(a, b, c=None):
    arrays = [a, b] + ([c] if c is not None else [])
    L = min([x.shape[-1] for x in arrays])
    arrays = [x[..., :L] for x in arrays]
    return arrays if c is not None else arrays[:2]

def _plot_triplet(w1, w2, w3, sr, titles, path):
    t = np.linspace(0, len(w1)/sr, num=len(w1), endpoint=False)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for ax, w, title in zip(axes, [w1, w2, w3], titles):
        ax.plot(t, w)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def _plot_pair(w1, w2, sr, titles, path):
    t = np.linspace(0, len(w1)/sr, num=len(w1), endpoint=False)
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    for ax, w, title in zip(axes, [w1, w2], titles):
        ax.plot(t, w)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def main():
    ck = torch.load(CKPT_PATH, map_location=DEVICE)
    cfg = ck.get("cfg", {})
    in_ch = cfg.get("in_ch", 2)
    Fbins = cfg.get("F", N_FFT//2 + 1)
    Tspec = cfg.get("T", TARGET_FRAMES)
    model = UNet2DConvOnly(in_ch=in_ch).to(DEVICE)
    if SET_CHANNELS_LAST and DEVICE == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(ck["model"], strict=True)
    params = make_ddpm_params(TIMESTEPS, BETA_SCHEDULE)

    # === Pure-noise generation (and capture initial noise for figure 2) ===
    shape = (1, in_ch, Fbins, Tspec)
    x_init, Y = sample_ddim_with_init(model, params, shape, DEVICE, eta=DDIM_ETA, steps=DDIM_STEPS, x_init=None)
    wavs_gen = istft_from_ri(Y)
    wavs_noise = istft_from_ri(x_init)
    save_wavs(wavs_gen, TARGET_SR, prefix="gen_from_pure_noise")

    # === Reconstruction from a noised real example (and capture xt for figure 1) ===
    test_path = first_test_wav()
    wav, _ = load_wav_mono(test_path)
    Xi = wav_to_ri_stft(wav)
    Xi_b = Xi.unsqueeze(0).to(DEVICE)
    if SET_CHANNELS_LAST and DEVICE == "cuda":
        Xi_b = Xi_b.to(memory_format=torch.channels_last)

    xt = forward_noise(Xi_b, params, t_scalar=INPUT_T_START)
    Y2 = ddim_refine_from_init(model, params, xt, DEVICE, t_start=INPUT_T_START, steps=DDIM_STEPS, eta=DDIM_ETA)

    wav_orig = istft_from_ri(Xi_b)
    wav_noised = istft_from_ri(xt)
    wav_recon = istft_from_ri(Y2)
    base = os.path.splitext(os.path.basename(test_path))[0]
    save_wavs(wav_recon, TARGET_SR, prefix=f"refined_{base}")

    # === Figures ===
    w0, wN, wR = [x[0].detach().cpu().numpy() for x in (wav_orig, wav_noised, wav_recon)]
    w0, wN, wR = _match_len(w0, wN, wR)
    _plot_triplet(
        w0, wN, wR, TARGET_SR,
        [f"Original ({base})", f"Noised (t={INPUT_T_START})", "Reconstructed"],
        os.path.join(FIG_DIR, "fig_recon.png")
    )

    wZ, wG = [x[0].detach().cpu().numpy() for x in (wavs_noise, wavs_gen)]
    wZ, wG = _match_len(wZ, wG)
    _plot_pair(
        wZ, wG, TARGET_SR,
        ["Pure Gaussian noise (ISTFT of initial RI)", "Generated from noise"],
        os.path.join(FIG_DIR, "fig_gen.png")
    )

    print(f"Saved audio to {OUT_DIR}")
    print(f"Saved figures to {FIG_DIR}")

if __name__ == "__main__":
    main()
