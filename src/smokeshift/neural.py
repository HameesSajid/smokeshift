"""A patch transformer for multi-horizon quantile forecasting, written in PyTorch.

Architecture (in the spirit of PatchTST, Nie et al. 2023):

    168 h of history  ->  21 patches of 8 h  ->  linear embedding
    + one token for calendar and location
    -> 3 transformer encoder layers
    -> flatten -> 48 horizons x 9 quantiles

Three choices matter more than the layer count:

* Inputs and outputs live in asinh space (asinh(x / 5)), which is linear near
  zero and logarithmic for large values. Quantiles survive a monotone
  transform, so predicting them there and mapping back is exact.
* The network predicts the change from the latest reading, like the tree
  models, so it can follow an episode above anything seen in training.
* Missing hours are forward-filled in the input and flagged in a mask
  channel. Missing targets are left out of the loss.

One global model serves all monitors. A variant adds two input channels: the
mean of neighbours within 300 km, and of those 300 to 800 km away.
"""
import json
import time

import numpy as np
import torch
from torch import nn

from . import config, spatial, store
from .bench import H, L, fill_context

PATCH = 8
SCALE = 5.0
Q = torch.tensor(config.QUANTILES)
VARIANTS = {"Transformer": False, "Transformer + neighbours": True}


def to_z(x):
    """Map concentrations to the asinh scale the network works in."""
    return np.arcsinh(x / SCALE)


def from_z(z):
    """Inverse of to_z."""
    return SCALE * np.sinh(z)


class PatchTransformer(nn.Module):
    """Encoder-only transformer over 8-hour patches, with a flatten head that emits 48 x 9 quantiles."""
    def __init__(self, n_channels, n_cov, d_model=128, n_heads=4, n_layers=3, dropout=0.1):
        """Build the patch embedding, covariate token, encoder stack and output head."""
        super().__init__()
        self.n_patches = L // PATCH
        self.embed = nn.Linear(n_channels * PATCH, d_model)
        self.cov = nn.Sequential(nn.Linear(n_cov, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches + 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 2 * d_model, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear((self.n_patches + 1) * d_model, H * len(Q)))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x, cov, anchor):
        """x [B, C, L], cov [B, n_cov], anchor [B] (latest reading, asinh space) -> [B, H, 9]."""
        b = x.shape[0]
        patches = x.unfold(2, PATCH, PATCH).permute(0, 2, 1, 3).reshape(b, self.n_patches, -1)
        tokens = torch.cat([self.cov(cov).unsqueeze(1), self.embed(patches)], dim=1) + self.pos
        out = self.head(self.norm(self.encoder(tokens)).reshape(b, -1)).view(b, H, len(Q))
        return anchor[:, None, None] + out


def pinball_loss(pred, y, mask):
    """Quantile (pinball) loss averaged over the nine levels and over observed targets only."""
    d = y.unsqueeze(-1) - pred
    loss = torch.maximum(Q.to(pred.device) * d, (Q.to(pred.device) - 1) * d).mean(dim=-1)
    return (loss * mask).sum() / mask.sum().clamp(min=1)


class Windows:
    """Cuts training and inference windows straight out of the panel matrix."""

    def __init__(self, bench, use_spatial):
        """Precompute the transformed, forward-filled input channels and which (monitor, hour) pairs are eligible."""
        self.bench = bench
        z = to_z(bench.values)
        self.target = z                                    # NaN where unobserved
        series = [z]
        if use_spatial:
            series += [to_z(s) for s in spatial.hourly_channels(bench)]
        chans = []
        for s in series:
            ok = np.isfinite(s)
            chans += [np.where(ok.any(axis=1, keepdims=True), np.nan_to_num(fill_context(s)), 0.0), ok]
        self.x = np.stack(chans, axis=1).astype("float32")           # [S, C, T]
        self.n_channels = self.x.shape[1]
        lon = bench.stations["lon"].to_numpy()
        lat = bench.stations["lat"].to_numpy()
        self.static = np.stack([(lat - 38) / 8, (lon + 96) / 20], axis=1).astype("float32")
        self.lon = lon
        self.hour = bench.hours.hour.to_numpy()
        self.doy = bench.hours.dayofyear.to_numpy()
        self.dow = bench.hours.dayofweek.to_numpy()
        self.n_cov = 8
        # Eligibility of every (monitor, hour) as an issue time, from running counts.
        ok = np.isfinite(bench.values)
        c = np.cumsum(ok, axis=1, dtype="int32")
        in_ctx = c.copy()
        in_ctx[:, L:] -= c[:, :-L]
        recent = c.copy()
        recent[:, 3:] -= c[:, :-3]
        self.eligible = (in_ctx >= 0.8 * L) & (recent > 0)

    def pairs(self, split):
        """All eligible (monitor, hour) issue pairs whose targets fall inside a split."""
        lo, hi = self.bench.bounds(split)
        e = self.eligible[:, max(lo, L - 1):hi - H]
        s, t = np.nonzero(e)
        return s.astype("int32"), (t + max(lo, L - 1)).astype("int32")

    def batch(self, s, t, device):
        """Cut inputs, covariates, anchors, targets and target masks for the given pairs."""
        idx = t[:, None] - L + 1 + np.arange(L)[None, :]
        x = self.x[s[:, None, None], np.arange(self.n_channels)[None, :, None], idx[:, None, :]]
        y = self.target[s[:, None], t[:, None] + 1 + np.arange(H)[None, :]]
        solar = (self.hour[t] + self.lon[s] / 15) % 24
        cov = np.stack([np.sin(2 * np.pi * solar / 24), np.cos(2 * np.pi * solar / 24),
                        np.sin(2 * np.pi * self.doy[t] / 365.25), np.cos(2 * np.pi * self.doy[t] / 365.25),
                        np.sin(2 * np.pi * self.dow[t] / 7), np.cos(2 * np.pi * self.dow[t] / 7),
                        self.static[s, 0], self.static[s, 1]], axis=1).astype("float32")
        # Anchor on the latest reading (forward-filled value at the issue time).
        anchor = x[:, 0, -1]
        mask = np.isfinite(y)
        f = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
        return f(x), f(cov), f(anchor), f(np.nan_to_num(y).astype("float32")), f(mask.astype("float32"))


# Model and training size by device. The CPU preset trains in about 15 minutes
# per variant on a laptop; the GPU preset is the larger model.
PRESETS = {
    "cpu": {"d_model": 96, "n_layers": 2, "batch": 256, "steps": 300, "epochs": 25, "patience": 4},
    "cuda": {"d_model": 128, "n_layers": 3, "batch": 512, "steps": 400, "epochs": 30, "patience": 5},
}


def fit(bench, name, log=print, seed=config.SEED, save=True):
    """Train one variant with AdamW, a one-cycle schedule and early stopping; return the best model.

    `seed` sets the weight initialisation and the order of training windows.
    The validation sample is the same for every seed, so that runs differ
    only in what the seed is meant to vary. `save=False` leaves the stored
    weights of the main run untouched.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preset = PRESETS[device]
    epochs, steps, batch, patience = (preset[k] for k in ("epochs", "steps", "batch", "patience"))
    w = Windows(bench, VARIANTS[name])
    model = PatchTransformer(w.n_channels, w.n_cov, preset["d_model"], n_layers=preset["n_layers"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    s_tr, t_tr = w.pairs("train")
    s_va, t_va = w.pairs("valid")
    pick = np.random.default_rng(config.SEED).choice(len(s_va), size=min(40_000, len(s_va)), replace=False)
    s_va, t_va = s_va[pick], t_va[pick]
    log(f"  {name}: {n_params:,} parameters, {len(s_tr):,} candidate training windows, device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * steps, pct_start=0.1)
    best, best_state, bad, history = np.inf, None, 0, []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for _ in range(steps):
            i = rng.integers(0, len(s_tr), batch)
            x, cov, anchor, y, mask = w.batch(s_tr[i], t_tr[i], device)
            loss = pinball_loss(model(x, cov, anchor), y, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()
        model.eval()
        with torch.no_grad():
            num = den = 0.0
            for j in range(0, len(s_va), 4096):
                x, cov, anchor, y, mask = w.batch(s_va[j:j + 4096], t_va[j:j + 4096], device)
                num += pinball_loss(model(x, cov, anchor), y, mask).item() * mask.sum().item()
                den += mask.sum().item()
        val = num / den
        history.append({"epoch": epoch + 1, "train": total / steps, "valid": val})
        log(f"    epoch {epoch + 1:>2}: train {total / steps:.4f}  valid {val:.4f}  ({time.time() - t0:.0f}s)")
        if val < best - 1e-5:
            best, bad = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    if save:
        config.MODELS.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, config.MODELS / f"{name}.pt")
    info = {"parameters": n_params, "best_valid_pinball_asinh": best, "epochs_run": len(history),
            "history": history, "device": device, "train_seconds": time.time() - t0, "use_spatial": VARIANTS[name],
            "preset": preset, "seed": seed}
    return model, w, info


@torch.no_grad()
def predict(bench, model, w, t_idx, batch=4096):
    """Quantile forecasts on the benchmark grid, mapped back to concentrations."""
    device = next(model.parameters()).device
    model.eval()
    elig = bench.eligible(t_idx)
    s, d = np.nonzero(elig)
    t = t_idx[d].astype("int32")
    q = np.full((bench.n_sites, len(t_idx), H, len(config.QUANTILES)), np.nan, dtype="float32")
    for j in range(0, len(s), batch):
        x, cov, anchor, _, _ = w.batch(s[j:j + batch], t[j:j + batch], device)
        q[s[j:j + batch], d[j:j + batch]] = from_z(model(x, cov, anchor).cpu().numpy())
    q = np.sort(q, axis=-1)
    return q[..., len(config.QUANTILES) // 2], q


def run(bench, log=print, names=None):
    """Train, predict and store each transformer variant and record its training history."""
    torch.set_num_threads(8)
    out = {}
    for name in names or VARIANTS:
        model, w, info = fit(bench, name, log)
        point, q = predict(bench, model, w, bench.issue_times("test"))
        store.save(name, point, q, {k: v for k, v in info.items() if k != "history"})
        out[name] = info
    config.TAB.mkdir(parents=True, exist_ok=True)
    path = config.TAB / "neural_training.json"
    prior = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    path.write_text(json.dumps(prior | out, indent=2), encoding="utf-8")
    return out
