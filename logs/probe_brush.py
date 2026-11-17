import sys
sys.path.insert(0, "viz"); sys.path.insert(0, ".")
import numpy as np
import selfies as sf
from core.embedding import build_structured_E
from core.fourier_theta import build_fourier_basis, theta_to_Z
from core.vocab import build_vocab_from_semantic_robust_alphabet

ELEMS = {"C", "N", "O", "F", "S", "Cl"}
smiles = "CC(=O)Oc1ccccc1C(=O)O"
L, K, D = 32, 16, 32
vocab = build_vocab_from_semantic_robust_alphabet(allow_elements=set(ELEMS))
tok2id = {t: i for i, t in enumerate(vocab)}
E = build_structured_E(vocab=vocab, D=D, pad_token="[PAD]", seed=7, target_std=0.4, allowed_elements=set(ELEMS))
toks = list(sf.split_selfies(sf.encoder(smiles)))
n = len(toks)
ids = np.array([tok2id[t] for t in toks] + [tok2id["[PAD]"]] * (L - n))
Phi = build_fourier_basis(L, K)
Theta = np.linalg.pinv(Phi) @ E[ids]
M = Phi.shape[1]
kof = np.r_[0, np.arange(1, K + 1), np.arange(1, K + 1)]
rng = np.random.RandomState(0)
print("n", n, "Theta rms", np.sqrt((Theta ** 2).mean()))


def decode_ids(Th):
    Z = theta_to_Z(Th, Phi)
    d2 = ((Z[:, None, :] - E[None, :, :]) ** 2).sum(-1)
    return d2.argmin(1), Z


def tied_ok(Th):
    Z = theta_to_Z(Th, Phi)
    d2 = ((Z[:, None, :] - E[None, :, :]) ** 2).sum(-1)
    return d2[np.arange(L), ids] <= d2.min(1) + 1e-9


def band_noise(lo, hi, z_rms):
    """Gaussian noise on rows lo<=k<=hi, rescaled so that the induced Z perturbation has the given RMS."""
    dTh = np.zeros_like(Theta)
    sel = (kof >= lo) & (kof <= hi) & (np.arange(M) > 0)
    dTh[sel] = rng.normal(size=(sel.sum(), D))
    dZ = theta_to_Z(dTh, Phi)
    return dTh * (z_rms / np.sqrt((dZ[:n] ** 2).mean()))


zr = np.sqrt((theta_to_Z(Theta, Phi)[:n] ** 2).mean())
print("Z rms (aspirin slots)", zr)
for z_rel in (0.3, 0.5, 0.8, 1.2):
    for name, (lo, hi) in {"low k1-4": (1, 4), "mid k5-8": (5, 8), "high k9-16": (9, 16)}.items():
        flips, runs, spans = [], [], []
        for _ in range(300):
            ok = tied_ok(Theta + band_noise(lo, hi, z_rel * zr))[:n]
            bad = np.where(~ok)[0]
            flips.append(len(bad))
            if len(bad) > 1:
                gaps = np.diff(bad)
                runs.append((gaps == 1).mean())    # fraction of flipped slots that touch another flipped slot
        print(f"Zrel {z_rel:.1f} {name:11s} flipped slots mean {np.mean(flips):4.1f}  adjacency {np.mean(runs) if runs else float('nan'):.2f}")
