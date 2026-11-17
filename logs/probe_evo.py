import sys
sys.path.insert(0, "viz"); sys.path.insert(0, ".")
import numpy as np
import selfies as sf
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from core.embedding import build_structured_E
from core.fourier_theta import build_fourier_basis, theta_to_Z
from core.vocab import build_vocab_from_semantic_robust_alphabet
RDLogger.DisableLog("rdApp.*")

ELEMS = {"C", "N", "O", "F", "S", "Cl"}
L, K, D = 32, 16, 32
vocab = build_vocab_from_semantic_robust_alphabet(allow_elements=set(ELEMS))
E = build_structured_E(vocab=vocab, D=D, pad_token="[PAD]", seed=7, target_std=0.4, allowed_elements=set(ELEMS))
pad = vocab.index("[PAD]")
Phi = build_fourier_basis(L, K); M = Phi.shape[1]
RMS = 0.0705

def decode(Th):
    Z = Phi @ Th
    d2 = ((Z[:, None, :] - E[None, :, :]) ** 2).sum(-1)
    d2[:12, pad] = np.inf          # simplified MIN_TOKENS_BEFORE_PAD: no PAD in the first 12 slots
    ids = d2.argmin(1)
    toks = []
    for i in ids:
        if i == pad:
            break
        toks.append(vocab[i])
    try:
        return sf.decoder("".join(toks))
    except Exception:
        return ""

def fp(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m is not None and m.GetNumAtoms() > 0 else None

target = sys.argv[1] if len(sys.argv) > 1 else "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
tfp = fp(target)
def fit(smi):
    f = fp(smi)
    return 0.0 if f is None else DataStructs.TanimotoSimilarity(f, tfp)

def mutate(Th, gen, G, rng, sig_rel=0.45):
    Y = Th.copy()
    sigma = sig_rel * RMS * (1 - 0.2 * gen / max(1, G - 1))
    mask = rng.rand(*Y.shape) < 0.8
    Y[mask] += rng.normal(0, sigma, mask.sum())
    rr = rng.rand(M) < 0.09
    Y[rr] = rng.normal(0, RMS, (rr.sum(), D))
    return Y

for seed in (1, 2, 3):
    rng = np.random.RandomState(seed)
    N, G = 40, 30
    pop = [rng.normal(0, RMS, (M, D)) for _ in range(N)]
    hist = []
    for g in range(G):
        smis = [decode(t) for t in pop]
        f = np.array([fit(s) for s in smis])
        order = np.argsort(-f)
        hist.append((f.max(), f.mean(), smis[order[0]]))
        elite = [pop[i] for i in order[: N // 4]]
        kids = []
        while len(elite) + len(kids) < N:
            a = rng.choice(len(elite))
            kids.append(mutate(elite[a], g, G, rng))
        pop = elite + kids
    print("seed", seed, "best/mean by gen:", [(round(h[0], 2), round(h[1], 2)) for h in hist[::5]], hist[-1][2])
