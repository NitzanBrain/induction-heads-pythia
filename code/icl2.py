# Paired ICL test, to remove the content confound of "loss@50 − loss@500" (different tokens at the two positions).
# For each 512-token window: targets = the last 62 tokens (positions 450..511).
#   A (own context): the real 450 preceding tokens.
#   B (foreign far context): the first 400 tokens replaced by 400 tokens from a distant part of the book;
#     the 50 tokens right before the targets stay real.
# Same targets, same local context, so ICL benefit = mean over targets of loss_B − loss_A (paired; bootstrap CI over windows).
# Run: <venv>/bin/python -I icl2.py <model dir> <text file> [n_windows]
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
md, tf = sys.argv[1], sys.argv[2]; NW = int(sys.argv[3]) if len(sys.argv) > 3 else 24
src = open(pathlib.Path(__file__).parent / "icl.py").read()
sys.argv = [sys.argv[0], md, "1", "10", "none"]
import json, re
import induction as M
exec(src[src.index("tj = json.load"):src.index("starts =")])   # same tokenizer + text cleaning as icl.py
import os  # optional ablation, e.g. ABL="3,2;3,0" zeroes heads L3H2 and L3H0
AB = {tuple(map(int, x.split(","))) for x in os.environ.get("ABL", "").split(";") if x}
starts = np.linspace(0, len(ids) - 513, NW).astype(int)
if AB and os.environ.get("MEAN"):  # mean-ablation: each head's mean output over 8 reference windows (chosen apart from the target starts, but they may overlap target windows)
    acc = []
    for r0 in np.linspace(256, len(ids) - 769, 8).astype(int):
        col = []; M.forward(np.array(ids[r0:r0 + 512]), (), col); acc.append(np.stack(col))
    mz = np.mean(acc, 0); AB = {k: mz[k[0], k[1]] for k in AB}
diffs, la, lb = [], [], []
for w, s0 in enumerate(starts):
    own = np.array(ids[s0:s0 + 512])
    f0 = starts[(w + NW // 2) % NW]; foreign = np.array(ids[f0:f0 + 400])
    mix = own.copy(); mix[:400] = foreign
    a = M.nll(M.forward(own, AB)[0], own)[449:]; b = M.nll(M.forward(mix, AB)[0], mix)[449:]
    la.append(a.mean()); lb.append(b.mean()); diffs.append((b - a).mean())
d = np.array(diffs); rng = np.random.default_rng(0)
if os.environ.get("DUMP"):  # per-window benefits, for paired comparisons between conditions (same windows)
    np.save(os.environ["DUMP"], d)
bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
print(f"model={pathlib.Path(md).name} ablate={sorted(AB)}{" MEAN" if isinstance(AB, dict) else ""} windows={NW} loss own={np.mean(la):.3f} foreign={np.mean(lb):.3f} "
      f"ICL benefit={d.mean():.3f} nats [95% CI {np.quantile(bs, .025):.3f}, {np.quantile(bs, .975):.3f}] median window diff={np.median(d):.3f}")
