# Causal test of K-composition: zero-ablate the previous-token heads and see whether the downstream induction
# heads lose their induction attention pattern (they should, if their keys read the prev-token heads' output).
# Control: ablate the same number of other heads from the same layer with low prev-token scores.
# Run: <venv>/bin/python -I kcomp.py <model dir>
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.argv = [sys.argv[0], sys.argv[1], "8", "100", "none"]
import induction as M
S, L, nh = M.S, M.L, M.nh
rng = np.random.default_rng(1)
seqs = [np.concatenate([r, r]) for r in (rng.integers(1000, 50000, S) for _ in range(8))]
IND = [(4, 6), (4, 10), (4, 11), (4, 8), (5, 6), (5, 0), (8, 2), (9, 8)]
conds = {"none": set(), "prev L3H2+L3H0": {(3, 2), (3, 0)}, "ctrl L3H3+L3H5": {(3, 3), (3, 5)},
         "prev L3H2 only": {(3, 2)}, "prev L3H0 only": {(3, 0)},
         "prev L3H2+L3H0+L2H8+L3H1": {(3, 2), (3, 0), (2, 8), (3, 1)}, "ctrl L3H3+L3H5+L2H0+L3H6": {(3, 3), (3, 5), (2, 0), (3, 6)},
         "prev L7H7 (late)": {(7, 7)}, "ctrl L7H0": {(7, 0)}}
i = np.arange(S, 2 * S)
print("induction score of each head under ablation; copy2 top1")
print(f"{'condition':28s} " + " ".join(f"L{l}H{h:<3d}" for l, h in IND) + "  top1")
for name, ab in conds.items():
    sc = np.zeros(len(IND)); acc = 0
    for ids in seqs:
        lg, P = M.forward(ids, ab)
        sc += [P[l, h, i, i - S + 1].mean() for l, h in IND]; acc += M.metrics(lg, ids)[3]
    sc /= len(seqs)
    print(f"{name:28s} " + " ".join(f"{x:6.3f}" for x in sc) + f"  {acc / len(seqs):.3f}")
