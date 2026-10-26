# Shift test (RESULTS §12): does the share of the natural-text context benefit removed by the 6 wave-1 induction heads
# differ between step 1000 and the final model? Inputs: per-window context benefits saved by icl2.py with DUMP set
# (dumps/<ckpt>-<condition>.npy, one value per window, same 24 windows for every condition).
# share = 1 - mean(ablated) / mean(unablated); CI from a joint bootstrap over windows (same resample for both checkpoints).
# Usage: python3 shift_test.py [seed]   -> shift-test.txt line. The lower CI bound moves by ~0.02 with the seed.
import sys
import numpy as np

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
d = {k: np.load(f"dumps/{k}.npy") for k in ("s1k-none", "s1k-w1", "fin-none", "fin-w1")}
n = len(d["fin-none"])
assert all(len(v) == n for v in d.values())


def share(none, abl, idx):
    return 1 - abl[idx].mean() / none[idx].mean()


rng = np.random.default_rng(seed)
draws = rng.integers(0, n, size=(5000, n))
s1 = np.array([share(d["s1k-none"], d["s1k-w1"], i) for i in draws])
s2 = np.array([share(d["fin-none"], d["fin-w1"], i) for i in draws])
ci = lambda x: np.percentile(x, [2.5, 97.5])
full = np.arange(n)
p1, p2 = share(d["s1k-none"], d["s1k-w1"], full), share(d["fin-none"], d["fin-w1"], full)
print(f"share removed by w1 (point estimate, bootstrap 95% CI, seed {seed}): "
      f"step1000 {p1:.2f} [{ci(s1)[0]:.2f},{ci(s1)[1]:.2f}]; final {p2:.2f} [{ci(s2)[0]:.2f},{ci(s2)[1]:.2f}]; "
      f"difference {p1 - p2:.2f} [{ci(s1 - s2)[0]:.2f},{ci(s1 - s2)[1]:.2f}] (n={n} windows, joint bootstrap, 5000 draws)")
