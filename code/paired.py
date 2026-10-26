# Paired comparison of two icl2 conditions on the SAME windows: mean of (benefit_A − benefit_B) with a bootstrap CI.
# Run: python3 -I paired.py a.npy b.npy   (uses numpy from the venv)
import sys, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2]); d = a - b; rng = np.random.default_rng(0)
bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
print(f"{sys.argv[1]} − {sys.argv[2]}: {d.mean():+.4f} [95% CI {np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}] "
      f"(as share of B: {d.mean() / b.mean():+.0%}); windows where A<B: {(d < 0).sum()}/{len(d)}")
