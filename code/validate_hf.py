# Validate my numpy GPT-NeoX forward pass (induction.py) against HF transformers on the same weights and tokens.
# Compares final logits, per-layer attention patterns, and the layer-4 residual stream.
# Run: <venv>/bin/python -I validate_hf.py <model dir>
import sys, pathlib, numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
md = sys.argv[1]; sys.argv = [sys.argv[0], md, "1", "10", "none"]
import induction as M
from transformers import GPTNeoXForCausalLM
hf = GPTNeoXForCausalLM.from_pretrained(md, torch_dtype=torch.float32, attn_implementation="eager").eval()
rng = np.random.default_rng(0)
for T in (20, 200):
    ids = rng.integers(1000, 50000, T)
    lg, P = M.forward(ids); x4 = M.forward(ids, resid_at=4)
    with torch.no_grad():
        o = hf(torch.tensor(ids)[None], output_attentions=True, output_hidden_states=True)
    hl = o.logits[0].numpy(); ha = np.stack([a[0].numpy() for a in o.attentions]); h5 = o.hidden_states[5][0].numpy()
    print(f"T={T}: max|logit diff|={np.abs(lg - hl).max():.4f} (logit scale std {hl.std():.1f}); "
          f"argmax agree={(lg.argmax(-1) == hl.argmax(-1)).mean():.3f}; max|attn diff|={np.abs(P - ha).max():.5f}; "
          f"max|resid L4 diff|={np.abs(x4 - h5).max():.4f} (resid norm {np.linalg.norm(h5, axis=1).mean():.1f})")
