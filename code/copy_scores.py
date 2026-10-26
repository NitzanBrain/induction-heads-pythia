# OV copy score per head (Wang et al. 2022 / McDougall et al. 2023 style) for Pythia-160m:
# effective embedding e_t = E_t + MLP0(LN(E_t)), head output o = LN_in(e_t) W_V W_O, logits = o U^T over a fixed set of
# 1000 common-ish tokens (ids 1000..1999). Copy score = fraction of t whose own logit ranks in the TOP 5% (positive
# copying) minus... reported separately: top-5% share and bottom-5% share. Negative (suppressing) heads: bottom share high.
import sys, json, math, numpy as np
from safetensors.numpy import load_file
md = sys.argv[1]; W = {k: v.astype(np.float32) for k, v in load_file(md + "/model.safetensors").items() if v.dtype != np.uint8}
cfg = json.load(open(md + "/config.json")); D, nh, L = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_hidden_layers"]; hd = D // nh
def ln(x, p):
    m = x.mean(-1, keepdims=True); s = ((x - m) ** 2).mean(-1, keepdims=True); return (x - m) / np.sqrt(s + 1e-5) * W[p + ".weight"] + W[p + ".bias"]
erf = np.vectorize(math.erf, otypes=[np.float32])
ids = np.arange(1000, 2000); E = W["gpt_neox.embed_in.weight"][ids]; p0 = "gpt_neox.layers.0."
f = ln(E, p0 + "post_attention_layernorm") @ W[p0 + "mlp.dense_h_to_4h.weight"].T + W[p0 + "mlp.dense_h_to_4h.bias"]
f = 0.5 * f * (1 + erf(f / np.sqrt(2))); e = E + f @ W[p0 + "mlp.dense_4h_to_h.weight"].T + W[p0 + "mlp.dense_4h_to_h.bias"]
U = W["embed_out.weight"][ids]; n = len(ids); k = n // 20
top = np.zeros((L, nh)); bot = np.zeros((L, nh))
for l in range(L):
    p = f"gpt_neox.layers.{l}."; x = ln(e, p + "input_layernorm")
    Wqkv = W[p + "attention.query_key_value.weight"].reshape(nh, 3 * hd, D); Wo = W[p + "attention.dense.weight"]
    for h in range(nh):
        lg = ((x @ Wqkv[h, 2 * hd:].T) @ Wo[:, h * hd:(h + 1) * hd].T) @ U.T
        r = (lg < lg[np.arange(n), np.arange(n)][:, None]).sum(1)  # rank of own token from the bottom (0 = lowest)
        top[l, h] = (r >= n - k).mean(); bot[l, h] = (r < k).mean()
np.set_printoptions(precision=2, suppress=True, linewidth=150)
print("own-token logit in TOP 5% (copying):\n", top); print("own-token logit in BOTTOM 5% (suppressing):\n", bot)
for l in (4, 5):
    print(f"layer {l}: " + " ".join(f"H{h}:+{top[l, h]:.2f}/-{bot[l, h]:.2f}" for h in range(nh)))
