# Own test of induction heads in Pythia (GPT-NeoX) in numpy. Forward pass validated against HF transformers (validate_hf.py, run14).
# Test (Olsson et al. 2022 style): random tokens of length S, repeated twice. For each head, measure
#   induction score  = mean attention from second-half position i to i-S+1 (the token AFTER the earlier copy),
#   prev-token score = mean attention to i-1 (over all positions >0).
# Then loss on the 1st vs 2nd copy, and the 2nd-copy loss after zero-ablating the top-k induction heads
# (vs k random non-induction heads as a control).
# Run: <venv>/bin/python -I induction.py <model dir> [n_seq] [S] [ablate k list]
import sys, json, struct, math, pathlib
import numpy as np, os
DT = np.float64 if os.environ.get("F64") else np.float32

d = pathlib.Path(sys.argv[1]); cfg = json.load(open(d / "config.json"))
NSEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 8
S = int(sys.argv[3]) if len(sys.argv) > 3 else 100
KS = ([] if sys.argv[4] == "none" else [int(k) for k in sys.argv[4].split(",")]) if len(sys.argv) > 4 else [1, 3, 6, 12]

def load(path):
    f = open(path, "rb"); n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); base = 8 + n
    raw = np.memmap(path, dtype=np.uint8, mode="r"); out = {}
    for k, v in hdr.items():
        if k == "__metadata__" or v["dtype"] == "U8":
            continue
        a, b = v["data_offsets"]; dt = {"F16": np.float16, "F32": np.float32}[v["dtype"]]
        out[k] = np.frombuffer(raw[base + a:base + b], dtype=dt).reshape(v["shape"]).astype(DT)
    return out

W = load(str(d / "model.safetensors"))
D, nh, L = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_hidden_layers"]
hd, eps, theta = D // nh, cfg["layer_norm_eps"], cfg["rotary_emb_base"]
rd = int(hd * cfg["rotary_pct"]); V = cfg["vocab_size"]
E, U = W["gpt_neox.embed_in.weight"], W["embed_out.weight"]
erf = np.vectorize(math.erf, otypes=[np.float32])

def ln(x, p):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * W[p + ".weight"] + W[p + ".bias"]

def rope(x, T):
    inv_f = 1.0 / theta ** (np.arange(0, rd, 2) / rd)
    ang = np.outer(np.arange(T), inv_f); ang = np.concatenate([ang, ang], -1)
    r, keep = x[..., :rd], x[..., rd:]; r1, r2 = r[..., :rd // 2], r[..., rd // 2:]
    return np.concatenate([r * np.cos(ang) + np.concatenate([-r2, r1], -1) * np.sin(ang), keep], -1)

def forward(ids, ablate=(), collect=None, resid_at=None):
    """returns logits [T,V] and attention patterns [L,nh,T,T]; ablate = set of (layer, head) zeroed before dense,
    or dict (layer, head) -> mean vector (mean-ablation); collect = list to receive per-layer mean z."""
    T = len(ids); x = E[ids]; mask = np.triu(np.full((T, T), -1e10, np.float32), 1); pats = []
    for l in range(L):
        p = f"gpt_neox.layers.{l}."
        h = ln(x, p + "input_layernorm")
        qkv = (h @ W[p + "attention.query_key_value.weight"].T + W[p + "attention.query_key_value.bias"])
        qkv = qkv.reshape(T, nh, 3 * hd).transpose(1, 0, 2)
        q, k, v = qkv[..., :hd], qkv[..., hd:2 * hd], qkv[..., 2 * hd:]
        q, k = rope(q, T), rope(k, T)
        s = q @ k.transpose(0, 2, 1) / np.sqrt(hd) + mask
        s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True); pats.append(s)
        z = s @ v
        if collect is not None:
            collect.append(z.mean(1))  # [nh, hd] per-head mean output over positions (for mean-ablation)
        for (al, ah) in ablate:
            if al == l:
                z[ah] = ablate[(al, ah)] if isinstance(ablate, dict) else 0  # dict = mean-ablation vectors
        att = z.transpose(1, 0, 2).reshape(T, D) @ W[p + "attention.dense.weight"].T + W[p + "attention.dense.bias"]
        h2 = ln(x, p + "post_attention_layernorm")
        f = h2 @ W[p + "mlp.dense_h_to_4h.weight"].T + W[p + "mlp.dense_h_to_4h.bias"]
        f = 0.5 * f * (1 + erf(f / np.sqrt(2)))
        x = x + att + f @ W[p + "mlp.dense_4h_to_h.weight"].T + W[p + "mlp.dense_4h_to_h.bias"]
        if resid_at == l:
            return x  # residual stream after block l (used by validate_hf.py)
    return ln(x, "gpt_neox.final_layer_norm") @ U.T, np.stack(pats)

def nll(logits, ids):
    lse = np.logaddexp.reduce(logits[:-1].astype(np.float64), -1)
    return lse - logits[np.arange(len(ids) - 1), ids[1:]]

def metrics(lg, ids):
    """copy-1 median nll; copy-2 (targets S+1..2S-1, i.e. predictable by copying) median nll, mean nll, top-1 acc.
    Medians because Pythia's logits are very spread (std ~80), so a few confident misses dominate means."""
    n = nll(lg, ids); c2 = n[S:]; acc = (lg[S:-1].argmax(-1) == ids[S + 1:]).mean()
    return np.array([np.median(n[:S - 1]), np.median(c2), c2.mean(), acc])

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    seqs = [np.concatenate([r, r]) for r in (rng.integers(1000, 50000, S) for _ in range(NSEQ))]
    ind = np.zeros((L, nh)); prev = np.zeros((L, nh)); base1 = 0.0; base2 = np.zeros(3)
    for ids in seqs:
        lg, P = forward(ids)
        i = np.arange(S, 2 * S); ind += P[:, :, i, i - S + 1].mean(-1)
        j = np.arange(1, 2 * S); prev += P[:, :, j, j - 1].mean(-1)
        m = metrics(lg, ids); base1 += m[0]; base2 += m[1:]
    ind /= NSEQ; prev /= NSEQ; base1 /= NSEQ; base2 /= NSEQ
    np.set_printoptions(precision=2, suppress=True, linewidth=150)
    print(f"model={d.name} n_seq={NSEQ} S={S}")
    print("induction score [layer x head]:\n", ind)
    print("prev-token score [layer x head]:\n", prev)
    order = sorted(((ind[l, h], l, h) for l in range(L) for h in range(nh)), reverse=True)
    print("top induction heads:", [(f"L{l}H{h}", round(float(s), 3)) for s, l, h in order[:12]])
    print("top prev-token heads:", [(f"L{l}H{h}", round(float(prev[l, h]), 3)) for _, l, h in
          sorted(((prev[l, h], l, h) for l in range(L) for h in range(nh)), reverse=True)[:8]])
    print(f"copy1 median nll={base1:.2f} | copy2 median nll={base2[0]:.3f} mean={base2[1]:.2f} top1={base2[2]:.3f} (uniform ln V={math.log(V):.2f})")
    nonind = [(l, h) for s, l, h in order[24:]]
    for K in KS:
        top = [(l, h) for _, l, h in order[:K]]
        ctrl = [nonind[t] for t in rng.choice(len(nonind), K, replace=False)]
        a = c = 0.0
        for ids in seqs:
            a = a + metrics(forward(ids, set(top))[0], ids)[1:]; c = c + metrics(forward(ids, set(ctrl))[0], ids)[1:]
        a, c = a / NSEQ, c / NSEQ
        print(f"ablate top-{K:2d} induction: copy2 median={a[0]:.3f} top1={a[2]:.3f} | {K} random non-induction: median={c[0]:.3f} top1={c[2]:.3f}")
