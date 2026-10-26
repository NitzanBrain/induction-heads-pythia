# Olsson et al. 2022 "in-context learning score" on natural text: loss at token 50 minus loss at token 500
# (512-token contexts), here as the median over contexts and over a ±5-token window around each index (my choice, for noise).
# Text: Pride and Prejudice (Project Gutenberg #1342, public domain), body only.
# Run: <venv>/bin/python -I icl.py <model dir> <text file> [n_ctx]
import sys, pathlib, json, re, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
md, tf = sys.argv[1], sys.argv[2]; NC = int(sys.argv[3]) if len(sys.argv) > 3 else 12
sys.argv = [sys.argv[0], md, "1", "10", "none"]
import induction as M

tj = json.load(open(pathlib.Path(md) / "tokenizer.json")); vocab = tj["model"]["vocab"]
merges = {tuple(m.split(" ") if isinstance(m, str) else m): i for i, m in enumerate(tj["model"]["merges"])}
bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)); rest = [i for i in range(256) if i not in bs]
b2u = {b: chr(c) for b, c in zip(bs + rest, bs + [256 + n for n in range(len(rest))])}
GPT2 = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""")
def bpe(w):
    p = list(w)
    while len(p) > 1:
        c = [(merges.get((p[i], p[i + 1])), i) for i in range(len(p) - 1)]; c = [x for x in c if x[0] is not None]
        if not c: break
        _, i = min(c); p[i:i + 2] = [p[i] + p[i + 1]]
    return [vocab[x] for x in p]
text = open(tf, encoding="utf-8").read().replace("\r\n", "\n")
# Gutenberg's indentation makes long runs of spaces; Pythia gives those ~800-nat losses (a large logit offset on whitespace
# tokens), which swamp everything. So collapse spaces and keep single line breaks between paragraphs.
text = re.sub(r"[ \t]+", " ", text); text = re.sub(r"\n\s*\n+", "\n\n", text)
text = text[text.index("Chapter I.") if "Chapter I." in text else 5000:][:400000]
ids = []
for w in GPT2.findall(text):
    ids += bpe("".join(b2u[b] for b in w.encode("utf-8")))
    if len(ids) > 200000: break
starts = np.linspace(0, len(ids) - 513, NC).astype(int)
L50, L500 = [], []
for s0 in starts:
    c = np.array(ids[s0:s0 + 512]); n = M.nll(M.forward(c)[0], c)   # n[t] = loss of predicting token t+1
    L50 += list(n[44:55]); L500 += list(n[494:505])
a, b = np.median(L50), np.median(L500)
print(f"model={pathlib.Path(md).name} n_ctx={NC} median loss@50={a:.3f} @500={b:.3f} ICL score={a - b:.3f} | means {np.mean(L50):.2f} {np.mean(L500):.2f}")
