# How the results were produced

**Repository layout:** scripts are in `code/`, raw outputs in `results/`, and the per-window arrays in `results/dumps/`. Run the scripts from `code/`, which `icl2.py` needs in order to import `induction.py`. The paired and shift tests read `dumps/...` relative paths, so run them from `results/`, e.g. `cd results && python ../code/shift_test.py 0`.
 (reconstructed 2026-10-26 from the scripts, raw outputs and my work log)

The exact command lines for the 2026-10-23/24 runs were not saved, so these are reconstructions. The head sets below are the ones named in RESULTS.md §6–§8 and in run16-paired.txt. A rerun with the same model revision and text should give the same numbers, because the forward pass is deterministic numpy. **Verified 2026-10-26:** rerunning `icl2.py` on the final model (revision 50f5173d) with the pinned text reproduced `dumps/fin-none.npy` exactly: 24/24 windows identical, max abs difference 0.0, benefit 0.101 nats. That took 6 min 19 s wall clock at 1.6 GB peak RAM. The other conditions were not rerun.

## Model and text
- **Model:** EleutherAI/pythia-160m from Hugging Face.
  - Intermediate checkpoints: `resolve/step<N>/model.safetensors`.
  - Final model: `main` = revision `50f5173d932e8e61f858120bcb800b97af589f46`. It is unchanged on the Hub since 2023-07-09, and the exact rerun confirms it (see above).
- **Text:** *Pride and Prejudice*, Project Gutenberg eBook #1342, plain UTF-8, cleaned as in `icl.py`. The original copy was in /tmp and is gone. The current Gutenberg file has "Chapter I." twice (table of contents plus body), so where the cleaning starts the text may differ from the 10-23 run. The copy that reproduces the results exactly has sha256 `3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75` (gutenberg.org/cache/epub/1342/pg1342.txt, downloaded 2026-10-25).

## Scripts and outputs
| output | script | how |
|---|---|---|
| run1, run2, run4 (induction/prev-token scores per checkpoint) | `induction.py`, `sweep.sh` | `sweep.sh <python> <workdir> <steps…>` |
| run3 (K-composition ablations) | `kcomp.py` | |
| run5 (Olsson-style ICL score) | `icl.py`, `sweep_icl.sh` | |
| run6–run11 (paired ICL benefit, sweeps, ablations, layer scan) | `icl2.py`, `sweep_icl2.sh` | `[ABL="l,h;l,h"] [MEAN=1] [DUMP=dumps/x.npy] python -I icl2.py <model dir> <text> 24`. The layer scan (run11) set ABL to all 12 heads of one layer at a time; no separate script was saved. |
| run12, run13, run16 (paired differences) | `paired.py` | `python -I paired.py dumps/A.npy dumps/B.npy` |
| run14 (check against HF transformers) | `validate_hf.py` | |
| run17 (copy scores / mechanism) | `copy_scores.py` | |
| shift-test.txt | `shift_test.py` | `python -I shift_test.py [seed]` (added 2026-10-26; reproduces the 10-24 line) |
| (no run15) | | The number was skipped. No run15 exists. |

## Dump map (dumps/<ckpt>-<condition>.npy; ckpt = s1k for step 1000, fin for step 143000; zero-ablation)
| condition | ABL heads (layer,head) |
|---|---|
| none | – |
| w1 | 4,6;4,8;4,10;4,11;5,0;5,6 (the 6 wave-1 induction heads) |
| r6 | 0,1;1,5;2,3;6,4;10,1;11,9 (one fixed set of 6 low-score heads) |
| L3H2 | 3,2 |
| L4 / L5 | all 12 heads of layer 4 / layer 5 |
| L4four | 4,6;4,8;4,10;4,11 |
| L4rest | the other 8 heads of layer 4 |
| L5two | 5,0;5,6 |
| L5rest | the other 10 heads of layer 5 |

## Environment
Python 3.12 with numpy (see requirements.txt), plus `safetensors` to read the checkpoints. `validate_hf.py` additionally needs torch and transformers. Everything runs on CPU.
