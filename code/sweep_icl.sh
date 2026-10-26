#!/bin/bash
# ICL score (loss@50 − loss@500 on natural text) across Pythia-160m checkpoints. Usage: sweep_icl.sh <python> <work dir> <text> steps...
PY=$1; WD=$2; TX=$3; shift 3
for st in "$@"; do
  dir=$WD/ck-step$st; mkdir -p $dir; cp $WD/p160/config.json $WD/p160/tokenizer.json $dir/
  if [ "$st" = "143000" ]; then dir=$WD/p160; else
    curl -sL -o $dir/model.safetensors https://huggingface.co/EleutherAI/pythia-160m/resolve/step$st/model.safetensors; fi
  echo "step $st: $($PY -I icl.py $dir $TX 16 | tail -1)"
  [ "$st" != "143000" ] && rm -f $dir/model.safetensors
done
