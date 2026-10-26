#!/bin/bash
# Paired ICL benefit (icl2.py: own vs foreign far context, 24 windows) across Pythia-160m checkpoints. Usage: sweep_icl2.sh <python> <work dir> <text> steps...
PY=$1; WD=$2; TX=$3; shift 3
for st in "$@"; do
  dir=$WD/ck-step$st; mkdir -p $dir; cp $WD/p160/config.json $WD/p160/tokenizer.json $dir/
  if [ "$st" = "143000" ]; then dir=$WD/p160; else
    curl -sL -o $dir/model.safetensors https://huggingface.co/EleutherAI/pythia-160m/resolve/step$st/model.safetensors; fi
  echo "step $st: $($PY -I icl2.py $dir $TX 24 | tail -1)"
  [ "$st" != "143000" ] && rm -f $dir/model.safetensors
done
