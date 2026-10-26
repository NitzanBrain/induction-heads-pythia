#!/bin/bash
# Checkpoint sweep: induction/prev-token scores + copy-2 loss across Pythia-160m training steps.
# Usage: sweep.sh <venv python> <work dir> step1 step2 ...   (downloads each checkpoint, measures, deletes it)
PY=$1; WD=$2; shift 2
for st in "$@"; do
  dir=$WD/ck-step$st; mkdir -p $dir
  for f in config.json tokenizer.json; do cp $WD/p160/$f $dir/; done
  curl -sL -o $dir/model.safetensors https://huggingface.co/EleutherAI/pythia-160m/resolve/step$st/model.safetensors
  echo "=== step $st ==="; $PY -I induction.py $dir 8 100 none | grep -v "^ \[\|^\[\|^ \["
  rm -f $dir/model.safetensors
done
