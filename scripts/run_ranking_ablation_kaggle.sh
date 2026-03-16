#!/usr/bin/env bash
set -uo pipefail
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore:invalid value encountered in divide:RuntimeWarning"

cd /kaggle/working
rm -rf MaxFlow
git clone https://github.com/DragX0826/MaxFlow.git
cd MaxFlow
git checkout 53c7c1d

pip install -q posebusters biopython fair-esm

TARGETS="1gpk,1b8o,1aq1,1eve,1fc0,1f0r"
SEEDS="42,43,44"

run_cfg () {
  local NAME="$1"
  local SCORE="$2"

  echo "=== RUN ${NAME} | selection_score=${SCORE} ==="

  python -u src/run_benchmark.py \
    --targets "${TARGETS}" \
    --seeds "${SEEDS}" \
    --steps 300 \
    --batch_size 16 \
    --num_gpus 2 \
    --kaggle \
    --fksmc \
    --socm \
    --amp \
    --compile_backbone \
    --mmff_snap_fraction 0.25 \
    --final_mmff_topk 5 \
    --final_mmff_max_iter 1000 \
    --adaptive_stop_thresh 0.02 \
    --adaptive_min_step_frac 0.70 \
    --adaptive_patience_frac 0.15 \
    --rerank_polish_mult 2 \
    --selection_score "${SCORE}" \
    --no_target_plots \
    --no_aggregate_figures \
    --no_pose_dump \
    --quiet \
    --output_dir "/kaggle/working/results/${NAME}"

  STATUS=$?
  echo "=== DONE ${NAME} exit=${STATUS} ==="
  return 0
}

run_cfg ranking_energy energy
run_cfg ranking_clash clash

python -u src/paper_metrics.py \
  --run ranking_energy="/kaggle/working/results/ranking_energy/**/benchmark_results.csv" \
  --run ranking_clash="/kaggle/working/results/ranking_clash/**/benchmark_results.csv" \
  --targets "${TARGETS}" \
  --output_dir /kaggle/working/results/ranking_ablation_tables

cd /kaggle/working
zip -r ranking_ablation_results.zip results/
echo "Done: /kaggle/working/ranking_ablation_results.zip"
