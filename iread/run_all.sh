#!/usr/bin/env bash
# iREAD 재현 전체 실행. 결과(집계값)는 logs/ 에 저장됨.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
mkdir -p logs

$PY 01_extract.py      2>&1 | tee logs/01_extract.log
$PY 02_cohort.py       2>&1 | tee logs/02_cohort.log
for v in A B; do
  $PY 03_features.py $v 2>&1 | tee logs/03_features_$v.log
  $PY 04_train.py $v    2>&1 | tee logs/04_train_$v.log
done
echo "완료. 결과 표: ~/KCD2026/work/results_A.csv, results_B.csv"
