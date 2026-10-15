#!/usr/bin/env bash
# 내부 열람용 둘러보기 — 실제 MIMIC-IV 로 생성 (이슈 ad40b868). 소유자가 자기 터미널에서 직접 실행한다.
#   bash ~/KCD2026/iread/internal/run_browse.sh
# 출력: ~/KCD2026/private_reports/browse_<시각>/index.html (+ keymap.csv, meta.json), 권한 0700/0600
# 화면에는 성공/실패 한 줄만. 상세 오류·traceback 은 ~/KCD2026/private_logs/browse_<시각>.log 에만.
# 주의: 아래 TTY 확인은 '실수로 에이전트가 실행하는 것'을 막는 절차적 장치일 뿐, 보안 경계가 아니다.
set -uo pipefail
if [ ! -t 0 ] || [ ! -t 1 ]; then
  echo "중단: 대화형 터미널에서 소유자가 직접 실행해야 합니다."; exit 2
fi
read -r -p "실제 환자 기록으로 내부 보고서를 만듭니다. 결과를 보드·GitHub·외부 LLM 에 올리지 않겠습니다 (yes 입력): " ans
[ "$ans" = "yes" ] || { echo "취소"; exit 1; }
K="$HOME/KCD2026"; TS=$(date +%Y%m%d-%H%M%S)
umask 077
mkdir -p "$K/private_reports" "$K/private_logs"; chmod 700 "$K/private_reports" "$K/private_logs"
"$K/.venv/bin/python" "$K/iread/internal/browse_gen.py" \
  --data "$K/datasets/mimic-iv/physionet.org/files/mimiciv/3.1" --work "$K/work" \
  --out "$K/private_reports/browse_$TS" --log "$K/private_logs/browse_$TS.log" --label "실제 자료 · 내부 전용" "$@" \
  2>>"$K/private_logs/browse_$TS.log"
