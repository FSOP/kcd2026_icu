#!/usr/bin/env bash
# B안: 원자료 파이프라인을 안전 래퍼로 실행. 화면에는 단계별 요약 JSON 한 줄씩만 나온다.
# 원문 출력·에러는 ~/KCD2026/private_logs/ 에만 저장 (사람만 열람, LLM 은 열지 않음).
# 실행: bash iread/run_safe.sh            (전체)
#       bash iread/run_safe.sh 03 A       (한 단계만: 01 | 02 | 03 A|B | 04 A|B [d1|d2|d3] | 04t A|B [d1|d2|d3] | 05 진단 | 06 A [d] | 08 A [d])
set -uo pipefail
cd "$(dirname "$0")"
PY="$(cd .. && pwd)/.venv/bin/python"   # 심볼릭 링크를 풀면 venv 가 아니게 되므로 realpath 금지

step() {   # step <스키마> <스크립트> [인자]   실패하면 중단
  "$PY" safe/safe_run.py "$@" || { echo '{"pipeline": "stopped"}'; exit 1; }
}

case "${1:-all}" in
  01) step extract 01_extract.py ;;
  02) step cohort 02_cohort.py ;;
  03) step features 03_features.py "${2:?A|B}" ;;
  04) step train 04_train.py "${2:?A|B}" "${3:-d1}" ;;   # 3번째 인자: 결과 정의 d1(주분석) | d2(논문) | d3(재입실만)
  04t) step tune 04_train.py "${2:?A|B}" "${3:-d1}" tune ;;   # holdout 튜닝 (test 미사용)
  08) step calib 08_calib.py "${2:?A|B}" "${3:-d1}" ;;
  09) step w6 09_w6.py "${2:?A|B}" "${3:-d1}" ;;
  10) step direct 10_direct_readm.py "${2:?A|B}" "${3:-d1}" ;;
  11) step disposition 11_disposition.py ;;   # ICU 퇴실 직후 위치·직접 퇴원 집계 (재학습 없음)
  12) step ddsens 12_dd_sensitivity.py "${2:?A|B}" "${3:-d1}" ;;   # 바로 퇴원 제외 민감도 (저장 예측 재집계)
  13) step table1 13_table1.py ;;   # 조원 공유용 기술통계 (B·①, 집계만)
  14) step shortreadm 14_short_readm.py ;;   # 짧은 간격 재입실 성격 집계 (재학습 없음)
  05) step diag 05_diag.py ;;
  06) step shap 06_shap.py "${2:?A|B}" "${3:-d1}" ;;
  all)   # 데이터 준비(01~03)만. 04 는 tune -> final 순서로 정의별로 명시 실행 (N6: test 1회 원칙)
    step extract 01_extract.py
    step cohort 02_cohort.py
    for v in A B; do
      step features 03_features.py "$v"
    done ;;
  *) echo '{"error": "usage"}'; exit 2 ;;
esac
