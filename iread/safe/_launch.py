"""safe_run 이 띄우는 자식 프로세스 진입점. 대상 스크립트를 실행하고 예외의 '종류와 위치'만 상태 파일에 남긴다.

예외 메시지·트레이스백 원문은 stderr(= 비공개 로그)로만 나간다.
"""
import json
import os
import runpy
import sys
import traceback

status_path, script, *args = sys.argv[1:]
sys.argv = [script, *args]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))

status = {"ok": True}
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as e:
    if not (e.code is None or e.code == 0):
        if not isinstance(e.code, int):
            print(e.code, file=sys.stderr)          # 문자열 종료 메시지는 비공개 로그로만
        status = {"ok": False, "exc": "SystemExit", "frames": []}
except BaseException as e:                           # noqa: BLE001
    traceback.print_exc()                            # 원문 -> 비공개 로그
    status = {"ok": False, "exc": type(e).__name__,
              "frames": [[f.filename, f.lineno] for f in traceback.extract_tb(e.__traceback__)]}
finally:
    sys.stdout.flush(); sys.stderr.flush()
    with open(status_path, "w") as f:
        json.dump(status, f)
sys.exit(0 if status["ok"] else 1)
