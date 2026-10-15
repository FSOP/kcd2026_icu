"""원자료 스크립트가 집계 결과를 내보낼 때 쓰는 함수.

safe_run 으로 실행하면 SAFE_AGG_OUT 경로에 JSON 으로 저장되고, safe_run 이 스키마 검사 후에만 공개한다.
safe_run 없이 직접 실행하면 아무것도 하지 않는다(사람이 직접 보는 기존 print 출력은 그대로).
"""
import json
import os


def emit(**fields):
    path = os.environ.get("SAFE_AGG_OUT")
    if not path:
        return
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    data.update(fields)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, default=float)
