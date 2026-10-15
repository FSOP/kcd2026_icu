"""B안 안전 래퍼: 원자료 스크립트를 실행하되, LLM 에게는 검사된 요약만 돌려준다.

사용법:  python safe/safe_run.py STEP script.py [args...]
  STEP 은 schemas.SCHEMAS 의 키 (집계 결과 스키마 선택)

동작
1. 자식 프로세스의 stdout/stderr(파이썬 print, 트레이스백, 네이티브 fd 출력, 손자 프로세스 출력 포함)를
   비공개 로그(디렉터리 0700, 파일 0600)에만 기록한다. 로그를 못 열면 실행하지 않는다(fail-closed).
2. 이 프로그램의 화면 출력(= LLM 이 보는 것)은 JSON 한 줄뿐이다:
   단계, 성공/실패, 종료 코드, 예외 종류, 코드 위치(파일:줄), 소요 시간, 스키마를 통과한 집계 결과.
   예외 메시지 본문은 어떤 경우에도 넣지 않는다.
3. 집계 결과는 schemas 의 allowlist 를 통과해야 공개된다. 위반이 있으면 결과 전체를 보류하고
   위반 '종류'만 알린다(위반한 값이나 모르는 필드 이름은 알리지 않음).

한계: 같은 OS 계정에서 동작하는 절차적 분리다. LLM 이 비공개 로그 파일을 직접 열지 않는다는 운영 규칙에 의존한다.
"""
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schemas import MIN_CELL, SCHEMAS  # noqa: E402

HERE = Path(__file__).resolve().parent
LOG_DIR = Path(os.environ.get("SAFE_LOG_DIR", Path.home() / "KCD2026/private_logs"))
IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
PATHPART = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")


class Reject(Exception):
    """스키마 위반. 메시지는 미리 정한 위반 종류 코드만 담는다."""


# ---------------------------------------------------------------- 집계 검사
def check(value, spec):
    kind = spec[0]
    if kind == "count":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise Reject("bad_type")
        if value != int(value) or value < 0:
            raise Reject("bad_count")
        value = int(value)
        return f"<{MIN_CELL}" if 0 < value < MIN_CELL else value
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 1000:
            raise Reject("bad_type")
        return value
    if kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise Reject("bad_type")
        if not spec[1] <= value <= spec[2]:
            raise Reject("out_of_range")
        return round(float(value), 4)
    if kind == "enum":
        if not isinstance(value, str) or value not in spec[1]:
            raise Reject("bad_enum")
        return value
    if kind == "table":
        if not isinstance(value, dict):
            raise Reject("bad_type")
        if not set(value) <= spec[1]:
            raise Reject("unknown_key")
        return {k: check(v, spec[2]) for k, v in value.items()}
    if kind == "record":
        if not isinstance(value, dict):
            raise Reject("bad_type")
        if not set(value) <= set(spec[1]):
            raise Reject("unknown_field")
        return {k: check(v, spec[1][k]) for k, v in value.items()}
    raise Reject("bad_schema")


def validate(step, data):
    schema = SCHEMAS.get(step)
    if schema is None:
        raise Reject("unknown_step")
    if not isinstance(data, dict):
        raise Reject("bad_type")
    if not set(data) <= set(schema):
        raise Reject("unknown_field")
    return {k: check(v, schema[k]) for k, v in data.items()}


# ---------------------------------------------------------------- 실행 상태 정리
def where(frames, code_dir):
    """트레이스백 프레임 중 우리 코드 안의 마지막 위치만 'file:line' 으로. 그 밖은 공개하지 않음."""
    code_dir = Path(code_dir).resolve()
    for fn, line in reversed(frames or []):
        try:
            p = Path(fn).resolve()
            rel = p.relative_to(code_dir)
        except (ValueError, OSError, TypeError):
            continue
        if isinstance(line, int) and all(PATHPART.match(s) for s in rel.parts):
            return f"{rel.as_posix()}:{line}"
    return "external"


def open_private_log(step):
    """0700 디렉터리에 0600 로그 파일 생성. 심볼릭 링크·남의 소유·권한 이상이면 거부."""
    LOG_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    st = os.lstat(LOG_DIR)
    if not os.path.isdir(LOG_DIR) or os.path.islink(LOG_DIR) or st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise PermissionError
    name = f"{datetime.now():%Y%m%d-%H%M%S-%f}_{step}.log"
    fd = os.open(LOG_DIR / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    return fd, name


def run(step, script, args, code_dir, timeout=None):
    out = {"step": step if IDENT.match(step) else "invalid"}
    if out["step"] == "invalid" or step not in SCHEMAS:
        out.update(ok=False, error="unknown_step")
        return out
    try:
        log_fd, log_name = open_private_log(step)
    except OSError:
        out.update(ok=False, error="log_unavailable")        # fail-closed: 실행하지 않음
        return out
    out["log"] = log_name

    with tempfile.TemporaryDirectory(prefix="safe_run_") as tmp:
        os.chmod(tmp, 0o700)
        status_path, agg_path = os.path.join(tmp, "status.json"), os.path.join(tmp, "agg.json")
        env = {**os.environ, "SAFE_AGG_OUT": agg_path, "PYTHONUNBUFFERED": "1"}
        t0 = time.time()
        try:
            rc = subprocess.run([sys.executable, str(HERE / "_launch.py"), status_path, script, *args],
                                stdin=subprocess.DEVNULL, stdout=log_fd, stderr=log_fd,
                                env=env, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = None
        finally:
            os.close(log_fd)
        out["elapsed_s"] = int(time.time() - t0)
        out["exit"] = rc if isinstance(rc, int) else "timeout"

        try:
            with open(status_path) as f:
                st = json.load(f)
        except (OSError, ValueError):
            st = {"ok": False, "exc": "NoStatus", "frames": []}
        ok = bool(st.get("ok")) and rc == 0
        out["ok"] = ok
        if not ok:
            exc = st.get("exc")
            out["exception"] = exc if isinstance(exc, str) and IDENT.match(exc) else "Unknown"
            out["where"] = where(st.get("frames"), code_dir)

        if os.path.exists(agg_path):
            try:
                with open(agg_path) as f:
                    out["result"] = validate(step, json.load(f))
            except Reject as r:
                out["result_withheld"] = str(r.args[0])
            except Exception:                              # noqa: BLE001  검사기 자체 오류도 값 노출 없이 보류
                out["result_withheld"] = "unreadable"
    return out


def main(argv):
    if len(argv) < 2:
        print(json.dumps({"ok": False, "error": "usage"}))
        return 2
    code_dir = os.environ.get("SAFE_CODE_DIR", str(HERE.parent))
    try:
        out = run(argv[0], argv[1], argv[2:], code_dir)
    except Exception:                                      # noqa: BLE001  래퍼 내부 오류도 트레이스백 없이
        out = {"ok": False, "error": "wrapper_error"}
    print(json.dumps(out, ensure_ascii=False))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
