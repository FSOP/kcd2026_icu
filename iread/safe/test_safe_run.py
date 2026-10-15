"""B안 안전 래퍼 누수 시험 (가상 데이터만).

실행: ../.venv/bin/python -m unittest safe.test_safe_run -v   (iread/ 에서)

각 시험은 가짜 '원자료 스크립트'가 일부러 가짜 환자값(CANARY)을 여러 경로로 흘리게 한 뒤,
safe_run 의 화면 출력(= LLM 이 보는 것)에 그 값이 전혀 없는지, 동시에 비공개 로그에는 남았는지 확인한다.
"""
import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAFE_RUN = HERE / "safe_run.py"
# 가짜 환자값. 일부러 눈에 띄는 문자열과 숫자를 섞음
CANARIES = ["patient_id=999", "lactate=7.2", "CANARY_SUBJ_10473921", "CANARY_HADM_28841577", "7.2913"]


class SafeRunLeakTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.code = Path(self.tmp.name) / "code"
        self.code.mkdir()
        self.logs = Path(self.tmp.name) / "logs"

    def tearDown(self):
        if self.logs.exists():
            os.chmod(self.logs, 0o700)
        self.tmp.cleanup()

    def script(self, body, name="02_fake.py"):
        p = self.code / name
        p.write_text("import sys, os\nsys.path.insert(0, %r)\nfrom agg import emit\n" % str(HERE)
                     + textwrap.dedent(body))
        return p

    def run_safe(self, step, script):
        env = {**os.environ, "SAFE_LOG_DIR": str(self.logs), "SAFE_CODE_DIR": str(self.code)}
        r = subprocess.run([sys.executable, str(SAFE_RUN), step, str(script)],
                           capture_output=True, text=True, env=env, timeout=60)
        visible = r.stdout + r.stderr                       # LLM 이 받게 되는 전부
        return visible, (json.loads(r.stdout) if r.stdout.strip() else {})

    def assert_no_leak(self, visible):
        for c in CANARIES:
            self.assertNotIn(c, visible)
        for token in ("patient_id", "lactate=", "Traceback"):
            self.assertNotIn(token, visible)

    def private_log(self, out):
        return (self.logs / out["log"]).read_text()

    # ------------------------------------------------------------ 누수 경로별 시험
    def test_exception_message_with_values(self):
        s = self.script("""
            print("row: patient_id=999 lactate=7.2")
            raise ValueError("bad row patient_id=999 lactate=7.2 CANARY_SUBJ_10473921")
        """)
        vis, out = self.run_safe("demo", s)
        self.assert_no_leak(vis)
        self.assertFalse(out["ok"])
        self.assertEqual(out["exception"], "ValueError")
        self.assertEqual(out["where"], "02_fake.py:6")        # 파일:줄은 공개
        log = self.private_log(out)
        self.assertIn("patient_id=999 lactate=7.2", log)     # 원문은 비공개 로그에만
        self.assertIn("Traceback", log)

    def test_keyerror_value_as_key(self):
        s = self.script("""
            d = {}
            d["CANARY_HADM_28841577"]
        """)
        vis, out = self.run_safe("demo", s)
        self.assert_no_leak(vis)
        self.assertEqual(out["exception"], "KeyError")

    def test_stdout_stderr_native_fd_and_child_process(self):
        s = self.script("""
            import subprocess, warnings
            print("patient_id=999")
            print("lactate=7.2", file=sys.stderr)
            os.write(1, b"CANARY_SUBJ_10473921\\n")
            os.write(2, b"CANARY_HADM_28841577\\n")
            subprocess.run(["echo", "7.2913"])
            warnings.warn("lactate=7.2 patient_id=999")
            emit(n=120, auroc=0.81)
        """)
        vis, out = self.run_safe("demo", s)
        self.assert_no_leak(vis)
        self.assertTrue(out["ok"])
        self.assertEqual(out["result"], {"n": 120, "auroc": 0.81})
        log = self.private_log(out)
        for c in CANARIES:
            self.assertIn(c, log)

    def test_systemexit_with_message(self):
        s = self.script('sys.exit("stop: patient_id=999 lactate=7.2")\n')
        vis, out = self.run_safe("demo", s)
        self.assert_no_leak(vis)
        self.assertFalse(out["ok"])
        self.assertEqual(out["exception"], "SystemExit")

    def test_exception_in_external_module_hides_path(self):
        s = self.script('import json\njson.loads("{patient_id=999 lactate=7.2}")\n')
        vis, out = self.run_safe("demo", s)
        self.assert_no_leak(vis)
        self.assertEqual(out["exception"], "JSONDecodeError")
        self.assertEqual(out["where"], "02_fake.py:5")       # 외부 라이브러리 경로가 아닌 우리 코드 위치

    # ------------------------------------------------------------ 집계 결과 스키마
    def test_extra_field_withholds_everything(self):
        s = self.script('emit(n=120, auroc=0.8, patient_id="CANARY_SUBJ_10473921")\n')
        vis, out = self.run_safe("demo", s)
        self.assert_no_leak(vis)
        self.assertNotIn("result", out)
        self.assertEqual(out["result_withheld"], "unknown_field")
        self.assertNotIn("120", vis)                          # 정상 필드도 함께 보류

    def test_string_value_rejected(self):
        s = self.script('emit(auroc="lactate=7.2")\n')
        vis, out = self.run_safe("demo", s)
        self.assert_no_leak(vis)
        self.assertEqual(out["result_withheld"], "bad_type")

    def test_value_as_table_key_rejected(self):
        s = self.script('emit(missing={"hr": 0.1, "CANARY_HADM_28841577": 0.2})\n')
        vis, out = self.run_safe("demo", s)
        self.assert_no_leak(vis)
        self.assertEqual(out["result_withheld"], "unknown_key")

    def test_enum_nan_range(self):
        for body, code in [('emit(version="CANARY_SUBJ_10473921")\n', "bad_enum"),
                           ('emit(auroc=float("nan"))\n', "bad_type"),
                           ('emit(auroc=7.2913)\n', "out_of_range"),
                           ('emit(n=-3)\n', "bad_count"),
                           ('emit(n=12.5)\n', "bad_count")]:
            vis, out = self.run_safe("demo", self.script(body))
            self.assert_no_leak(vis)
            self.assertEqual(out["result_withheld"], code, body)

    def test_malformed_agg_file(self):
        s = self.script('open(os.environ["SAFE_AGG_OUT"], "w").write("[patient_id=999 lactate=7.2")\n')
        vis, out = self.run_safe("demo", s)
        self.assert_no_leak(vis)
        self.assertEqual(out["result_withheld"], "unreadable")

    def test_nested_record_validation(self):
        good = 'emit(procedure_unit_transfers=12, cohort={"A": {"n_samples": 500, "flow": {"raw": 600}}})\n'
        vis, out = self.run_safe("cohort", self.script(good))
        self.assertEqual(out["result"]["cohort"]["A"]["flow"]["raw"], 600)
        bad = 'emit(cohort={"A": {"n_samples": 500, "CANARY_SUBJ_10473921": 1}})\n'
        vis, out = self.run_safe("cohort", self.script(bad))
        self.assert_no_leak(vis)
        self.assertEqual(out["result_withheld"], "unknown_field")

    def test_small_cell_suppressed(self):
        s = self.script('emit(n=7, missing={"hr": 0.25})\n')
        vis, out = self.run_safe("demo", s)
        self.assertEqual(out["result"], {"n": "<10", "missing": {"hr": 0.25}})

    def test_unknown_step(self):
        vis, out = self.run_safe("patient_dump", self.script("print(1)\n"))
        self.assertEqual(out, {"step": "patient_dump", "ok": False, "error": "unknown_step"})

    # ------------------------------------------------------------ 비공개 로그 보호
    def test_log_permissions(self):
        vis, out = self.run_safe("demo", self.script("print('patient_id=999')\n"))
        self.assertEqual(stat.S_IMODE(os.stat(self.logs).st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(os.stat(self.logs / out["log"]).st_mode), 0o600)

    def test_fail_closed_when_log_unwritable(self):
        self.logs.mkdir(mode=0o700)
        os.chmod(self.logs, 0o500)
        marker = Path(self.tmp.name) / "ran"
        s = self.script(f"open({str(marker)!r}, 'w').write('x')\n")
        vis, out = self.run_safe("demo", s)
        self.assertEqual(out, {"step": "demo", "ok": False, "error": "log_unavailable"})
        self.assertFalse(marker.exists())                     # 로그가 없으면 실행 자체를 안 함

    def test_rejects_group_readable_log_dir(self):
        self.logs.mkdir()
        os.chmod(self.logs, 0o755)
        vis, out = self.run_safe("demo", self.script("print(1)\n"))
        self.assertEqual(out.get("error"), "log_unavailable")


if __name__ == "__main__":
    unittest.main()
