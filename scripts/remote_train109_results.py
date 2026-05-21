"""Fetch final train109 AWS retrain metrics through Jupyter terminals."""
from __future__ import annotations

import http.cookiejar
import json
import re
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, "/tmp/codex_py")

from websocket import WebSocketTimeoutException, create_connection


PASSWORD = "CaseOLAP2025"
BASE = "http://18.224.98.201:8888"
LOGIN_URL = BASE + "/login?next=%2Flab"
WS_BASE = "ws://18.224.98.201:8888"
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\].*?\x07")

RESULTS_CMD = r"""
cd /home/ubuntu/research-project
python3 - <<'PY'
import json
from pathlib import Path

run_dir = Path('logs/v9_retrain_train109_clean_eval_20260318_211638')
print('RUN_DIR', run_dir)

print('UNCALIBRATED')
for path in sorted(run_dir.glob('eval_*.json')):
    data = json.loads(path.read_text())
    name = path.stem.removeprefix('eval_')
    print(name, data['accuracy'], data['macro_f1'])

print('CALIBRATED')
for path in sorted(run_dir.glob('calibration_*.json')):
    data = json.loads(path.read_text())
    name = path.stem.removeprefix('calibration_')
    best = data.get('best_constrained_thresholds')
    if best:
        print(name, best['accuracy'], best['macro_f1'])
    else:
        print(name, None, None)

print('ENSEMBLES')
for path in sorted(run_dir.glob('ensemble_*.json')):
    data = json.loads(path.read_text())
    name = path.stem
    if 'calibrated' in data:
        print(name, data['calibrated']['accuracy'], data['calibrated']['macro_f1'])
    elif 'average' in data:
        print(name, data['average']['accuracy'], data['average']['macro_f1'])
    else:
        print(name, None, None)
PY
printf '\037%s:%s\n' '__RESULTS__' $?
"""


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text).replace("\r", "")


def main() -> None:
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    html = opener.open(LOGIN_URL, timeout=20).read().decode()
    xsrf = re.search(r'name="_xsrf" value="([^"]+)"', html).group(1)
    login_data = urllib.parse.urlencode({"_xsrf": xsrf, "password": PASSWORD}).encode()
    opener.open(urllib.request.Request(LOGIN_URL, data=login_data, method="POST"), timeout=20)
    req = urllib.request.Request(
        BASE + "/api/terminals",
        data=b"{}",
        method="POST",
        headers={"Content-Type": "application/json", "X-XSRFToken": xsrf},
    )
    term = json.loads(opener.open(req, timeout=20).read().decode())
    cookie = "; ".join(f"{c.name}={c.value}" for c in cj)
    ws = create_connection(
        f"{WS_BASE}/terminals/websocket/{term['name']}",
        cookie=cookie,
        origin=BASE,
        timeout=10,
    )

    try:
        deadline = time.time() + 3
        while time.time() < deadline:
            try:
                ws.recv()
            except Exception:
                break

        ws.send(json.dumps(["stdin", RESULTS_CMD.replace("\n", "\r")]))
        out: list[str] = []
        deadline = time.time() + 90
        while time.time() < deadline:
            try:
                raw = ws.recv()
            except WebSocketTimeoutException:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(msg, list) or len(msg) != 2:
                continue
            kind, text = msg
            if kind != "stdout":
                continue
            out.append(text)
            if "\x1f__RESULTS__:" in "".join(out):
                break

        print(strip_ansi("".join(out)))
    finally:
        ws.close()


if __name__ == "__main__":
    main()
