"""Stop the running train109 AWS retrain through Jupyter terminals."""
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

STOP_CMD = r"""
cd /home/ubuntu/research-project
pkill -f 'run_v9_train109.sh'
pkill -f 'train_bert.py'
sleep 2
pgrep -af 'run_v9_train109|train_bert.py|hfpef_v9_train_autocorrect109' || true
printf '\037%s:%s\n' '__STOP__' $?
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

        ws.send(json.dumps(["stdin", STOP_CMD.replace("\n", "\r")]))
        out: list[str] = []
        deadline = time.time() + 60
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
            if "\x1f__STOP__:" in "".join(out):
                break

        print(strip_ansi("".join(out)))
    finally:
        ws.close()


if __name__ == "__main__":
    main()
