"""Upload inputs and start the train109 AWS retrain through Jupyter terminals."""
from __future__ import annotations

import base64
import http.cookiejar
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, "/tmp/codex_py")

from websocket import WebSocketTimeoutException, create_connection


PASSWORD = "CaseOLAP2025"
BASE = "http://18.224.98.201:8888"
LOGIN_URL = BASE + "/login?next=%2Flab"
WS_BASE = "ws://18.224.98.201:8888"
LOG_PATH = "/tmp/run_v9_train109_console.log"
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\].*?\x07")

START_CMD = rf"""
cd /home/ubuntu/research-project
. /home/ubuntu/myenv/bin/activate
pkill -f 'run_v9_train109.sh' || true
pkill -f 'train_bert.py' || true
rm -f {LOG_PATH}
nohup bash run_v9_train109.sh > {LOG_PATH} 2>&1 &
PID=$!
sleep 3
echo PID:$PID
echo LOG:{LOG_PATH}
python3 - <<'PY'
from pathlib import Path

path = Path('{LOG_PATH}')
if path.exists():
    lines = path.read_text(errors='ignore').splitlines()
    print('LOG_LINES', len(lines))
    for line in lines[:20]:
        print(line)
PY
printf '\037%s:%s\n' '__START__' $?
"""


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text).replace("\r", "")


class RemoteTerminal:
    def __init__(self) -> None:
        self.cj = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cj))
        html = self.opener.open(LOGIN_URL, timeout=20).read().decode()
        self.xsrf = re.search(r'name="_xsrf" value="([^"]+)"', html).group(1)
        login_data = urllib.parse.urlencode({"_xsrf": self.xsrf, "password": PASSWORD}).encode()
        self.opener.open(urllib.request.Request(LOGIN_URL, data=login_data, method="POST"), timeout=20)
        req = urllib.request.Request(
            BASE + "/api/terminals",
            data=b"{}",
            method="POST",
            headers={"Content-Type": "application/json", "X-XSRFToken": self.xsrf},
        )
        term = json.loads(self.opener.open(req, timeout=20).read().decode())
        cookie = "; ".join(f"{c.name}={c.value}" for c in self.cj)
        self.ws = create_connection(
            f"{WS_BASE}/terminals/websocket/{term['name']}",
            cookie=cookie,
            origin=BASE,
            timeout=10,
        )

        deadline = time.time() + 3
        while time.time() < deadline:
            try:
                self.ws.recv()
            except Exception:
                break

    def close(self) -> None:
        self.ws.close()

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        payload = json.dumps(
            {
                "type": "file",
                "format": "base64",
                "content": base64.b64encode(local_path.read_bytes()).decode(),
            }
        ).encode()
        req = urllib.request.Request(
            BASE + "/api/contents/" + urllib.parse.quote(remote_path),
            data=payload,
            method="PUT",
            headers={"Content-Type": "application/json", "X-XSRFToken": self.xsrf},
        )
        self.opener.open(req, timeout=60).read()

    def run(self, cmd: str, timeout: int) -> str:
        self.ws.send(json.dumps(["stdin", cmd.replace("\n", "\r")]))
        out: list[str] = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                raw = self.ws.recv()
            except WebSocketTimeoutException:
                continue
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(message, list) or len(message) != 2:
                continue
            kind, text = message
            if kind != "stdout":
                continue
            out.append(text)
            if "\x1f__START__:" in "".join(out):
                break
        return strip_ansi("".join(out))


def main() -> None:
    term = RemoteTerminal()
    try:
        term.upload_file(
            Path("data/hfpef_v9_train_autocorrect109.json"),
            "research-project/data/hfpef_v9_train_autocorrect109.json",
        )
        term.upload_file(
            Path("notes/server/run_v9_retrain_train109_clean_eval.sh"),
            "research-project/run_v9_train109.sh",
        )
        print(term.run(START_CMD, timeout=120))
    finally:
        term.close()


if __name__ == "__main__":
    main()
