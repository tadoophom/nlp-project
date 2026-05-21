"""Launch and monitor the train109 AWS retrain through Jupyter terminals."""
from __future__ import annotations

import argparse
import base64
import http.cookiejar
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, "/tmp/codex_py")

from websocket import WebSocketTimeoutException, create_connection


PASSWORD = "CaseOLAP2025"
BASE = "http://18.224.98.201:8888"
LOGIN_URL = BASE + "/login?next=%2Flab"
WS_BASE = "ws://18.224.98.201:8888"
LOG_PATH = "/tmp/run_v9_train109_console.log"
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\].*?\x07")

SETUP_CMD = r"""
cd /home/ubuntu/research-project
. /home/ubuntu/myenv/bin/activate
pwd
ls run_v9_train109.sh
ls data/hfpef_v9_train_autocorrect109.json
ls data/splits/hfpef_v8_eval_relabel4_noleak_large.json
"""

START_CMD = rf"""
cd /home/ubuntu/research-project
. /home/ubuntu/myenv/bin/activate
rm -f {LOG_PATH}
nohup bash run_v9_train109.sh > {LOG_PATH} 2>&1 &
PID=$!
echo PID:$PID
echo LOG:{LOG_PATH}
"""

STATUS_CMD = rf"""
cd /home/ubuntu/research-project
if [ -f {LOG_PATH} ]; then
  python3 - <<'PY'
from pathlib import Path

path = Path('{LOG_PATH}')
text = path.read_text(errors='ignore')
lines = text.splitlines()
print('LOG_LINES', len(lines))
for line in lines[-80:]:
    print(line)
PY
else
  echo LOG_MISSING
fi
"""

SUMMARY_CMD = rf"""
cd /home/ubuntu/research-project
python3 - <<'PY'
from pathlib import Path

text = Path('{LOG_PATH}').read_text(errors='ignore')
idx = text.rfind('RESULTS SUMMARY')
if idx == -1:
    print(text[-8000:])
else:
    print(text[idx:])
PY
"""


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text).replace("\r", "")


class RemoteTerminal:
    def __init__(self) -> None:
        self.cj = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cj))
        html = self.opener.open(LOGIN_URL, timeout=20).read().decode()
        xsrf = re.search(r'name="_xsrf" value="([^"]+)"', html).group(1)
        self.xsrf = xsrf
        login_data = urllib.parse.urlencode({"_xsrf": xsrf, "password": PASSWORD}).encode()
        self.opener.open(urllib.request.Request(LOGIN_URL, data=login_data, method="POST"), timeout=20)
        req = urllib.request.Request(
            BASE + "/api/terminals",
            data=b"{}",
            method="POST",
            headers={"Content-Type": "application/json", "X-XSRFToken": xsrf},
        )
        term = json.loads(self.opener.open(req, timeout=20).read().decode())
        cookie = "; ".join(f"{c.name}={c.value}" for c in self.cj)
        self.ws = create_connection(
            f"{WS_BASE}/terminals/websocket/{term['name']}",
            cookie=cookie,
            origin=BASE,
            timeout=10,
        )
        self._drain_initial()
        print(f"CONNECTED terminal={term['name']}", flush=True)

    def _drain_initial(self) -> None:
        deadline = time.time() + 3
        while time.time() < deadline:
            try:
                self.ws.recv()
            except Exception:
                break

    def run(self, cmd: str, timeout: int) -> tuple[int, str]:
        marker = f"__CODERUN_{uuid.uuid4().hex}__"
        sentinel = "\x1f" + marker
        payload = cmd.rstrip("\n") + f"\nprintf '\\037%s:%s\\n' '{marker}' $?\n"
        self.ws.send(json.dumps(["stdin", payload.replace("\n", "\r")]))

        deadline = time.time() + timeout
        chunks: list[str] = []
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
            chunks.append(text)
            combined = "".join(chunks)
            if sentinel in combined:
                cleaned = strip_ansi(combined)
                match = re.search(re.escape(sentinel) + r":(\d+)", cleaned)
                return (int(match.group(1)) if match else -1), cleaned
        raise TimeoutError(f"timed out waiting for command marker: {cmd[:80]}")

    def close(self) -> None:
        self.ws.close()

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        content = base64.b64encode(local_path.read_bytes()).decode()
        payload = json.dumps(
            {
                "type": "file",
                "format": "base64",
                "content": content,
            }
        ).encode()
        req = urllib.request.Request(
            BASE + "/api/contents/" + urllib.parse.quote(remote_path),
            data=payload,
            method="PUT",
            headers={
                "Content-Type": "application/json",
                "X-XSRFToken": self.xsrf,
            },
        )
        self.opener.open(req, timeout=60).read()
        print(f"UPLOADED {local_path} -> {remote_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch and monitor remote train109 retrain")
    parser.add_argument("--poll-seconds", type=int, default=90)
    parser.add_argument("--max-polls", type=int, default=80)
    args = parser.parse_args()

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

        status, output = term.run(SETUP_CMD, timeout=180)
        print("SETUP_STATUS", status, flush=True)
        print(output[-4000:], flush=True)
        if status != 0:
            raise RuntimeError("setup failed")

        status, output = term.run(START_CMD, timeout=60)
        print("START_STATUS", status, flush=True)
        print(output[-2000:], flush=True)
        if status != 0:
            raise RuntimeError("start failed")

        for poll in range(1, args.max_polls + 1):
            time.sleep(args.poll_seconds)
            status, output = term.run(STATUS_CMD, timeout=120)
            print(f"POLL {poll} STATUS {status}", flush=True)
            print(output[-12000:], flush=True)
            if "=== DONE ===" in output:
                break
            if "Traceback (most recent call last):" in output:
                break

        status, output = term.run(SUMMARY_CMD, timeout=120)
        print("SUMMARY_STATUS", status, flush=True)
        print(output[-16000:], flush=True)
    finally:
        term.close()


if __name__ == "__main__":
    main()
