from __future__ import annotations

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

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, "/tmp/codex_py")

from websocket import WebSocketTimeoutException, create_connection


PASSWORD = "CaseOLAP2025"
BASE = "http://18.224.98.201:8888"
LOGIN_URL = BASE + "/login?next=%2Flab"
WS_BASE = "ws://18.224.98.201:8888"
REMOTE_ROOT = Path("/home/ubuntu/research-project")
REMOTE_SCRIPT = "research-project/scripts/evaluation/run_v9_publication_eval.py"
REMOTE_PUBLICATION_JSON = "research-project/logs/v9_publication_eval.json"
REMOTE_CALIBRATED_JSON = "research-project/logs/v9_train_calibrated_eval.json"
LOCAL_PUBLICATION_JSON = ROOT / "logs/v9_publication_eval.json"
LOCAL_CALIBRATED_JSON = ROOT / "logs/v9_train_calibrated_eval.json"
LOCAL_BOOTSTRAP_JSON = ROOT / "logs/v9_bootstrap_cis.json"
LOCAL_ERROR_JSON = ROOT / "logs/v9_error_analysis.json"
LOCAL_CONSOLE_LOG = ROOT / "logs/v9_remote_publication_eval_console.log"
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\].*?\x07")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text).replace("\r", "")


def bootstrap_metrics(
    y_true: list[str],
    y_pred: list[str],
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    sample_size = len(y_true)
    acc_samples = np.empty(n_resamples, dtype=np.float64)
    macro_f1_samples = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        sample_indices = rng.integers(0, sample_size, size=sample_size)
        yt = y_true_arr[sample_indices]
        yp = y_pred_arr[sample_indices]
        acc_samples[index] = accuracy_score(yt, yp)
        macro_f1_samples[index] = f1_score(yt, yp, average="macro", zero_division=0)
    return {
        "accuracy": {
            "point": float(accuracy_score(y_true, y_pred)),
            "ci_lower": float(np.percentile(acc_samples, 2.5)),
            "ci_upper": float(np.percentile(acc_samples, 97.5)),
        },
        "macro_f1": {
            "point": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "ci_lower": float(np.percentile(macro_f1_samples, 2.5)),
            "ci_upper": float(np.percentile(macro_f1_samples, 97.5)),
        },
    }


def build_bootstrap_output(publication_payload: dict) -> dict:
    per_sample = publication_payload["per_sample"]
    y_true = [row["gold_label"] for row in per_sample]
    model_results = {}
    for model_name in publication_payload["models"]:
        y_pred = [row["model_predictions"][model_name]["label"] for row in per_sample]
        model_results[model_name] = bootstrap_metrics(y_true, y_pred)
    majority_pred = [row["majority_vote"] for row in per_sample]
    return {
        "source_file": str(LOCAL_PUBLICATION_JSON),
        "n_samples": len(per_sample),
        "n_resamples": 10_000,
        "per_model": model_results,
        "majority_vote": bootstrap_metrics(y_true, majority_pred),
    }


def build_error_output(publication_payload: dict) -> dict:
    grouped_errors: dict[str, list[dict]] = {}
    for row in publication_payload["per_sample"]:
        predicted = row["majority_vote"]
        gold = row["gold_label"]
        if predicted == gold:
            continue
        key = f"{gold}->{predicted}"
        grouped_errors.setdefault(key, []).append(
            {
                "index": row["index"],
                "sentence": row["sentence"],
            }
        )
    sorted_groups = dict(
        sorted(grouped_errors.items(), key=lambda item: (-len(item[1]), item[0]))
    )
    return {
        "source_file": str(LOCAL_PUBLICATION_JSON),
        "total_errors": sum(len(entries) for entries in sorted_groups.values()),
        "groups": {
            key: {
                "count": len(entries),
                "sentences": entries,
            }
            for key, entries in sorted_groups.items()
        },
    }


class RemoteTerminal:
    def __init__(self) -> None:
        self.cookie_jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cookie_jar))
        login_html = self.opener.open(LOGIN_URL, timeout=20).read().decode()
        self.xsrf = re.search(r'name="_xsrf" value="([^"]+)"', login_html).group(1)
        login_data = urllib.parse.urlencode({"_xsrf": self.xsrf, "password": PASSWORD}).encode()
        self.opener.open(urllib.request.Request(LOGIN_URL, data=login_data, method="POST"), timeout=20)
        request = urllib.request.Request(
            BASE + "/api/terminals",
            data=b"{}",
            method="POST",
            headers={"Content-Type": "application/json", "X-XSRFToken": self.xsrf},
        )
        terminal = json.loads(self.opener.open(request, timeout=20).read().decode())
        cookie = "; ".join(f"{cookie.name}={cookie.value}" for cookie in self.cookie_jar)
        self.websocket = create_connection(
            f"{WS_BASE}/terminals/websocket/{terminal['name']}",
            cookie=cookie,
            origin=BASE,
            timeout=10,
        )
        self._drain_initial()
        print(f"[remote] connected terminal={terminal['name']}", flush=True)

    def _drain_initial(self) -> None:
        deadline = time.time() + 3
        while time.time() < deadline:
            try:
                self.websocket.recv()
            except Exception:
                break

    def close(self) -> None:
        self.websocket.close()

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        payload = json.dumps(
            {
                "type": "file",
                "format": "base64",
                "content": base64.b64encode(local_path.read_bytes()).decode(),
            }
        ).encode()
        request = urllib.request.Request(
            BASE + "/api/contents/" + urllib.parse.quote(remote_path),
            data=payload,
            method="PUT",
            headers={"Content-Type": "application/json", "X-XSRFToken": self.xsrf},
        )
        self.opener.open(request, timeout=60).read()
        print(f"[upload] {local_path} -> {remote_path}", flush=True)

    def download_text_file(self, remote_path: str) -> str:
        query = urllib.parse.urlencode({"content": 1, "format": "text"})
        response = self.opener.open(
            BASE + "/api/contents/" + urllib.parse.quote(remote_path) + "?" + query,
            timeout=60,
        ).read()
        payload = json.loads(response.decode())
        if payload["format"] == "base64":
            return base64.b64decode(payload["content"]).decode()
        return payload["content"]

    def run_streaming(self, command: str, timeout_seconds: int, log_path: Path) -> tuple[int, str]:
        marker = f"__CODERUN_{uuid.uuid4().hex}__"
        sentinel = "\x1f" + marker
        payload = command.rstrip("\n") + f"\nprintf '\\037%s:%s\\n' '{marker}' $?\n"
        self.websocket.send(json.dumps(["stdin", payload.replace("\n", "\r")]))

        log_path.parent.mkdir(parents=True, exist_ok=True)
        collected: list[str] = []
        printed = ""
        deadline = time.time() + timeout_seconds
        with log_path.open("w") as log_handle:
            while time.time() < deadline:
                try:
                    raw_message = self.websocket.recv()
                except WebSocketTimeoutException:
                    continue
                try:
                    message = json.loads(raw_message)
                except json.JSONDecodeError:
                    continue
                if not isinstance(message, list) or len(message) != 2:
                    continue
                kind, text = message
                if kind != "stdout":
                    continue
                collected.append(text)
                cleaned = strip_ansi("".join(collected))
                delta = cleaned[len(printed):]
                if delta:
                    log_handle.write(delta)
                    log_handle.flush()
                    print(delta, end="", flush=True)
                    printed = cleaned
                if sentinel in cleaned:
                    status_match = re.search(re.escape(sentinel) + r":(\d+)", cleaned)
                    status_code = int(status_match.group(1)) if status_match else -1
                    final_output = cleaned.replace(f"{sentinel}:{status_code}\n", "")
                    return status_code, final_output
        raise TimeoutError(f"Timed out after {timeout_seconds}s waiting for remote command")


def main() -> None:
    local_script = ROOT / "scripts/evaluation/run_v9_publication_eval.py"
    remote_command = f"""
cd {REMOTE_ROOT}
. /home/ubuntu/myenv/bin/activate
python3 scripts/evaluation/run_v9_publication_eval.py
"""

    terminal = RemoteTerminal()
    try:
        terminal.upload_file(local_script, REMOTE_SCRIPT)
        print("[remote] starting evaluation", flush=True)
        status_code, _ = terminal.run_streaming(remote_command, timeout_seconds=5400, log_path=LOCAL_CONSOLE_LOG)
        if status_code != 0:
            raise RuntimeError(f"Remote evaluation failed with exit code {status_code}")

        publication_text = terminal.download_text_file(REMOTE_PUBLICATION_JSON)
        calibrated_text = terminal.download_text_file(REMOTE_CALIBRATED_JSON)
        LOCAL_PUBLICATION_JSON.parent.mkdir(parents=True, exist_ok=True)
        LOCAL_PUBLICATION_JSON.write_text(publication_text)
        LOCAL_CALIBRATED_JSON.write_text(calibrated_text)
        print(f"[download] {REMOTE_PUBLICATION_JSON} -> {LOCAL_PUBLICATION_JSON}", flush=True)
        print(f"[download] {REMOTE_CALIBRATED_JSON} -> {LOCAL_CALIBRATED_JSON}", flush=True)
    finally:
        terminal.close()

    publication_payload = json.loads(LOCAL_PUBLICATION_JSON.read_text())
    bootstrap_payload = build_bootstrap_output(publication_payload)
    LOCAL_BOOTSTRAP_JSON.write_text(json.dumps(bootstrap_payload, indent=2))
    print(f"[local] wrote {LOCAL_BOOTSTRAP_JSON}", flush=True)

    error_payload = build_error_output(publication_payload)
    LOCAL_ERROR_JSON.write_text(json.dumps(error_payload, indent=2))
    print(f"[local] wrote {LOCAL_ERROR_JSON}", flush=True)


if __name__ == "__main__":
    main()
