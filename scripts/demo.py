import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def call(url, value=None):
    request = Request(url, data=json.dumps(value).encode() if value is not None else None,
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=15) as response:
        return json.load(response)


def main():
    root = Path(__file__).resolve().parents[1]
    processes = []
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as temporary:
        try:
            for arguments in (
                ["-m", "learning.provider"],
                ["-m", "learning.service", "--database", str(Path(temporary) / "analyses.sqlite3")],
            ):
                processes.append(subprocess.Popen([sys.executable, *arguments], cwd=root))
            deadline = started + 65
            for port in (8765, 8766):
                while True:
                    try:
                        call(f"http://127.0.0.1:{port}/health")
                        break
                    except (URLError, TimeoutError):
                        if time.monotonic() >= deadline:
                            raise RuntimeError("服务未在启动期限内就绪")
                        time.sleep(.1)
            print(json.dumps(call("http://127.0.0.1:8766/catalog"), ensure_ascii=False, indent=2))
            record = call("http://127.0.0.1:8766/analyses", {
                "request_key": "local-overview", "release": "term-a", "analysis": "overview", "parameters": {},
            })
            print(json.dumps(record, ensure_ascii=False, indent=2))
            loaded = call("http://127.0.0.1:8766/analyses/" + record["id"])
            if loaded != record:
                raise RuntimeError("保存的分析与返回值不一致")
            profile = call("http://127.0.0.1:8766/analyses", {
                "request_key": "local-shared-barriers", "release": "term-a", "analysis": "shared_barriers",
                "parameters": {},
            })
            structure = profile["result"]["structure"]
            print(json.dumps({
                "id": profile["id"], "mode": profile["result"]["mode"],
                "structure_version": profile["result"]["structure_version"],
                "dimensions": [{"id": row["id"], "kind": row["kind"], "views": row["views"]}
                               for row in structure["dimensions"]],
                "dimension_count": structure["estimation"].get("dimension_count"),
                "position_status": profile["result"]["position_summary"]["person_status"],
                "diagnostics": sorted(profile["result"]["diagnostics"]),
            }, ensure_ascii=False, indent=2))
            reused = call("http://127.0.0.1:8766/analyses", {
                "request_key": "local-shared-barriers-apply", "release": "term-b",
                "analysis": "shared_barriers", "parameters": {"structure_from": profile["id"]},
            })
            if reused["result"]["mode"] != "apply" or reused["result"]["structure_source"]["analysis"] != profile["id"]:
                raise RuntimeError("沿用已保存结构失败")
            if call("http://127.0.0.1:8766/analyses/" + profile["id"])["result"]["mode"] != "fit":
                raise RuntimeError("沿用结构不得改动来源分析")
            print(f"本地分析完成，用时 {time.monotonic() - started:.2f} 秒")
        finally:
            for process in processes:
                process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
