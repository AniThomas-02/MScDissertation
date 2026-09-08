#!/usr/bin/env python3
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import json, re, sys, time

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)

SAFE = re.compile(r"[^A-Za-z0-9_.-]+") #sanitise the session id

class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        from urllib.parse import urlparse, unquote
        rel = unquote(urlparse(path).path).lstrip("/") or "index.html"
        return str((WEB / rel).resolve())

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def do_POST(self):
        if self.path.split("?",1)[0] != "/collect":
            self.send_error(404); return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            sid = SAFE.sub("_", str(data.get("session_id") or "unknown"))[:180]
            records = data.get("records") or []
            out = LOGS / f"{sid}.jsonl"
            with out.open("a", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, separators=(",",":")) + "\n")
            payload = json.dumps({"ok": True, "written": len(records), "file": out.name}).encode()
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(payload)))
            self.end_headers(); self.wfile.write(payload)
        except Exception as e:
            payload=json.dumps({"ok":False,"error":str(e)}).encode()
            self.send_response(400); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(payload))); self.end_headers(); self.wfile.write(payload)

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"Serving experiment files from {WEB}")
    print(f"Saving received JSONL under {LOGS}")
    print(f"Listening on 0.0.0.0:{port}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
