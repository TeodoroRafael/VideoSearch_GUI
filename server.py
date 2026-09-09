"""
VideoSearch GUI - proof of concept backend.

Serves the static front-end (index.html/style.css/app.js) from this same
folder and exposes:

  GET  /api/search?q=<query>   - proves the search box is wired to Python.
  POST /api/videos             - receives an opened video, stores it under
                                  database/<video name>/ and reports back
                                  the path to use as the reference copy.

Run:
    python3 server.py
Then open:
    http://localhost:8000

Swap process_query() for the real search logic later.
"""

import functools
import json
import os
import re
import socket
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

PORT = 8000
STATIC_DIR = Path(__file__).resolve().parent
DATABASE_DIR = STATIC_DIR / "database"


def process_query(query: str) -> str:
    # POC placeholder — swap for the real search logic (embeddings, model
    # inference, whatever ends up finding frames for `query`).
    return f"{query} searching"


def sanitize_name(name: str) -> str:
    name = Path(name).stem.strip()
    name = re.sub(r"[^A-Za-z0-9 _-]", "_", name)
    return name or "video"


VIDEO_EXTENSIONS = {
    ".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm",
    ".wmv", ".flv", ".mpg", ".mpeg", ".ts", ".3gp",
}


def find_existing_video(video_dir: Path) -> Path | None:
    # Only look at files with a video extension — a folder can also hold
    # subtitles, .nfo/poster files, etc. dropped in manually, and those
    # must never be picked as "the video" just for sorting first.
    candidates = sorted(
        p for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    return candidates[0] if candidates else None


RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/search":
            self.handle_search(parsed)
        elif self.headers.get("Range") and self.handle_range_request():
            pass
        else:
            super().do_GET()

    def handle_range_request(self) -> bool:
        # <video> seeking needs HTTP Range/206 support: without it the
        # browser can only treat what it has already buffered as seekable,
        # so scrubbing/clicking the timeline silently does nothing.
        # SimpleHTTPRequestHandler doesn't implement this, so we do it here
        # for any static file (falls back to a normal full-file GET for
        # everything else, including requests with no Range header).
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return False

        match = RANGE_RE.match(self.headers["Range"])
        if not match:
            return False

        file_size = os.path.getsize(path)
        start_str, end_str = match.groups()
        if start_str:
            start = int(start_str)
            end = int(end_str) if end_str else file_size - 1
        else:
            # suffix range, e.g. "bytes=-500" -> last 500 bytes
            start = max(file_size - int(end_str), 0)
            end = file_size - 1
        end = min(end, file_size - 1)

        if start > end or start >= file_size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{file_size}")
            self.end_headers()
            return True

        length = end - start + 1
        with open(path, "rb") as f:
            f.seek(start)
            data = f.read(length)

        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            # The browser aborts in-flight range requests whenever the user
            # seeks again before the previous chunk finished sending — that's
            # normal <video> scrubbing behavior, not a server error.
            pass
        return True

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/videos":
            self.handle_upload_video()
        else:
            self.send_error(404)

    def handle_search(self, parsed):
        params = parse_qs(parsed.query)
        query = params.get("q", [""])[0]
        result = process_query(query)

        self.respond_json({"query": query, "result": result})

    def handle_upload_video(self):
        raw_filename = unquote(self.headers.get("X-Filename", "video"))
        length = int(self.headers.get("Content-Length", 0))
        video_name = sanitize_name(raw_filename)
        video_dir = DATABASE_DIR / video_name

        # A folder for this video name already holds a video file -> reuse
        # it as the reference copy instead of writing a duplicate.
        existing = find_existing_video(video_dir) if video_dir.is_dir() else None

        if existing:
            self.rfile.read(length)  # drain the upload body, it's unused
            target = existing
            reused = True
        else:
            video_dir.mkdir(parents=True, exist_ok=True)
            target = video_dir / Path(raw_filename).name
            self.save_body_to_file(target, length)
            reused = False

        rel_path = "/" + str(target.relative_to(STATIC_DIR)).replace("\\", "/")
        self.respond_json({"name": video_name, "path": rel_path, "reused": reused})

    def save_body_to_file(self, target: Path, length: int):
        with open(target, "wb") as f:
            remaining = length
            while remaining > 0:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)

    def respond_json(self, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # keep the console readable during the POC
        if urlparse(self.path).path in ("/api/search", "/api/videos"):
            super().log_message(format, *args)

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            # Client (e.g. the <video> element) closed the connection
            # mid-transfer, which happens routinely during seeking.
            self.close_connection = True


if __name__ == "__main__":
    DATABASE_DIR.mkdir(exist_ok=True)
    handler = functools.partial(Handler, directory=str(STATIC_DIR))
    server = ThreadingHTTPServer(("localhost", PORT), handler)
    print(f"Serving VideoSearch GUI on http://localhost:{PORT}")
    server.serve_forever()
