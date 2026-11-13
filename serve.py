"""Serve the repo for the UI with caching turned off, so edits to ui/index.html
and new runs show up on a normal reload.

    python3 serve.py          # http://localhost:8000/ui/
    python3 serve.py 8765
"""

import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Expires", "0")
        super().end_headers()


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    handler = partial(NoCacheHandler, directory=str(Path(__file__).parent))
    with ThreadingHTTPServer(("", port), handler) as httpd:
        print(f"Serving on http://localhost:{port}/ui/  (Ctrl+C to stop)")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
