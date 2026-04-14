from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
DEFAULT_HTML = WEB_ROOT / "index.html"


def _response(status: str, body: bytes, content_type: str = "text/plain; charset=utf-8"):
    headers = [
        ("Content-Type", content_type),
        ("Content-Length", str(len(body))),
    ]
    return status, headers, [body]


def app(environ, start_response) -> Iterable[bytes]:
    path = environ.get("PATH_INFO", "/")

    if path in ("/", "/index", "/index.html"):
        if DEFAULT_HTML.exists():
            content = DEFAULT_HTML.read_bytes()
            status, headers, body = _response("200 OK", content, "text/html; charset=utf-8")
        else:
            message = b"Dashboard file not found: web/index.html\n"
            status, headers, body = _response("404 Not Found", message)
        start_response(status, headers)
        return body

    if path.startswith("/web/"):
        rel = path.removeprefix("/web/")
        target = (WEB_ROOT / rel).resolve()
        if not str(target).startswith(str(WEB_ROOT.resolve())):
            status, headers, body = _response("403 Forbidden", b"forbidden\n")
            start_response(status, headers)
            return body
        if target.exists() and target.is_file():
            content = target.read_bytes()
            content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            status, headers, body = _response("200 OK", content, content_type)
            start_response(status, headers)
            return body
        status, headers, body = _response("404 Not Found", b"not found\n")
        start_response(status, headers)
        return body

    if path == "/health":
        status, headers, body = _response("200 OK", b"ok\n")
        start_response(status, headers)
        return body

    status, headers, body = _response("404 Not Found", b"not found\n")
    start_response(status, headers)
    return body


if __name__ == "__main__":
    from wsgiref.simple_server import make_server

    host = "0.0.0.0"
    port = 8000
    with make_server(host, port, app) as server:
        print(f"Serving on http://{host}:{port}")
        server.serve_forever()
