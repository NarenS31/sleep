from __future__ import annotations

from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
GRAPH_HTML = ROOT / "outputs" / "sleep_features_3d_connected.html"


def _response(status: str, body: bytes, content_type: str = "text/plain; charset=utf-8"):
    headers = [
        ("Content-Type", content_type),
        ("Content-Length", str(len(body))),
    ]
    return status, headers, [body]


def app(environ, start_response) -> Iterable[bytes]:
    path = environ.get("PATH_INFO", "/")

    if path in ("/", "/index", "/index.html"):
        if GRAPH_HTML.exists():
            content = GRAPH_HTML.read_bytes()
            status, headers, body = _response("200 OK", content, "text/html; charset=utf-8")
        else:
            message = (
                "Graph file not found. Generate it first with:\n"
                "python scripts/create_3d_connected_sleep_graph.py\n"
            ).encode("utf-8")
            status, headers, body = _response("404 Not Found", message)
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
