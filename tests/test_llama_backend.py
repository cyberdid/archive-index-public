import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from archive_index.embed import LlamaServerEmbedder


class _Handler(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        inputs = json.loads(self.rfile.read(length))["input"]
        _Handler.seen.extend(inputs)
        data = [{"index": i, "embedding": [float(len(t)), 1.0, 2.0]} for i, t in enumerate(inputs)]
        body = json.dumps({"data": data}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_llama_backend_adds_prefixes_and_normalises():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        embedder = LlamaServerEmbedder(f"http://127.0.0.1:{server.server_port}")
        vectors = embedder.documents(["alpha"], ["Doc"])
        assert _Handler.seen[-1] == "title: Doc | text: alpha"
        assert np.isclose(np.linalg.norm(vectors[0]), 1.0, atol=1e-5)
        q = embedder.query("beta")
        assert _Handler.seen[-1] == "task: search result | query: beta"
        assert q.shape == (3,)
    finally:
        server.shutdown()
