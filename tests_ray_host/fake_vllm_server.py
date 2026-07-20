from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json


class _Handler(BaseHTTPRequestHandler):
    server: "_Server"

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(200, b"")
            return
        if self.path == "/v1/models":
            self._json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.model_id,
                            "root": self.server.model_path,
                            "max_model_len": self.server.max_model_len,
                        }
                    ],
                }
            )
            return
        if self.path == "/metrics":
            self._send(
                200,
                (
                    "# TYPE vllm:num_requests_waiting gauge\n"
                    "vllm:num_requests_waiting 0\n"
                    "# TYPE vllm:num_requests_running gauge\n"
                    "vllm:num_requests_running 0\n"
                ).encode(),
                content_type="text/plain; version=0.0.4",
            )
            return
        self._send(404, b"not found")

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._send(404, b"not found")
            return
        length = int(self.headers.get("content-length", "0"))
        request = json.loads(self.rfile.read(length))
        content = request["messages"][-1]["content"]
        self._json(
            {
                "id": "chatcmpl-test",
                "model": request["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": f"ok:{content}"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2},
            }
        )

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def _json(self, value: object) -> None:
        self._send(200, json.dumps(value).encode(), content_type="application/json")

    def _send(
        self,
        status: int,
        body: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _Server(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        *,
        model_id: str,
        model_path: str,
        max_model_len: int,
    ) -> None:
        super().__init__(address, _Handler)
        self.model_id = model_id
        self.model_path = model_path
        self.max_model_len = max_model_len


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--max-model-len", required=True, type=int)
    args, _ = parser.parse_known_args()
    server = _Server(
        (args.host, args.port),
        model_id=args.served_model_name,
        model_path=args.model,
        max_model_len=args.max_model_len,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
