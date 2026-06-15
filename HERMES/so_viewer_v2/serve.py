"""
SO Viewer v2 — static file server on port 80 + /api/* reverse proxy to localhost:8080
Run: python3 serve.py
"""
import sys
import http.server
import urllib.request
import socketserver

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 80
UPSTREAM = "http://localhost:8080"
PROXY_TIMEOUT = 60  # /api/scan can take ~5-30s for cross-product scan

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Proxy /api/* to upstream Flask
        if self.path.startswith("/api/"):
            url = UPSTREAM + self.path
            try:
                with urllib.request.urlopen(url, timeout=PROXY_TIMEOUT) as r:
                    body = r.read()
                    self.send_response(r.status)
                    # CORS: allow v2 page to call from any port
                    self.send_header("Access-Control-Allow-Origin", "*")
                    for k, v in r.headers.items():
                        if k.lower() not in ("transfer-encoding", "connection", "content-length", "access-control-allow-origin"):
                            self.send_header(k, v)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
            except Exception as e:
                self.send_error(502, f"Upstream error: {e}")
                return
        return super().do_GET()

class ReuseTCPServer(socketserver.TCPServer):
    allow_reuse_address = True  # avoid "Address already in use" after restart (TIME_WAIT)

print(f"SO Viewer v2 serving on http://localhost:{PORT}  (proxy /api/* → {UPSTREAM})")
with ReuseTCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
