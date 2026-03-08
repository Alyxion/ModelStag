#!/usr/bin/env python3
"""
Simple HTTPS reverse proxy for ModelStag.

Forwards HTTPS requests to the HTTP FastAPI backend.
Uses Python's built-in ssl and http.server modules.

Usage:
    python https_proxy.py [--port 8443] [--backend http://localhost:8000]
"""

import argparse
import http.server
import socket
import ssl
import sys
import threading
import urllib.request
import urllib.error
from pathlib import Path


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler that proxies to backend."""

    backend_url = "http://localhost:8000"

    def log_message(self, format, *args):
        """Custom log format."""
        print(f"[HTTPS Proxy] {self.address_string()} - {format % args}")

    def do_proxy(self):
        """Forward request to backend."""
        # Build target URL
        target_url = f"{self.backend_url}{self.path}"

        # Get request body if present
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        # Build request
        req = urllib.request.Request(
            target_url,
            data=body,
            method=self.command,
        )

        # Copy headers (except Host)
        for header, value in self.headers.items():
            if header.lower() not in ('host', 'content-length'):
                req.add_header(header, value)

        # Add forwarded headers
        req.add_header('X-Forwarded-For', self.client_address[0])
        req.add_header('X-Forwarded-Proto', 'https')

        try:
            # Make request to backend
            with urllib.request.urlopen(req, timeout=300) as response:
                # Send response status
                self.send_response(response.status)

                # Copy response headers
                for header, value in response.headers.items():
                    if header.lower() not in ('transfer-encoding', 'connection'):
                        self.send_header(header, value)
                self.end_headers()

                # Stream response body
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for header, value in e.headers.items():
                if header.lower() not in ('transfer-encoding', 'connection'):
                    self.send_header(header, value)
            self.end_headers()
            self.wfile.write(e.read())

        except urllib.error.URLError as e:
            self.send_error(502, f"Backend unavailable: {e.reason}")

        except Exception as e:
            self.send_error(500, f"Proxy error: {e}")

    def do_GET(self):
        self.do_proxy()

    def do_POST(self):
        self.do_proxy()

    def do_PUT(self):
        self.do_proxy()

    def do_DELETE(self):
        self.do_proxy()

    def do_PATCH(self):
        self.do_proxy()

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, PATCH, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Access-Control-Max-Age', '86400')
        self.end_headers()


class ThreadedHTTPServer(http.server.HTTPServer):
    """HTTP server that handles requests in threads."""

    allow_reuse_address = True
    daemon_threads = True

    def process_request(self, request, client_address):
        """Handle request in a new thread."""
        thread = threading.Thread(
            target=self.process_request_thread,
            args=(request, client_address)
        )
        thread.daemon = True
        thread.start()

    def process_request_thread(self, request, client_address):
        """Thread target for request handling."""
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


def find_certs():
    """Find certificate files."""
    script_dir = Path(__file__).parent
    certs_dir = script_dir.parent / "certs"

    cert_file = certs_dir / "cert.pem"
    key_file = certs_dir / "key.pem"

    if not cert_file.exists() or not key_file.exists():
        return None, None

    return str(cert_file), str(key_file)


def main():
    parser = argparse.ArgumentParser(description="HTTPS reverse proxy for ModelStag")
    parser.add_argument("--port", type=int, default=8443, help="HTTPS port (default: 8443)")
    parser.add_argument("--backend", default="http://localhost:8000", help="Backend URL")
    parser.add_argument("--cert", help="Path to certificate file")
    parser.add_argument("--key", help="Path to private key file")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    # Find or use provided certificates
    cert_file = args.cert
    key_file = args.key

    if not cert_file or not key_file:
        cert_file, key_file = find_certs()

    if not cert_file or not key_file:
        print("ERROR: Certificate files not found!")
        print("")
        print("Generate certificates first:")
        print("  ./scripts/generate-certs.sh")
        print("")
        print("Or provide certificate paths:")
        print("  python https_proxy.py --cert /path/to/cert.pem --key /path/to/key.pem")
        sys.exit(1)

    # Set backend URL on handler class
    ProxyHandler.backend_url = args.backend

    # Create server
    server = ThreadedHTTPServer((args.host, args.port), ProxyHandler)

    # Wrap socket with SSL
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(cert_file, key_file)
    server.socket = ssl_context.wrap_socket(server.socket, server_side=True)

    print(f"HTTPS Proxy started")
    print(f"  Listening: https://{args.host}:{args.port}")
    print(f"  Backend:   {args.backend}")
    print(f"  Cert:      {cert_file}")
    print("")
    print("Press Ctrl+C to stop")
    print("")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
