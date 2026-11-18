"""Serve the evaluator UI (Laptop 2):  python ui/serve.py [port]  ->  http://localhost:8080"""
import http.server, socketserver, sys, os
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
os.chdir(os.path.dirname(os.path.abspath(__file__)))
with socketserver.TCPServer(("", port), http.server.SimpleHTTPRequestHandler) as srv:
    print(f"PathSnap UI on http://localhost:{port}"); srv.serve_forever()
