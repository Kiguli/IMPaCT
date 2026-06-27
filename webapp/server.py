#!/usr/bin/env python3
# ============================================================================
# IMPaCT web app — backend (Phase 4).
#
# A dependency-free (Python standard library only) server that:
#   * serves the static front-end (webapp/static/), and
#   * exposes POST /api/solve : given a small IMDP (.imdp or PRISM text) + a
#     property, runs the `imdp_solve` CLI (--json) and returns the model
#     structure + per-state satisfaction probabilities for visualization.
#
# Design (per the plan, modelled on the user's TRUST app): the C++ core is
# untouched; the backend just shells to the verified CLI. Zero pip installs so it
# runs anywhere (incl. the user's Windows machine: python webapp/server.py, with
# tools/imdp_solve.exe built). A Flask/Inertia/Vue rebuild can layer on later;
# this stdlib version keeps the small-model visualization runnable today.
#
# Run:
#   c++ -std=c++17 -O2 tools/imdp_solve.cpp src/imdp_io.cpp src/prism.cpp \
#       src/solve.cpp src/omaximization.cpp src/graph_utils.cpp src/omega.cpp \
#       -o tools/imdp_solve
#   python3 webapp/server.py            # then open http://localhost:8000
# ============================================================================
import argparse, json, os, subprocess, sys, tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")

def tool(name):
    for n in (name, name + ".exe"):
        cand = os.path.join(HERE, "..", "tools", n)
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return os.path.abspath(os.path.join(HERE, "..", "tools", name))

SOLVER = tool("imdp_solve")
STATE_CAP = 200          # hard safety cap for the backend (front-end caps rendering)

# (endpoint, tool, file-suffix, extra-args-builder) for the model-file CLIs.
def _grid_args(req):    return ["--eps", str(req.get("eps", 1e-6))]
def _zone_args(req):    return [str(req.get("target", "-")), req.get("engine", "zone"), req.get("bound", "pess")]
def _belief_args(req):  return [str(req.get("horizon", 4)), req.get("bound", "max")]
CLI_ENDPOINTS = {
    "/api/grid":       ("grid_heatmap",      ".sys",   _grid_args),
    "/api/zonegraph":  ("ta_zonegraph",      ".pta",   _zone_args),
    "/api/belieftree": ("pomdp_belieftree",  ".pomdp", _belief_args),
}

CONTENT = {".html": "text/html", ".js": "application/javascript", ".css": "text/css"}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            path = "/index.html"
        fp = os.path.normpath(os.path.join(STATIC, path.lstrip("/")))
        if not fp.startswith(STATIC) or not os.path.isfile(fp):
            return self._send(404, "not found", "text/plain")
        ext = os.path.splitext(fp)[1]
        with open(fp, "rb") as f:
            self._send(200, f.read(), CONTENT.get(ext, "application/octet-stream"))

    def do_POST(self):
        if self.path not in ("/api/solve",) and self.path not in CLI_ENDPOINTS:
            return self._send(404, json.dumps({"error": "not found"}))
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._send(400, json.dumps({"error": f"bad request: {e}"}))

        if self.path in CLI_ENDPOINTS:
            return self._run_model_cli(req)

        model = req.get("model", "")
        fmt = req.get("format", "imdp")            # imdp | prism
        prop = req.get("prop", "reach")
        label = req.get("label", "")
        bound = req.get("bound", "both")
        eps = str(req.get("eps", 1e-6))
        if not model.strip():
            return self._send(400, json.dumps({"error": "empty model"}))
        if not os.path.exists(SOLVER):
            return self._send(500, json.dumps({"error": f"solver not built: {SOLVER}. Build tools/imdp_solve (see webapp/README)."}))

        suffix = ".prism" if fmt == "prism" else ".imdp"
        tf = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False)
        try:
            tf.write(model); tf.close()
            cmd = [SOLVER, tf.name, prop, label, "--bound", bound, "--eps", eps, "--json"]
            if req.get("trace"): cmd.append("--trace")
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if p.returncode != 0:
                return self._send(400, json.dumps({"error": p.stderr.strip() or "solve failed"}))
            out = json.loads(p.stdout)
            out["stateCap"] = STATE_CAP
            return self._send(200, json.dumps(out))
        except subprocess.TimeoutExpired:
            return self._send(504, json.dumps({"error": "solve timed out (model too large for the web demo)"}))
        except Exception as e:
            return self._send(500, json.dumps({"error": str(e)}))
        finally:
            try: os.unlink(tf.name)
            except OSError: pass

    def _run_model_cli(self, req):
        name, suffix, argbuilder = CLI_ENDPOINTS[self.path]
        binpath = tool(name)
        model = req.get("model", "")
        if not model.strip():
            return self._send(400, json.dumps({"error": "empty model"}))
        if not os.path.exists(binpath):
            return self._send(500, json.dumps({"error": f"tool not built: {binpath} (see webapp/README)"}))
        tf = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False)
        try:
            tf.write(model); tf.close()
            cmd = [binpath, tf.name] + argbuilder(req)
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if p.returncode != 0:
                return self._send(400, json.dumps({"error": p.stderr.strip() or "tool failed"}))
            return self._send(200, p.stdout)   # tools already emit JSON
        except subprocess.TimeoutExpired:
            return self._send(504, json.dumps({"error": "timed out (model too large for the demo)"}))
        except Exception as e:
            return self._send(500, json.dumps({"error": str(e)}))
        finally:
            try: os.unlink(tf.name)
            except OSError: pass

    def log_message(self, *a):  # quieter
        pass


def main():
    global SOLVER
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--solver", default=SOLVER)
    args = ap.parse_args()
    SOLVER = os.path.abspath(args.solver)
    print(f"IMPaCT web app on http://localhost:{args.port}  (solver: {SOLVER})")
    if not os.path.exists(SOLVER):
        print("  WARNING: solver not found — build tools/imdp_solve (see webapp/README.md).", file=sys.stderr)
    ThreadingHTTPServer(("", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
