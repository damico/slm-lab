"""CLI e API HTTP locais. Recomenda uma classe; nao executa a acao."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from decision_core import ContextBuilder, file_hash, read_json, softmax, tokenizer
from decision_lab import Predictor


class DecisionEngine:
    def __init__(self, data, adapter, calibration):
        path = Path(data, "bundle.json")
        self.bundle, self.cal = read_json(path), read_json(calibration)
        self.builder = ContextBuilder(self.bundle, tokenizer(self.bundle))
        self.model = Predictor(self.bundle, "classifier", adapter)
        if (self.cal["model_key"] != self.model.model_key or self.cal["bundle_sha256"] != file_hash(path)
                or self.cal["labels"] != self.bundle["labels"]):
            raise ValueError("Calibracao nao corresponde a este modelo/dataset.")
        if "threshold" not in self.cal:
            raise ValueError("Execute a selecao do limiar antes de servir.")

    def decide(self, payload):
        if not isinstance(payload, dict) or set(payload) != {"history"}:
            raise ValueError("Entrada deve conter somente history.")
        built = self.builder.build(payload["history"])
        if not built["eligible"]:
            return {"choice": None, "requires_review": True, "reason": "context_exceeded"}
        raw = self.model(built["text"])
        probs = softmax(raw["logits"], self.cal["temperature"])
        labels = self.bundle["labels"]
        best = max(range(len(probs)), key=probs.__getitem__)
        return {"choice": labels[best], "probabilities": dict(zip(labels, probs)),
                "confidence": probs[best], "requires_review": probs[best] < self.cal["threshold"],
                "threshold": self.cal["threshold"], "dropped_turns": built["dropped_turns"],
                "policy_ids": built["policy_ids"], "model_key": self.model.model_key}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--calibration", required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", help="Arquivo JSON com history.")
    mode.add_argument("--port", type=int, help="API em 127.0.0.1.")
    args = ap.parse_args()
    engine = DecisionEngine(args.data, args.adapter, args.calibration)
    if args.input:
        print(json.dumps(engine.decide(read_json(args.input)), ensure_ascii=False, indent=2))
        return
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            status = 200
            try:
                if self.path != "/decide":
                    self.send_error(404)
                    return
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1_000_000:
                    raise ValueError("Tamanho de entrada invalido.")
                answer = engine.decide(json.loads(self.rfile.read(size)))
            except (ValueError, TypeError, KeyError) as exc:
                status, answer = 400, {"error": str(exc)}
            except Exception:
                status, answer = 500, {"error": "Falha na inferencia; verifique o terminal do servidor."}
            body = json.dumps(answer, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    print(f"API local em http://127.0.0.1:{args.port}/decide", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
