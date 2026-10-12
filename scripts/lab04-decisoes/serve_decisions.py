"""CLI e API HTTP locais. Recomenda uma classe; nao executa a acao."""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from decision_core import ContextBuilder, file_hash, read_json, softmax, tokenizer
from decision_lab import Predictor


class DecisionEngine:
    def __init__(self, data, adapter, calibration):
        path = Path(data, "bundle.json")
        self.data = Path(data)
        self.bundle, self.cal = read_json(path), read_json(calibration)
        self.builder = ContextBuilder(self.bundle, tokenizer(self.bundle))
        self.model = Predictor(self.bundle, "classifier", adapter)
        if (self.cal["model_key"] != self.model.model_key or self.cal["bundle_sha256"] != file_hash(path)
                or self.cal["labels"] != self.bundle["labels"]):
            raise ValueError("Calibracao nao corresponde a este modelo/dataset.")
        if "threshold" not in self.cal:
            raise ValueError("Execute a selecao do limiar antes de servir.")

    def context(self, payload):
        if not isinstance(payload, dict) or set(payload) != {"history"}:
            raise ValueError("Entrada deve conter somente history.")
        built = self.builder.build(payload["history"])
        policies = {item["id"]: item["text"] for item in self.bundle["policy_docs"]}
        return {"prompt": built["text"], "eligible": built["eligible"],
                "input_tokens": len(built["input_ids"]), "max_seq": self.bundle["max_seq"],
                "dropped_turns": built["dropped_turns"], "policy_ids": built["policy_ids"],
                "policies": [{"id": key, "text": policies[key]} for key in built["policy_ids"]],
                "labels": self.bundle["labels"]}

    def catalog(self):
        return {"labels": self.bundle["labels"], "actions": self.bundle["actions"],
                "model": self.bundle["model"], "threshold": self.cal["threshold"]}

    def health(self):
        return {"status": "ready", "model_key": self.model.model_key,
                "model": self.bundle["model"], "classes": len(self.bundle["labels"]),
                "threshold": self.cal["threshold"], "temperature": self.cal["temperature"]}

    def examples(self):
        expected_path = self.data / "demo_expected.json"
        expected = read_json(expected_path) if expected_path.exists() else []
        result = []
        for index, path in enumerate(sorted((self.data / "demo").glob("case_*.json"))):
            payload = read_json(path)
            if set(payload) != {"history"}:
                raise ValueError("Exemplo de demonstracao invalido.")
            reference = expected[index] if index < len(expected) else {}
            result.append({"name": path.stem, "history": payload["history"],
                           "expected": reference.get("expected"), "id": reference.get("id")})
        return {"examples": result}

    def decide(self, payload):
        context = self.context(payload)
        if not context["eligible"]:
            return {"choice": None, "requires_review": True, "reason": "context_exceeded",
                    "dropped_turns": context["dropped_turns"], "input_tokens": context["input_tokens"]}
        start = time.perf_counter()
        raw = self.model(context["prompt"])
        probs = softmax(raw["logits"], self.cal["temperature"])
        labels = self.bundle["labels"]
        best = max(range(len(probs)), key=probs.__getitem__)
        return {"choice": labels[best], "probabilities": dict(zip(labels, probs)),
                "confidence": probs[best], "requires_review": probs[best] < self.cal["threshold"],
                "threshold": self.cal["threshold"], "dropped_turns": context["dropped_turns"],
                "policy_ids": context["policy_ids"], "input_tokens": context["input_tokens"],
                "latency_s": time.perf_counter() - start, "model_key": self.model.model_key}


def create_server(engine, port):
    class Handler(BaseHTTPRequestHandler):
        def answer(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            routes = {"/health": engine.health, "/catalog": engine.catalog, "/examples": engine.examples}
            if self.path not in routes:
                self.answer(404, {"error": "Rota desconhecida."})
                return
            try:
                self.answer(200, routes[self.path]())
            except Exception:
                self.answer(500, {"error": "Falha ao consultar dados do laboratorio."})

        def do_POST(self):
            routes = {"/decide": engine.decide, "/context": engine.context}
            if self.path not in routes:
                self.answer(404, {"error": "Rota desconhecida."})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1_000_000:
                    raise ValueError("Tamanho de entrada invalido.")
                result = routes[self.path](json.loads(self.rfile.read(size)))
                self.answer(200, result)
            except (ValueError, TypeError, KeyError) as exc:
                self.answer(400, {"error": str(exc)})
            except Exception:
                self.answer(500, {"error": "Falha na inferencia; verifique o terminal do servidor."})
    return HTTPServer(("127.0.0.1", port), Handler)


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
    print(f"API local em http://127.0.0.1:{args.port}/decide", flush=True)
    with create_server(engine, args.port) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
