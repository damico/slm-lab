"""Calibracao, limiar de revisao, metricas e bootstrap por conversa."""
import argparse
import collections
import json
import math
import random
from pathlib import Path
import numpy as np
from decision_core import file_hash, meta_path, read_json, read_rows, save_json, softmax, write_rows


def aligned(reference, predictions):
    refs, pred = read_rows(reference), read_rows(predictions)
    meta = read_json(meta_path(predictions))
    if meta["input_sha256"] != file_hash(reference):
        raise ValueError("Predicoes nao foram geradas deste arquivo de referencia.")
    mapping = {r["id"]: r for r in pred}
    ids = {r["id"] for r in refs}
    if len(mapping) != len(pred) or len(ids) != len(refs) or set(mapping) - ids:
        raise ValueError("IDs duplicados ou estranhos.")
    return refs, mapping, meta


def calibration_matrix(refs, mapping, expected_split, labels):
    if {r["split"] for r in refs} != {expected_split}:
        raise ValueError("Esperado exclusivamente split " + expected_split)
    values, targets = [], []
    for row in refs:
        if not row["eligible"]:
            continue
        p = mapping.get(row["id"], {})
        if p.get("status") != "ok" or len(p.get("logits", [])) != len(labels):
            raise ValueError("Predicoes completas com logits sao obrigatorias nesta etapa.")
        values.append(p["logits"])
        targets.append(row["label_id"])
    z, y = np.asarray(values, dtype=float), np.asarray(targets, dtype=int)
    if len(y) == 0 or not np.isfinite(z).all():
        raise ValueError("Logits vazios ou invalidos.")
    return z, y


def calibrated_probs(logits, temperature):
    z = logits / temperature
    z -= z.max(axis=1, keepdims=True)
    ex = np.exp(z)
    return ex / ex.sum(axis=1, keepdims=True)


def fit_temperature(args):
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp
    refs, mapping, meta = aligned(args.reference, args.predictions)
    z, y = calibration_matrix(refs, mapping, "calibration", meta["labels"])
    def nll(log_t):
        scaled = z / math.exp(log_t)
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(y)), y]))
    fit = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    if not fit.success:
        raise RuntimeError("Ajuste de temperatura falhou.")
    save_json(args.out, {"temperature": math.exp(float(fit.x)), "model_key": meta["model_key"],
        "bundle_sha256": meta["bundle_sha256"], "labels": meta["labels"],
        "n_calibration": len(y), "nll_before": nll(0), "nll_after": nll(float(fit.x)),
        "calibration_reference_sha256": file_hash(args.reference),
        "calibration_predictions_sha256": file_hash(args.predictions)})
    print(Path(args.out).read_text())


def check_calibration(cal, meta):
    for key in ("model_key", "bundle_sha256", "labels"):
        if cal[key] != meta[key]:
            raise ValueError("Calibracao pertence a outro modelo/bundle: " + key)


def select_threshold(args):
    refs, mapping, meta = aligned(args.reference, args.predictions)
    cal = read_json(args.calibration)
    check_calibration(cal, meta)
    z, y = calibration_matrix(refs, mapping, "selection", meta["labels"])
    probs = calibrated_probs(z, cal["temperature"])
    confidence = probs.max(axis=1)
    errors = probs.argmax(axis=1) != y
    best = {"threshold": 1.01, "n_selected": 0, "observed_error": None, "coverage": 0.0}
    # Grade fixa escolhida antes do teste. Criterio empirico, sem garantia de risco futuro.
    for threshold in np.linspace(0, 1, 201):
        chosen = confidence >= threshold
        n = int(chosen.sum())
        error = float(errors[chosen].mean()) if n else None
        if n >= args.min_selected and error <= args.target_error and n > best["n_selected"]:
            best = {"threshold": float(threshold), "n_selected": n,
                    "observed_error": error, "coverage": n / len(y)}
    cal.update(best, target_error=args.target_error, min_selected=args.min_selected,
               selection_reference_sha256=file_hash(args.reference),
               selection_predictions_sha256=file_hash(args.predictions))
    save_json(args.out, cal)
    print(json.dumps(cal, indent=2))


def probability_metrics(probabilities, targets):
    p, y = np.asarray(probabilities), np.asarray(targets)
    if not len(y):
        return None
    confidence, correct = p.max(axis=1), p.argmax(axis=1) == y
    brier = ((p - np.eye(p.shape[1])[y]) ** 2).sum(axis=1).mean()
    ece, bins = 0.0, []
    for i in range(10):
        mask = (confidence >= i / 10) & (confidence < (i + 1) / 10 if i < 9 else confidence <= 1)
        n = int(mask.sum())
        if n:
            acc, conf = float(correct[mask].mean()), float(confidence[mask].mean())
            ece += n / len(y) * abs(acc - conf)
            bins.append({"lo": i / 10, "hi": (i + 1) / 10, "n": n, "accuracy": acc, "confidence": conf})
    return {"n": len(y), "nll": float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-15)).mean()),
            "brier_sum_classes": float(brier), "ece_10_bins": ece, "reliability_bins": bins}


def score(args):
    from sklearn.metrics import f1_score, confusion_matrix
    refs, mapping, meta = aligned(args.reference, args.predictions)
    labels = meta["labels"]
    cal = read_json(args.calibration) if args.calibration else None
    if cal:
        check_calibration(cal, meta)
    temperature = cal["temperature"] if cal else 1.0
    threshold = cal.get("threshold", 1.01) if cal else 1.01
    cases, before, after, probabilistic_targets = [], [], [], []
    for row in refs:
        p = mapping.get(row["id"], {})
        candidate = p.get("choice") if p.get("status") == "ok" else None
        confidence = None
        if p.get("status") == "ok" and "logits" in p:
            if len(p["logits"]) != len(labels):
                raise ValueError("Dimensao dos logits incorreta.")
            probs = softmax(p["logits"], temperature)
            candidate = labels[int(np.argmax(probs))]
            confidence = max(probs)
            if row["eligible"]:
                before.append(softmax(p["logits"]))
                after.append(probs)
                probabilistic_targets.append(row["label_id"])
        correct = bool(row["eligible"] and candidate == row["label"])
        cases.append({"id": row["id"], "conversation_id": row["conversation_id"],
            "eligible": row["eligible"], "expected": row["label"], "predicted": candidate,
            "correct": correct, "confidence": confidence,
            "selected": bool(row["eligible"] and confidence is not None and confidence >= threshold),
            "status": p.get("status", "missing_prediction")})
    eligible = [c for c in cases if c["eligible"]]
    selected = [c for c in cases if c["selected"]]
    actions = [c for c in eligible if c["expected"] != "retrieve_utterance"]
    rate = lambda cs: sum(c["correct"] for c in cs) / len(cs) if cs else None
    lat = [p["latency_s"] for p in mapping.values() if p.get("status") == "ok" and "latency_s" in p]
    per_class = {label: {"n": sum(c["expected"] == label for c in eligible),
                        "accuracy": rate([c for c in eligible if c["expected"] == label])} for label in labels}
    ys = [c["expected"] for c in eligible]
    ps = [c["predicted"] or "__invalid__" for c in eligible]
    summary = {"model_key": meta["model_key"], "n_total": len(cases), "n_eligible": len(eligible),
        "accuracy_all": rate(cases), "accuracy_eligible": rate(eligible),
        "macro_f1": float(f1_score(ys, ps, labels=labels, average="macro", zero_division=0)) if eligible else None,
        "action_only_n": len(actions), "action_only_accuracy": rate(actions), "per_class": per_class,
        "n_selected": len(selected), "coverage_all": len(selected) / len(cases) if cases else None,
        "selected_error": 1 - rate(selected) if selected else None,
        "probabilities_before": probability_metrics(before, probabilistic_targets),
        "probabilities_after": probability_metrics(after, probabilistic_targets),
        "threshold": threshold, "latency_success_n": len(lat),
        "p50_s": float(np.quantile(lat, .5)) if lat else None,
        "p95_s": float(np.quantile(lat, .95)) if lat else None,
        "api_prompt_tokens": sum(p.get("usage", {}).get("prompt_tokens", 0) for p in mapping.values()),
        "api_completion_tokens": sum(p.get("usage", {}).get("completion_tokens", 0) for p in mapping.values())}
    save_json(args.out + ".summary.json", summary)
    write_rows(args.out + ".cases.jsonl", cases)
    all_labels = labels + ["__invalid__"]
    save_json(args.out + ".confusion.json", {"labels": all_labels,
        "rows_expected_columns_predicted": confusion_matrix(ys, ps, labels=all_labels).tolist()})
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_class", "probabilities_before", "probabilities_after")}, indent=2))


def compare(args):
    a = {r["id"]: r for r in read_rows(args.a)}
    b = {r["id"]: r for r in read_rows(args.b)}
    if a.keys() != b.keys() or any(a[k]["expected"] != b[k]["expected"] or a[k]["eligible"] != b[k]["eligible"] for k in a):
        raise ValueError("Conjuntos de avaliacao diferentes.")
    groups = collections.defaultdict(lambda: [0, 0])
    for key in a:
        g = groups[a[key]["conversation_id"]]
        g[0] += int(b[key]["correct"]) - int(a[key]["correct"])
        g[1] += 1
    values = list(groups.values())
    if not values:
        raise ValueError("Nenhuma conversa para comparar.")
    rng = random.Random(42)
    boot = []
    for _ in range(2000):
        sample = rng.choices(values, k=len(values))
        boot.append(sum(v[0] for v in sample) / sum(v[1] for v in sample))
    print(json.dumps({"conversations": len(groups), "decisions": len(a),
        "delta_B_minus_A": sum(v[0] for v in values) / len(a),
        "cluster_bootstrap95": np.quantile(boot, [.025, .975]).tolist()}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    for command in ("calibrate", "select", "score"):
        p = sub.add_parser(command)
        p.add_argument("reference")
        p.add_argument("predictions")
        p.add_argument("--out", required=True)
        if command != "calibrate":
            p.add_argument("--calibration", required=command == "select")
        if command == "select":
            p.add_argument("--target-error", type=float, default=.1)
            p.add_argument("--min-selected", type=int, default=100)
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    args = ap.parse_args()
    {"calibrate": fit_temperature, "select": select_threshold, "score": score, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
