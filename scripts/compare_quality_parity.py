#!/usr/bin/env python3
"""Fail-closed reference-vs-candidate real-token NLL parity evidence.

The two reports MUST share token IDs, corpus and scoring protocol before an
FP16-relative quality loss can be called evidence. This does not claim that
either run was performed on the Tier-0 acceptance machine.
"""
import argparse
import json
import math
from pathlib import Path

REQUIRED_EQUAL = (
    "protocol", "token_fingerprint_fnv1a64", "token_ids_sha256_le_i32",
    "evaluated_token_count",
    "scored_target_tokens", "window_tokens",
)

def validate_evidence(report: dict, label: str) -> None:
    if not isinstance(report, dict):
        raise ValueError(f"{label} evidence must be a JSON object")
    if report.get("protocol") != "kiln-reset-window-next-token-nll-v1":
        raise ValueError(f"{label} evidence has an unknown or missing scoring protocol")
    for key, width in (("token_fingerprint_fnv1a64", 16),
                       ("token_ids_sha256_le_i32", 64)):
        value = report.get(key)
        if (not isinstance(value, str) or len(value) != width
                or any(char not in "0123456789abcdef" for char in value)):
            raise ValueError(f"{label} evidence has an invalid {key}")
    token_count = report.get("evaluated_token_count")
    scored = report.get("scored_target_tokens")
    window = report.get("window_tokens")
    if type(token_count) is not int or not 2 <= token_count <= 65536:
        raise ValueError(f"{label} evidence has an invalid token count")
    if type(scored) is not int or scored != token_count - 1:
        raise ValueError(f"{label} evidence skipped or duplicated target tokens")
    if type(window) is not int or not 2 <= window <= 8192:
        raise ValueError(f"{label} evidence has an invalid scoring window")
    perplexity = report.get("perplexity")
    if (type(perplexity) not in (int, float)
            or not math.isfinite(perplexity) or perplexity < 1):
        raise ValueError(f"{label} evidence has invalid perplexity")


def assess(reference: dict, candidate: dict, max_relative_ppl_loss: float) -> dict:
    if not (math.isfinite(max_relative_ppl_loss)
            and 0 <= max_relative_ppl_loss <= 1):
        raise ValueError("relative perplexity loss must be finite and within 0..1")
    validate_evidence(reference, "reference")
    validate_evidence(candidate, "candidate")
    for key in REQUIRED_EQUAL:
        if reference.get(key) != candidate.get(key):
            raise ValueError(f"mismatched scoring/corpus tokenization: {key}")
    if reference.get("protocol") != "kiln-reset-window-next-token-nll-v1":
        raise ValueError("unknown perplexity scoring protocol")
    scored = reference.get("scored_target_tokens")
    if type(scored) is not int or scored <= 0:
        raise ValueError("no scored tokens")
    reference_ppl = reference.get("perplexity")
    candidate_ppl = candidate.get("perplexity")
    if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0
           for x in (reference_ppl, candidate_ppl)):
        raise ValueError("invalid perplexity result")
    relative_loss = candidate_ppl / reference_ppl - 1
    # A decimal boundary such as 10.1 / 10.0 can round above exactly
    # 1%; accept only machine-rounding noise, not a material regression.
    within_limit = (relative_loss <= max_relative_ppl_loss
        or math.isclose(relative_loss, max_relative_ppl_loss,
                        rel_tol=1e-12, abs_tol=1e-12))
    return {
        "status": "pass" if within_limit else "fail",
        "protocol": reference["protocol"],
        "evaluated_tokens": scored,
        "token_fingerprint_fnv1a64": reference["token_fingerprint_fnv1a64"],
        "reference_perplexity": reference_ppl,
        "candidate_perplexity": candidate_ppl,
        "relative_perplexity_loss": relative_loss,
        "max_allowed_relative_perplexity_loss": max_relative_ppl_loss,
        "tier0_deployment_accepted": False,
        "note": "A controlled parity comparison, not proof of Tier-0 speed/memory or official WikiText-2 result.",
    }

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--max-relative-ppl-loss", type=float, default=0.01)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    try:
        verdict = assess(reference, candidate, args.max_relative_ppl_loss)
    except ValueError as exc:
        raise SystemExit(f"quality parity INVALID: {exc}")
    payload = json.dumps(verdict, indent=2) + "\n"
    if args.out:
        with args.out.open("x", encoding="utf-8") as out:
            out.write(payload)
    else:
        print(payload, end="")
    if verdict["status"] != "pass":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
