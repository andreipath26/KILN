#!/usr/bin/env python3
"""Run the retained Tier 0 evidence captures for Phases 3.5, 5 and 6.

The aggregate report is deliberately never a deployment certificate: target
identity, manual quality review, sustained thermals and several physical-I/O
gates still need an explicit review after the run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve_file(base: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    try:
        path = path.resolve(strict=True)
    except OSError:
        return None
    return path if path.is_file() and os.access(path, os.R_OK) else None


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def file_identity(path: Path | None) -> tuple[int, int, int, int] | None:
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


class Capture:
    def __init__(self, out: Path):
        self.out = out
        self.records: list[dict] = []

    def run(self, name: str, command: list[str], *, env: dict[str, str] | None = None) -> bool:
        stdout_path = self.out / f"{name}.stdout.log"
        stderr_path = self.out / f"{name}.stderr.log"
        started = time.monotonic()
        record = {"name": name, "command": command, "status": "failed",
                  "exit_code": None, "duration_seconds": None,
                  "stdout_log": stdout_path.name, "stderr_log": stderr_path.name}
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                result = subprocess.run(command, cwd=ROOT, env={**os.environ, **(env or {})},
                                        stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                        check=False)
            record["exit_code"] = result.returncode
            record["status"] = "passed" if result.returncode == 0 else "failed"
        except OSError as error:
            stderr_path.write_text(f"could not run command: {error}\n", encoding="utf-8")
            record["error"] = str(error)
        finally:
            record["duration_seconds"] = round(time.monotonic() - started, 3)
            self.records.append(record)
            self.write_records()
        return record["status"] == "passed"

    def blocked(self, name: str, reason: str) -> None:
        self.records.append({"name": name, "status": "blocked", "reason": reason})
        self.write_records()

    def write_records(self) -> None:
        (self.out / "commands.json").write_text(
            json.dumps(self.records, indent=2) + "\n", encoding="utf-8")


def machine_info() -> dict:
    result = {"captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "platform": platform.platform(), "architecture": platform.machine(),
              "processor": platform.processor(), "logical_cpus": os.cpu_count(),
              "tier0_machine_verified": False}
    try:
        result["cpuinfo"] = next(line.split(":", 1)[1].strip()
                                 for line in Path("/proc/cpuinfo").read_text().splitlines()
                                 if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    try:
        result["memory_info"] = {
            key: value.strip() for line in Path("/proc/meminfo").read_text().splitlines()
            if ":" in line and (key := line.split(":", 1)[0]) in
            {"MemTotal", "MemAvailable", "SwapTotal"}
            for value in [line.split(":", 1)[1]]
        }
    except OSError:
        pass
    return result


def run(config_path: Path, out: Path) -> int:
    config_path = config_path.expanduser().resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("qualification config must be a JSON object")
    for section in ("phase35", "phase5", "phase6"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"qualification config needs a {section} object")

    out = out.expanduser().resolve()
    if out.exists():
        raise ValueError(f"refusing to overwrite existing output directory: {out}")
    out.mkdir(parents=True, mode=0o700)
    cap = Capture(out)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (out / "machine.json").write_text(json.dumps(machine_info(), indent=2) + "\n", encoding="utf-8")
    model_paths: dict[str, Path | None] = {}
    for section, key in (("phase35", "dense_model"), ("phase5", "moe_model"), ("phase6", "llada_model")):
        model_paths[key] = resolve_file(config_path.parent, config[section].get("model"))
    initial_file_ids = {key: file_identity(path) for key, path in model_paths.items()}

    build_ok = cap.run("build-release", ["cargo", "build", "--release", "-p", "kiln-cli", "-p", "kiln-bench"])
    kiln = ROOT / "target/release/kiln"
    phase_status: dict[str, str] = {}

    dense = model_paths["dense_model"]
    if dense is None:
        cap.blocked("phase35", "phase35.model must name a readable dense GGUF")
        phase_status["phase35"] = "blocked"
    elif not build_ok:
        cap.blocked("phase35", "release build failed; review build-release logs")
        phase_status["phase35"] = "blocked"
    else:
        settings = config["phase35"]
        env = {"KILN_BENCH_TOKENS": str(settings.get("tokens", 256)),
               "KILN_BENCH_RUNS": str(settings.get("runs", 5)),
               "KILN_BENCH_WARMUP": str(settings.get("warmup", 1)),
               "KILN_SKIP_BUILD": "1"}
        if settings.get("ollama_model"):
            env["KILN_OLLAMA_MODEL"] = str(settings["ollama_model"])
        phase_status["phase35"] = "captured" if cap.run(
            "phase35", ["bash", "scripts/run_phase35_suite.sh", str(dense), str(out / "phase35")], env=env
        ) else "failed"

    moe = model_paths["moe_model"]
    if moe is None:
        cap.blocked("phase5", "phase5.model must name a readable MoE GGUF")
        phase_status["phase5"] = "blocked"
    elif not build_ok or not kiln.is_file():
        cap.blocked("phase5", "release executable is unavailable; review build-release logs")
        phase_status["phase5"] = "blocked"
    else:
        settings = config["phase5"]
        command = [str(kiln), "measure", str(moe), "--out", str(out / "phase5"),
                   "--tokens", str(settings.get("tokens", 128)), "--runs", str(settings.get("runs", 5)),
                   "--min-tps", str(settings.get("min_tps", 3.0)),
                   "--max-rss-gib", str(settings.get("max_rss_gib", 15.0))]
        if settings.get("threads") is not None:
            command.extend(["--threads", str(settings["threads"])])
        reference = resolve_file(config_path.parent, settings.get("reference_model"))
        corpus = resolve_file(config_path.parent, settings.get("corpus"))
        if reference and corpus:
            command.extend(["--reference", str(reference), "--corpus", str(corpus)])
        elif settings.get("reference_model") or settings.get("corpus"):
            cap.blocked("phase5-quality", "both readable reference_model and corpus are required")
        phase_status["phase5"] = "captured" if cap.run("phase5", command) else "failed"

    llada = model_paths["llada_model"]
    if llada is None:
        cap.blocked("phase6", "phase6.model must name a readable LLaDA GGUF")
        phase_status["phase6"] = "blocked"
    elif not build_ok or not kiln.is_file():
        cap.blocked("phase6", "release executable is unavailable; review build-release logs")
        phase_status["phase6"] = "blocked"
    else:
        settings = config["phase6"]
        answers = out / "phase6-confidence-answers.jsonl"
        baseline = out / "phase6-confidence.json"
        clad = out / "phase6-clad.json"
        common = ["--kiln", str(kiln), "--model", str(llada),
                  "--repeats", str(settings.get("repeats", 3)),
                  "--max-tokens", str(settings.get("max_tokens", 32)),
                  "--block-length", str(settings.get("block_length", 32))]
        if settings.get("threads") is not None:
            common.extend(["--threads", str(settings["threads"])])
        baseline_ok = cap.run("phase6-confidence", [sys.executable, "scripts/run_phase6_clad_suite.py",
            *common, "--decoder", "confidence", "--answers-jsonl", str(answers), "--output", str(baseline)])
        if baseline_ok and answers.is_file() and baseline.is_file():
            try:
                model_sha = digest(llada)
                phase_status["phase6"] = "captured" if cap.run("phase6-clad", [
                    sys.executable, "scripts/run_phase6_clad_suite.py", *common,
                    "--decoder", "clad", "--reference-jsonl", str(answers),
                    "--reference-model-sha256", model_sha,
                    "--reference-runtime", "KILN confidence decoder baseline",
                    "--output", str(clad)]) else "failed"
            except OSError as error:
                cap.blocked("phase6-clad", f"could not hash LLaDA model: {error}")
                phase_status["phase6"] = "failed"
        else:
            cap.blocked("phase6-clad", "confidence baseline failed; see phase6-confidence logs")
            phase_status["phase6"] = "failed"

    model_file_stable = True
    model_report = {}
    for key, path in model_paths.items():
        if path is None:
            model_report[key] = None
            continue
        try:
            identity = file_identity(path)
            stable = identity == initial_file_ids[key]
            model_file_stable = model_file_stable and stable
            model_report[key] = {"path": str(path), "sha256_after_capture": digest(path),
                                 "identity_stable_during_capture": stable}
        except OSError as error:
            model_file_stable = False
            model_report[key] = {"path": str(path), "error": str(error),
                                 "identity_stable_during_capture": False}
    report = {
        "schema_version": 1,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "overall_status": "inconclusive",
        "deployment_ready": False,
        "phase_capture_status": phase_status,
        "models": model_report,
        "model_file_stability": "pass" if model_file_stable else "fail",
        "machine_report": "machine.json",
        "command_log": "commands.json",
        "limits": [
            "The runner never treats this aggregate as Tier 0 acceptance.",
            "Confirm Dell Latitude 7490 identity, AC/power and sustained thermal conditions.",
            "Review Phase 3.5 run-notes and every Phase 5/6 failure or quality miss.",
            "Phase 5 physical storage, total-system memory and sustained Qwen3-30B-A3B gates need target evidence.",
        ],
    }
    (out / "qualification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["overall_status"], "deployment_ready": False,
                      "output": str(out), "phases": phase_status}, indent=2))
    return 0 if (all(status == "captured" for status in phase_status.values())
                 and len(phase_status) == 3 and model_file_stable) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="JSON model and measurement settings")
    parser.add_argument("--out", required=True, type=Path, help="new directory for all phase reports and logs")
    args = parser.parse_args()
    try:
        return run(args.config, args.out)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Tier 0 qualification: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
