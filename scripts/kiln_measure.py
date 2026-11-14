#!/usr/bin/env python3
"""One-command KILN measurement suite; all raw logs are kept in its run folder."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def model_block_device_snapshot(model: Path) -> dict | None:
    """Read whole-device Linux block counters for the model's backing device."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        model_stat = model.stat()
        major, minor = os.major(model_stat.st_dev), os.minor(model_stat.st_dev)
        node = Path("/sys/dev/block") / f"{major}:{minor}"
        resolved = node.resolve(strict=True)
        # Report the parent disk for partitions when sysfs exposes it.
        if (resolved / "partition").is_file():
            resolved = resolved.parent
        fields = (resolved / "stat").read_text(encoding="ascii").split()
        if len(fields) < 4:
            return None
        reads, sectors, read_ms = int(fields[0]), int(fields[2]), int(fields[3])
        return {
            "device": resolved.name,
            "model_major_minor": f"{major}:{minor}",
            "read_ios": reads,
            "read_sectors": sectors,
            "read_bytes": sectors * 512,
            "read_time_ms": read_ms,
            "scope": "whole block device; includes unrelated processes and may be a virtual device",
        }
    except (OSError, ValueError):
        return None


def model_block_device_delta(before: dict | None, after: dict | None) -> dict:
    if not before or not after or before.get("device") != after.get("device"):
        return {"status": "unavailable"}
    counters = ("read_ios", "read_sectors", "read_bytes", "read_time_ms")
    if any(after.get(key, -1) < before.get(key, 0) for key in counters):
        return {"status": "counter_reset", "device": after["device"]}
    return {
        "status": "captured",
        "device": after["device"],
        "model_major_minor": after.get("model_major_minor"),
        "read_ios_delta": after["read_ios"] - before["read_ios"],
        "read_sectors_delta": after["read_sectors"] - before["read_sectors"],
        "read_bytes_delta": after["read_bytes"] - before["read_bytes"],
        "read_time_ms_delta": after["read_time_ms"] - before["read_time_ms"],
        "scope": after["scope"],
    }


class Run:
    def __init__(self, out: Path, model: Path | None = None):
        self.out = out
        self.model = model
        self.stages: list[dict] = []
        self.commands = out / "commands.jsonl"

    def stage(self, name: str, argv: list[str], *, env: dict[str, str] | None = None,
              stdout: str | None = None, stderr: str | None = None,
              skip_reason: str | None = None) -> dict:
        record = {"name": name, "command": argv, "status": "skipped" if skip_reason else "running"}
        if skip_reason:
            record["reason"] = skip_reason
            self.stages.append(record)
            self._write_command(record)
            return record
        out_path = self.out / (stdout or f"{name}.stdout.log")
        err_path = self.out / (stderr or f"{name}.stderr.log")
        record.update({"stdout_log": str(out_path), "stderr_log": str(err_path)})
        storage_before = model_block_device_snapshot(self.model) if self.model else None
        self._write_command(record)
        try:
            with out_path.open("wb") as out_stream, err_path.open("wb") as err_stream:
                result = subprocess.run(argv, cwd=ROOT, env={**os.environ, **(env or {})},
                                        stdin=subprocess.DEVNULL, stdout=out_stream,
                                        stderr=err_stream, check=False)
            record["exit_code"] = result.returncode
            record["status"] = "passed" if result.returncode == 0 else "failed"
        except OSError as error:
            record["status"] = "failed"
            record["error"] = str(error)
        if self.model:
            storage_after = model_block_device_snapshot(self.model)
            record["storage_device_io"] = model_block_device_delta(storage_before, storage_after)
        self.stages.append(record)
        self._write_command(record)
        return record

    def _write_command(self, record: dict) -> None:
        with self.commands.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def mean_metric(report: dict | None, metric: str) -> float | None:
    if not report:
        return None
    value = report.get("statistics", {}).get(metric, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) else None


def compare_token_outputs(reference: dict | None, candidate: dict | None) -> dict:
    """Fail closed unless every paired greedy run emitted identical tokens."""
    if not reference or not candidate:
        return {"status": "inconclusive", "matched_runs": 0, "total_runs": 0,
                "reason": "baseline or candidate report is missing"}
    expected, observed = reference.get("runs"), candidate.get("runs")
    if not isinstance(expected, list) or not isinstance(observed, list) or not expected or len(expected) != len(observed):
        return {"status": "fail", "matched_runs": 0,
                "total_runs": len(expected) if isinstance(expected, list) else 0,
                "reason": "reports do not contain the same nonempty run count"}
    mismatches = []
    for index, (left, right) in enumerate(zip(expected, observed)):
        required = ("generated_tokens", "token_fingerprint_fnv1a64", "token_ids_sha256_le_i32")
        if not isinstance(left, dict) or not isinstance(right, dict) or any(
                key not in left or key not in right for key in required):
            return {"status": "fail", "matched_runs": index, "total_runs": len(expected),
                    "reason": f"run {index + 1} is missing token parity fields"}
        for row in (left, right):
            count, fnv, digest = (row["generated_tokens"], row["token_fingerprint_fnv1a64"],
                                  row["token_ids_sha256_le_i32"])
            if (type(count) is not int or count < 0 or not isinstance(fnv, str) or len(fnv) != 16
                    or not isinstance(digest, str) or len(digest) != 64):
                return {"status": "fail", "matched_runs": index, "total_runs": len(expected),
                        "reason": f"run {index + 1} has invalid token parity fields"}
            try:
                int(fnv, 16)
                int(digest, 16)
            except ValueError:
                return {"status": "fail", "matched_runs": index, "total_runs": len(expected),
                        "reason": f"run {index + 1} has non-hexadecimal token hashes"}
        if any(left[key] != right[key] for key in required):
            mismatches.append(index + 1)
    return {"status": "fail" if mismatches else "pass",
            "matched_runs": len(expected) - len(mismatches), "total_runs": len(expected),
            "mismatched_run_numbers": mismatches,
            "comparison": "generated token count plus FNV-1a and SHA-256 of little-endian token IDs"}


def peak_rss_gib(report: dict | None) -> float | None:
    if not report:
        return None
    value = report.get("process_memory", {}).get("after_benchmark", {}).get("peak_rss_to_date")
    if isinstance(value, (int, float)):
        return float(value) * 1024 / (1024 ** 3)  # KiB to GiB
    return None


def read_audit(path: Path) -> dict | None:
    if not path.is_file():
        return None
    result = {"records": 0, "requested": 0, "loaded": 0, "expert_bytes_served": 0,
              "pread_bytes": 0, "cache_hits": 0, "malformed_records": 0}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) not in (4, 6):
                result["malformed_records"] += 1
                continue
            try:
                requested, loaded, served = map(int, fields[1:4])
                pread = int(fields[4]) if len(fields) == 6 else served
                hits = int(fields[5]) if len(fields) == 6 else 0
            except ValueError:
                result["malformed_records"] += 1
                continue
            result["records"] += 1
            result["requested"] += requested
            result["loaded"] += loaded
            result["expert_bytes_served"] += served
            result["pread_bytes"] += pread
            result["cache_hits"] += hits
    except (OSError, UnicodeError):
        return None
    result["load_completion_ratio"] = result["loaded"] / result["requested"] if result["requested"] else None
    return result


def host_info(model: Path, model_sha256: str) -> dict:
    try:
        model_size = model.stat().st_size
    except OSError:
        model_size = None
    info = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(), "platform": platform.platform(),
        "architecture": platform.machine(), "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": sys.version, "model_path": str(model),
        "model_size_bytes": model_size, "model_sha256": model_sha256,
        "model_filesystem_free_bytes": shutil.disk_usage(model.parent).free,
    }
    try:
        info["cpuinfo"] = next(line.strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                                if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    try:
        info["memory_info"] = {
            key: value for line in Path("/proc/meminfo").read_text().splitlines()
            if (parts := line.split(":", 1)) and parts[0] in {"MemTotal", "MemAvailable", "SwapTotal"}
            for key, value in [(parts[0], parts[1].strip())]
        }
    except OSError:
        pass
    cgroup = Path("/sys/fs/cgroup")
    if cgroup.exists():
        info["cgroup_limits"] = {}
        for name in ("memory.max", "cpu.max", "memory.swap.max"):
            value = cgroup / name
            if value.is_file():
                try:
                    info["cgroup_limits"][name] = value.read_text().strip()
                except OSError:
                    continue
    cpu_freq = Path("/sys/devices/system/cpu/cpu0/cpufreq")
    if cpu_freq.exists():
        info["cpu_frequency"] = {}
        for name in ("scaling_governor", "scaling_cur_freq", "scaling_min_freq",
                     "scaling_max_freq", "cpuinfo_max_freq"):
            value = cpu_freq / name
            if value.is_file():
                try:
                    info["cpu_frequency"][name] = value.read_text().strip()
                except OSError:
                    continue
    devices = []
    for path in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
        device = {"name": path.parent.name}
        for key in ("vendor", "device", "class"):
            try:
                device[key] = (path / key).read_text().strip()
            except OSError:
                pass
        if len(device) > 1:
            devices.append(device)
    if devices:
        info["drm_devices"] = devices
    findmnt = shutil.which("findmnt")
    if findmnt:
        mount = subprocess.run([findmnt, "--json", "--target", str(model),
                                "--output", "SOURCE,FSTYPE,OPTIONS"],
                               text=True, capture_output=True, check=False)
        if mount.returncode == 0:
            try:
                info["model_filesystem"] = json.loads(mount.stdout)
            except json.JSONDecodeError:
                info["model_filesystem_raw"] = mount.stdout.strip()
    thermal = Path("/sys/class/thermal")
    if thermal.exists():
        info["thermal_zones"] = {}
        for temp in sorted(thermal.glob("thermal_zone*/temp")):
            try:
                info["thermal_zones"][str(temp.parent.name)] = temp.read_text().strip()
            except OSError:
                continue
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--cache-mib", type=int, default=64)
    parser.add_argument("--min-tps", type=float, default=3.0)
    parser.add_argument("--max-rss-gib", type=float, default=15.0)
    args = parser.parse_args()
    try:
        model = args.model.expanduser().resolve(strict=True)
        if not model.is_file() or model.stat().st_size == 0:
            raise ValueError("--model must be a non-empty local regular file")
        if args.reference is not None:
            reference = args.reference.expanduser().resolve(strict=True)
            if not reference.is_file() or args.corpus is None:
                raise ValueError("--reference requires a readable reference GGUF and --corpus")
        else:
            reference = None
        if args.corpus is not None:
            corpus = args.corpus.expanduser().resolve(strict=True)
            if not corpus.is_file() or corpus.stat().st_size == 0:
                raise ValueError("--corpus must be a non-empty UTF-8 file")
        else:
            corpus = None
        if (args.tokens < 1 or args.runs < 1 or args.cache_mib < 0 or args.cache_mib > 512
                or args.threads == 0 or args.min_tps < 0 or args.max_rss_gib <= 0):
            raise ValueError("tokens/runs/threads/cache and target limits are outside their valid range")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    if args.out is None:
        out = ROOT / f"kiln-measure-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    else:
        out = args.out.expanduser().resolve()
    if out.exists():
        parser.error(f"refusing to overwrite existing run directory: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(mode=0o700)
    os.chmod(out, 0o700)
    runner = Run(out, model)
    model_identity = file_identity(model)
    model_digest = sha256(model)
    if file_identity(model) != model_identity:
        parser.error("model changed while its initial digest was being captured")
    (out / "host-and-model.json").write_text(json.dumps(host_info(model, model_digest), indent=2) + "\n")
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                  capture_output=True, check=False)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True,
                               capture_output=True, check=False)
        submodule = subprocess.run(["git", "submodule", "status", "vendor/llama.cpp"], cwd=ROOT,
                                   text=True, capture_output=True, check=False)
        provenance = {
            "kiln_commit": revision.stdout.strip() if revision.returncode == 0 else None,
            "kiln_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
            "llama_submodule": submodule.stdout.strip() if submodule.returncode == 0 else None,
            "build_profile": "release", "model_sha256": model_digest,
            "scope_note": "A single host run; Linux process I/O is not physical SSD telemetry.",
        }
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    except OSError as error:
        (out / "provenance-error.txt").write_text(str(error) + "\n")

    kiln = ROOT / "target/release/kiln"
    bench = ROOT / "target/release/kiln-bench"
    failures: list[str] = []
    blocking_failures: list[str] = []

    def record_failure(stage: dict, *, blocking: bool = True) -> None:
        if stage["status"] == "failed":
            failures.append(stage["name"])
            if blocking:
                blocking_failures.append(stage["name"])

    stage = runner.stage("build-release", ["cargo", "build", "--release", "-p", "kiln-cli", "-p", "kiln-bench"],
                         stdout="build.log", stderr="build.stderr.log")
    record_failure(stage)
    build_ok = stage["status"] == "passed" and kiln.is_file() and bench.is_file()
    if build_ok:
        stage = runner.stage("preflight-plan", [str(kiln), "plan", str(model), "--json"],
                             stdout="model-plan.json", stderr="model-plan.stderr.log")
        record_failure(stage, blocking=False)
        trace = out / "router-trace.tsv"
        bench_args = [str(kiln), "bench", str(model), "--native-greedy", "--tokens", str(args.tokens),
                      "--warmup", "1", "--runs", str(args.runs), "--json"]
        if args.threads:
            bench_args += ["--threads", str(args.threads)]
        stage = runner.stage("resident-baseline", bench_args,
                             env={"KILN_MOE_ROUTER_TRACE": str(trace)},
                             stdout="resident-baseline.json", stderr="resident-baseline.stderr.log")
        record_failure(stage)

        index = out / "native-expert-index.tsv"
        index_report = out / "native-expert-index.json"
        stage = runner.stage("native-index-export", [str(bench), "expert-export-gguf-stream-index",
                             "--source-gguf", str(model), "--auto-verified",
                             "--output-file", str(index), "--json"],
                             stdout="native-expert-index.json", stderr="native-expert-index.stderr.log")
        record_failure(stage, blocking=False)  # Unsupported MoE layout is a coverage gap, not a baseline failure.
        index_ok = stage["status"] == "passed" and index.is_file() and index.stat().st_size > 0

        if trace.is_file() and trace.stat().st_size > 0:
            if index_ok:
                stage = runner.stage("router-source-io", [sys.executable,
                                     str(ROOT / "scripts/capture_direct_gguf_expert_io.py"),
                                     str(model), str(index), str(index_report), str(trace),
                                     "--runs", str(args.runs), "--max-read-mib", "1024",
                                     "--output", str(out / "router-source-io.json")],
                                     stdout="router-source-io-summary.json", stderr="router-source-io.stderr.log")
                record_failure(stage, blocking=False)
                stage = runner.stage("cache-trace-replay", [sys.executable,
                                     str(ROOT / "scripts/replay_moe_cache_trace.py"), str(model),
                                     str(index), str(index_report), str(trace), "--cache-mib",
                                     "0", str(args.cache_mib), "--output", str(out / "cache-replay.json")],
                                     stdout="cache-replay-summary.json", stderr="cache-replay.stderr.log")
                record_failure(stage, blocking=False)
            else:
                runner.stage("router-source-io", [], skip_reason="verified native expert index unavailable")
                runner.stage("cache-trace-replay", [], skip_reason="verified native expert index unavailable")
            grouped_trace = out / "router-holdout-input.json"
            stage = runner.stage("group-router-holdout-input", [sys.executable,
                                 str(ROOT / "scripts/convert_moe_router_trace.py"), str(trace),
                                 "--out", str(out / "router-operator-events.json"),
                                 "--grouped-out", str(grouped_trace)],
                                 stdout="router-trace-conversion.json",
                                 stderr="router-trace-conversion.stderr.log")
            record_failure(stage, blocking=False)
            try:
                grouped_events = json.loads(grouped_trace.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                grouped_events = None
            if stage["status"] == "passed" and isinstance(grouped_events, list):
                token_count = len({event["token_position"] for event in grouped_events})
                train_until = int(token_count * 0.8)
                if 2 <= train_until < token_count:
                    stage = runner.stage("router-predictor-holdout", [str(bench), "router-holdout",
                                         "--trace-file", str(grouped_trace),
                                         "--train-until-token", str(train_until),
                                         "--top-k", "2", "--json"],
                                         stdout="router-predictor-holdout.json",
                                         stderr="router-predictor-holdout.stderr.log")
                    record_failure(stage, blocking=False)
                else:
                    runner.stage("router-predictor-holdout", [], skip_reason=
                                 f"need at least 3 distinct decode rows; captured {token_count}")
            else:
                runner.stage("router-predictor-holdout", [], skip_reason=
                             "router trace lacks usable decode-boundary grouped events; see conversion logs")
        else:
            runner.stage("router-source-io", [], skip_reason="trace or verified single-file MoE index unavailable")
            runner.stage("cache-trace-replay", [], skip_reason="trace or verified single-file MoE index unavailable")
            runner.stage("group-router-holdout-input", [], skip_reason="decode-marked router trace unavailable")
            runner.stage("router-predictor-holdout", [], skip_reason="decode-marked router trace unavailable")

        # Execute the real graph-selected source path under no-cache, LRU and LFU.
        # These are paired inference runs; replay above is separately labeled simulation.
        if index_ok:
            for policy in ("none", "lru", "lfu"):
                audit = out / f"stream-{policy}.audit.tsv"
                run_args = [str(kiln), "bench", str(model), "--native-greedy", "--tokens",
                            str(args.tokens), "--warmup", "1", "--runs", str(args.runs), "--json"]
                if args.threads:
                    run_args += ["--threads", str(args.threads)]
                env = {"KILN_MOE_STREAM_BLOB": str(model),
                       "KILN_MOE_STREAM_INDEX": str(index),
                       "KILN_MOE_STREAM_AUDIT": str(audit),
                       "KILN_MOE_STREAM_CACHE_POLICY": policy,
                       "KILN_MOE_STREAM_CACHE_MIB": str(args.cache_mib)}
                stage = runner.stage(f"stream-{policy}", run_args, env=env,
                                     stdout=f"stream-{policy}.json",
                                     stderr=f"stream-{policy}.stderr.log")
                record_failure(stage)
        else:
            for policy in ("none", "lru", "lfu"):
                runner.stage(f"stream-{policy}", [], skip_reason="verified native expert index unavailable")

        if reference and corpus:
            for label, quality_model in (("reference", reference), ("candidate", model)):
                stage = runner.stage(f"quality-{label}", [str(kiln), "eval-ppl", str(quality_model),
                                     "--text-file", str(corpus), "--threads", str(args.threads or 1), "--json"],
                                     stdout=f"quality-{label}.json", stderr=f"quality-{label}.stderr.log")
                record_failure(stage)
            if read_json(out / "quality-reference.json") and read_json(out / "quality-candidate.json"):
                stage = runner.stage("quality-parity", [sys.executable,
                                     str(ROOT / "scripts/compare_quality_parity.py"),
                                     "--reference", str(out / "quality-reference.json"),
                                     "--candidate", str(out / "quality-candidate.json"),
                                     "--out", str(out / "quality-parity.json")],
                                     stdout="quality-parity.stdout.log", stderr="quality-parity.stderr.log")
                record_failure(stage)
            else:
                runner.stage("quality-parity", [], skip_reason="one or both NLL scoring runs did not produce JSON")
        else:
            runner.stage("quality-reference", [], skip_reason="supply --reference and --corpus to measure quality parity")
            runner.stage("quality-candidate", [], skip_reason="supply --reference and --corpus to measure quality parity")
            runner.stage("quality-parity", [], skip_reason="FP16 reference and matched corpus were not supplied")
    else:
        for name in ("preflight-plan", "resident-baseline", "native-index-export", "router-source-io",
                     "cache-trace-replay", "group-router-holdout-input", "router-predictor-holdout",
                     "stream-none", "stream-lru", "stream-lfu",
                     "quality-reference", "quality-candidate", "quality-parity"):
            runner.stage(name, [], skip_reason="release build failed; see build.log and build.stderr.log")

    reports = {name: read_json(out / filename) for name, filename in {
        "resident": "resident-baseline.json", "stream_none": "stream-none.json",
        "stream_lru": "stream-lru.json", "stream_lfu": "stream-lfu.json",
        "quality": "quality-parity.json", "cache_replay": "cache-replay.json",
        "predictor": "router-predictor-holdout.json",
    }.items()}
    metrics = {}
    parity = {}
    for name, report in reports.items():
        if name.startswith("stream_") or name == "resident":
            policy = name.removeprefix("stream_")
            token_comparison = compare_token_outputs(reports["resident"], report) if policy != "resident" else None
            if token_comparison is not None:
                parity[policy] = token_comparison
            stage_name = "resident-baseline" if policy == "resident" else f"stream-{policy}"
            stage_record = next((stage for stage in runner.stages if stage.get("name") == stage_name), {})
            metrics[name] = {
                "decode_tokens_per_second_mean": mean_metric(report, "decode_tokens_per_second"),
                "decode_forward_tokens_per_second_mean": mean_metric(report, "decode_forward_tokens_per_second"),
                "peak_process_rss_gib": peak_rss_gib(report),
                "kernel_attributed_read_bytes": (report or {}).get("process_io", {})
                    .get("measured_inference", {}).get("kernel_storage_read_bytes_delta"),
                "expert_io_audit": read_audit(out / f"stream-{policy}.audit.tsv") if policy != "resident" else None,
                "token_parity_vs_resident": token_comparison,
                "host_block_device_io": stage_record.get("storage_device_io"),
            }
    quality = reports["quality"]
    predictor = reports["predictor"]
    source_io = read_json(out / "router-source-io.json")
    try:
        model_stable = file_identity(model) == model_identity
    except OSError:
        model_stable = False
    if not model_stable:
        failures.append("model-file-stability")
        blocking_failures.append("model-file-stability")
        record = {"name": "model-file-stability", "command": [], "status": "failed",
                  "reason": "model file identity, size, or modification time changed during measurement"}
        runner.stages.append(record)
        runner._write_command(record)
    primary = reports["stream_none"] or reports["resident"]
    tps = mean_metric(primary, "decode_tokens_per_second")
    rss = peak_rss_gib(primary)
    speed_status = "inconclusive" if tps is None else ("pass" if tps >= args.min_tps else "fail")
    memory_status = "inconclusive" if rss is None else ("pass" if rss < args.max_rss_gib else "fail")
    quality_status = quality.get("status") if quality else "inconclusive"
    parity_status = ("fail" if any(item["status"] == "fail" for item in parity.values())
                     else "pass" if parity and all(item["status"] == "pass" for item in parity.values())
                     else "inconclusive")
    observed_failure = bool(blocking_failures) or "fail" in (speed_status, memory_status, quality_status, parity_status)
    summary = {
        "schema_version": 1,
        "run_directory": str(out),
        "model_sha256": model_digest,
        "overall_status": "fail" if observed_failure else "inconclusive",
        "acceptance_claim": False,
        "checks": {
            "decode_speed": {"status": speed_status, "observed_tps": tps,
                              "minimum_tps": args.min_tps,
                              "basis": "stream-none when supported, otherwise resident baseline"},
            "process_peak_rss": {"status": memory_status, "observed_gib": rss,
                                  "maximum_gib": args.max_rss_gib,
                                  "basis": "process peak RSS; excludes OS page cache"},
            "quality_parity": {"status": quality_status,
                               "report": "quality-parity.json" if quality else None},
            "generated_token_parity": {"status": parity_status,
                                       "per_policy": parity,
                                       "basis": "generated-token count and two token-ID hashes against the resident baseline"},
            "model_file_stability": {"status": "pass" if model_stable else "fail",
                                     "basis": "device, inode, size, and modification time compared before and after measurement"},
            "phase5_target_hardware_and_sustained_acceptance": {
                "status": "inconclusive",
                "reason": "A single run does not prove the named Tier 0 host, sustained behavior, or all roadmap gates.",
            },
            "physical_storage_traffic": {
                "status": "not_measured",
                "host_block_device_deltas": {stage.get("name"): stage.get("storage_device_io")
                                              for stage in runner.stages
                                              if stage.get("storage_device_io", {}).get("status") == "captured"},
                "direct_io_diagnostic": ((source_io or {}).get("direct_io_diagnostic")
                                          if source_io else None),
                "reason": "Block counters cover the whole backing device, not just KILN. O_DIRECT bypasses the OS page cache when supported, but does not prove physical-media completion or isolate inference traffic.",
            },
        },
        "phase5_gate_coverage": {
            "5A_storage_ram_compute": {
                "status": "partial" if reports["cache_replay"] else "inconclusive",
                "evidence": ["live process RSS and process-attributed I/O", "indexed source expert-read replay when supported"] + (["best-effort O_DIRECT replay of traced expert ranges"] if (source_io or {}).get("direct_io_diagnostic", {}).get("status") == "captured" else []) + (["host-wide model-device read-sector/time deltas around measurement stages"] if any(stage.get("storage_device_io", {}).get("status") == "captured" for stage in runner.stages) else []),
                "still_needed": ["repeated target-host storage runs and isolated device telemetry", "full host memory and thermal acceptance run"],
            },
            "5B_cache_and_predictor": {
                "status": "partial" if reports["cache_replay"] or predictor else "inconclusive",
                "evidence": ["live no-cache/LRU/LFU runs when native indexing is supported", "offline LRU/LFU/adaptive policy simulation"] + (["chronological real-router grouped holdout"] if predictor else []),
                "predictor_holdout": ({key: predictor.get(key) for key in
                    ("train_events", "test_events", "top_k", "accuracy_at_k", "coverage",
                     "active_expert_recall_at_k", "unigram_baseline_accuracy_at_k",
                     "exploratory_threshold_80_percent")} if predictor else None),
                "still_needed": ["prefetch useful/wasted bytes and end-to-end impact"],
            },
            "5C_async_io_overlap": {
                "status": "not_measured",
                "still_needed": ["async queue depth, completion latency, overlap, CPU cost, and platform comparison"],
            },
            "5D_end_to_end_quality_and_acceptance": {
                "status": "partial" if quality else "inconclusive",
                "evidence": ["resident and selected-expert inference throughput/RSS when supported"] + (["exact generated-token parity across live cache policies"] if parity_status == "pass" else []) + (["matched-corpus reference parity"] if quality else []),
                "still_needed": ["named Tier 0 host and sustained acceptance-duration run", "official corpus and model/weight/tokenizer pin where applicable"],
            },
        },
        "live_metrics": metrics,
        "cache_replay": {
            "status": (reports["cache_replay"] or {}).get("status", "inconclusive"),
            "kind": "offline trace simulation; not runtime cache performance",
            "report": "cache-replay.json" if reports["cache_replay"] else None,
        },
        "predictor_holdout": {"status": "captured" if predictor else "inconclusive",
                              "kind": "chronological grouped holdout; exploratory only",
                              "report": "router-predictor-holdout.json" if predictor else None},
        "stages": runner.stages,
        "failed_stages": failures,
        "blocking_failures": blocking_failures,
        "logs": {"commands": "commands.jsonl", "build": "build.log", "stderr_pattern": "*.stderr.log"},
        "interpretation": "Inconclusive means evidence is missing or target acceptance conditions were not established. Review every stage log; failed stages do not erase artifacts from successful stages.",
    }
    (out / "host-after.json").write_text(json.dumps(host_info(model, model_digest), indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["overall_status"], "run_directory": str(out),
                      "summary": str(out / "summary.json"), "failed_stages": failures}, indent=2))
    return 1 if observed_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
