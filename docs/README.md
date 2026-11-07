# Documentation guide

KILN has one authoritative plan: the [master roadmap](../KILN%20MASTER%20ROADMAP%20AND%20RULES%20SET.md). It owns mission targets, phase order/status, engineering rules, research leads, and the dated evidence log. Other documents explain current technical boundaries or repeatable procedures; they do not change roadmap gates.

## Current references

- [Architecture and repository map](architecture.md) — runtime boundaries and verified implementation limits.
- [Quantization policy](quantization-policy.md) — the format-support decision record required by KILN-E35.
- [Phase 3.5 measurement guide](phase35-measurement.md) — dense-path capture, parity, profiling, and result interpretation.
- [Phase 5 streaming guide](phase5-streaming.md) — sidecar, I/O, cache, predictor, dispatch components, and open integration gates.
- [Predictor training and evaluation](training-pipeline-design.md) — Phase 5.0 trace requirements and leakage-resistant evaluation.
- [UMF and expert sidecar status](umf-design.md) — implemented GGUF-preserving sidecar versus unimplemented embedded-container proposal.
- [Runtime loading and release design](runtime-loading-design.md) — current linked development build and release requirements.
- [UI architecture and continuation guide](ui-architecture.md) — roadmap-wide UI responsibilities and implemented boundary.

## Reading rules

“Target,” “proposal,” “implemented,” “CI-verified,” “measured,” and “accepted” are different states. Use the roadmap for the current phase ledger and evidence. A merged component or passing CI does not establish hardware performance or end-to-end model support. Historical experiment results remain labeled with their original conditions.

The October 2026 standalone design drafts and one-file-per-experiment notes were consolidated into the references above. Git history retains their original versions; the current docs folder is the maintained reading path.
