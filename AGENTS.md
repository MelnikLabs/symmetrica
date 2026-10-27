# AGENTS.md — symmetrica

Authoritative contributor and coding-agent guidance for Symmetrica. Read this index and the topic guidance relevant to the task before changing code. When project guidance conflicts with generic language or tool advice, follow the project guidance and raise unclear or contradictory policy before acting.

| Field | Value |
|---|---|
| **Project status** | WIP — no stable public API |
| **Owner** | MelnikLabs |
| **Guidance version** | 2026-03 |
| **Review cadence** | Update the relevant guidance when architecture decisions change. |
| **Escalation path** | Open an architecture issue and obtain the required review before changing the workspace dependency graph or a prohibited policy. |

## Read guidance by task

Before changing code or project guidance, read every topic below that applies to the task. This index routes to the authoritative details; it does not replace them.

| When working on… | Read |
|---|---|
| Crate boundaries, dependencies, project scope, or architecture decisions | [Architecture and crate boundaries](docs/agent-guidance/architecture-and-crates.md) |
| Rust edition/nightly features, type-level invariants, or shape algebra | [Rust, types, and shapes](docs/agent-guidance/rust-and-type-system.md) |
| Derivatives, pullbacks, checkpoints, or differentiability | [Structural autodiff](docs/agent-guidance/structural-autodiff.md) |
| Floating-point behavior, stability, or gradient checks | [Numerical reliability](docs/agent-guidance/numerical-reliability.md) |
| Randomness, purity, mutation, aliasing, or tensor ownership | [Purity and ownership](docs/agent-guidance/purity-and-ownership.md) |
| Onboarding, unsafe code, tests, docs, prohibited items, or change process | [Engineering practices](docs/agent-guidance/engineering-practices.md) |
| Backend kernels, fusion, compile time, or benchmarks | [Backend execution and performance](docs/agent-guidance/backend-performance.md) |
| Rule IDs, enforcement priorities, or detection gaps | [Enforcement reference](docs/agent-guidance/enforcement-reference.md) |

## Always-on guardrails

- Inspect the current code and repository state; guidance does not replace checking the implementation.
- Preserve the dependency direction: `algebra` has no workspace dependencies; `backend` may depend on `algebra`; `tensor` may depend on `algebra` and the abstract backend contract.
- Keep tensor semantics type-driven and autodiff structural. Follow the detailed type-system and autodiff policies linked above.
- Treat the prohibited list as blocking. Changing it or the workspace dependency graph requires an architecture issue and the required review.
