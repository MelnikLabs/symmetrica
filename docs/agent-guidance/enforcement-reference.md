# Enforcement Reference

> For AI agents and automated tooling. Each rule carries a **priority label**
> that defines both urgency and blocking behaviour. Work violations in
> P0 → P1 → P2 → P3 order. Do not submit a PR with any P0 or P1 violation.

## Priority Definitions

| Priority | Meaning | Blocks Merge | Exception Path |
|---|---|---|---|
| **P0** | Architectural invariant — CI enforces automatically | ✅ Yes — hard block | None |
| **P1** | Architectural invariant — requires human verification | ✅ Yes — reviewer blocks | Architect sign-off, recorded in Issue |
| **P2** | Quality standard — flagged in review | ❌ No | Reviewer judgment |
| **P3** | Advisory — noted in PR comment | ❌ No | None needed |

---

## A.1 Rule Registry

| ID | Priority | Rule | Automation | Caveat |
|---|---|---|---|---|
| `R-001` | **P0** | No crate DAG violation | ✅ grep scoped per crate (see §A.4) | `tensor/` may import `Backend` trait from `backend/` — correct. Must never name a concrete impl type. |
| `R-002` | **P1** | No runtime autodiff tape | ⚠️ grep `Vec<Box<dyn` (partial) | Renamed types evade — human must verify new struct definitions |
| `R-003` | **P0** | Unsafe requires `// SAFETY:` | ✅ `clippy::undocumented_unsafe_blocks` | Do NOT replace with grep — clippy handles look-behind |
| `R-004` | **P1** | Stop-gradient must be structural | 🔴 Manual only | "gradient context" not machine-detectable |
| `R-005` | **P1** | No implicit broadcasting | 🔴 Manual only | Context-dependent shape analysis |
| `R-006` | **P1** | No full Hessian construction | 🔴 Manual only | Requires semantic understanding |
| `R-007` | **P0** | No hidden RNG in `algebra/`/`tensor/` | ✅ grep `rand::` scoped per crate | Method-only usage escapes — see §A.4 |
| `R-008` | **P0** | No `Box<dyn Trait>` in `algebra/`/`tensor/` | ✅ grep `Box<dyn` scoped per crate | None |
| `R-009` | **P2** | New trait needs `proptest` property tests | 🔴 Manual only | N/A |
| `R-010` | **P2** | `Differentiable` impl needs gradient check | ✅ `cargo test grad_check` | Low FP risk |
| `R-011` | **P2** | Float `Field`/`Ring` impl needs law doc | 🔴 Manual only | N/A |
| `R-012` | **P2** | Numerically sensitive op needs `// STABILITY:` | ⚠️ word-boundary grep `\bexp\(` etc. | High FP risk with bare `exp` pattern — use `\bexp\(` |
| `R-013` | **P2** | Non-smooth op needs `// NON-DIFFERENTIABLE:` | ⚠️ grep function names | Catches names, not all usages |
| `R-014` | **P2** | Nightly feature use needs `// MGCA:` comment | ⚠️ grep feature gate names | N/A |
| `R-015` | **P2** | Associated type selection follows the [regular-versus-GAT policy](./rust-and-type-system.md) | 🔴 Manual only | New lifetime on `Scalar`/`Gradient`/`Shape`/`Backend`/`Derivative` is a violation; missing lifetime on `View`/`Iter`/`Slice` is a violation |
| `R-016` | **P1** | Kernel fusion only in `backend/` Layer 2 | ⚠️ grep for fusion patterns in wrong crates (partial) | Semantic fusion (not named `fuse`) evades grep — human must verify new multi-op types in `algebra/`/`tensor/` |
| `R-017` | **P2** | Shape arithmetic lives only in `tensor::shape` | ⚠️ grep `shape` arithmetic outside `tensor/shape/` (partial) | N/A |
| `R-018` | **P2** | Compile-time regression >10% investigated before merge | ⚠️ `cargo build --timings` comparison | No automated gate — requires contributor to run and report |
| `R-019` | **P2** | New backend kernels have `criterion` benchmark | 🔴 Manual only | N/A |
| `R-020` | **P3** | Commit message follows `<crate>: <description>` format | 🔴 Manual only | Advisory |
| `R-021` | **P3** | Draft PR opened within 1 day of starting work | 🔴 Manual only | Advisory |

**P0 count: 4 rules fully automated.**
**P1 count: 5 rules require human review — they block merge but cannot be scripted.**
**P2 count: 9 rules are quality flags — they do not block merge.**
**P3 count: 2 advisory rules — noted but not enforced.**

---

## A.2 Enforcement Script

```bash
#!/bin/bash
# scripts/enforce_agents.sh
# Enforces P0 rules automatically. P1 rules require human review.
# P2/P3 rules are advisory — flagged but do not exit 1.

set -euo pipefail
FAILED=0

echo "=== AGENTS.md P0/P1 Enforcement ==="
echo "P0: auto-block | P1: human-review block | P2: advisory flag"
echo ""

# ── P0: R-001 — Crate DAG ─────────────────────────────────────────────────
if grep -rn "^use tensor::\|^use backend::" algebra/src/ 2>/dev/null; then
    echo "❌ P0 / R-001: algebra/ imports tensor/ or backend/ — forbidden"
    FAILED=1
fi
if grep -rn "CpuBackend\|CudaBackend\|WasmBackend" tensor/src/ 2>/dev/null; then
    echo "❌ P0 / R-001: tensor/ names a concrete backend type — forbidden"
    echo "   tensor/ must only use B: Backend trait bounds, never concrete structs"
    FAILED=1
fi

# ── P0: R-003 — Undocumented unsafe ───────────────────────────────────────
cargo clippy --workspace -- -D clippy::undocumented_unsafe_blocks 2>&1 \
    || { echo "❌ P0 / R-003: Undocumented unsafe block"; FAILED=1; }

# ── P0: R-007 — Hidden RNG ────────────────────────────────────────────────
if grep -rn "rand::" algebra/src/ tensor/src/ 2>/dev/null; then
    echo "❌ P0 / R-007: RNG import in algebra/ or tensor/"
    FAILED=1
fi

# ── P0: R-008 — Box<dyn Trait> in hot paths ───────────────────────────────
if grep -rn "Box<dyn" algebra/src/ tensor/src/ 2>/dev/null; then
    echo "❌ P0 / R-008: Box<dyn Trait> in algebra/ or tensor/"
    FAILED=1
fi

# ── P1: R-002 — Runtime tape (partial) ────────────────────────────────────
if grep -rn "Vec<Box<dyn" algebra/src/ tensor/src/ 2>/dev/null; then
    echo "⛔ P1 / R-002: Possible runtime tape — HUMAN REVIEW REQUIRED before merge"
    FAILED=1
fi

# ── P1: R-016 — Fusion outside backend/ (partial) ─────────────────────────
if grep -rn "fuse\|fused_\|FusedOp\|FusionKernel" algebra/src/ tensor/src/ 2>/dev/null; then
    echo "⛔ P1 / R-016: Possible kernel fusion in algebra/ or tensor/ — HUMAN REVIEW REQUIRED"
    echo "   Fusion is permitted only in backend/ Layer 2 (see backend-performance.md)"
    FAILED=1
fi

# ── Standard CI gates ─────────────────────────────────────────────────────
cargo test --workspace        || { echo "❌ Tests failed"; FAILED=1; }
cargo fmt --check             || { echo "❌ Format check failed"; FAILED=1; }
cargo doc --workspace --no-deps 2>&1 | grep "^error" \
    && { echo "❌ Doc build errors"; FAILED=1; } || true

# ── P2: Advisory flags (do not exit 1) ────────────────────────────────────
echo ""
echo "=== P2 Advisory Flags ==="

# R-010: Gradient check tests
cargo test grad_check --workspace 2>&1 \
    || echo "⚠️  P2 / R-010: grad_check tests missing or failing — add before merge"

# R-012: Stability annotations
if grep -rPn '\b(exp|ln|log|sqrt)\(' algebra/src/ tensor/src/ 2>/dev/null \
    | grep -v "// STABILITY:" | grep -v "exp_m1\|ln_1p"; then
    echo "⚠️  P2 / R-012: Numerically sensitive op without // STABILITY: comment"
fi

# R-013: Non-smooth ops
if grep -rn "\brelu\|\babs\|\bsign\b" algebra/src/ tensor/src/ 2>/dev/null \
    | grep -v "// NON-DIFFERENTIABLE:"; then
    echo "⚠️  P2 / R-013: Non-smooth op without // NON-DIFFERENTIABLE: annotation"
fi

# R-017: Shape arithmetic outside tensor::shape
if grep -rPn '\bshape\b.*[+\-\*]|\brank\b.*[+\-\*]' tensor/src/ 2>/dev/null \
    | grep -v "tensor/src/shape"; then
    echo "⚠️  P2 / R-017: Shape arithmetic outside tensor::shape — consolidate in the Rust, Types, and Shapes guidance"
fi

# R-018: Compile-time budget (advisory reminder — no automated gate)
echo "⚠️  P2 / R-018: Run 'cargo build --workspace --timings' and confirm <10% regression"

# R-019: Criterion benchmarks for new backend kernels (advisory reminder)
echo "⚠️  P2 / R-019: New backend/ kernels require criterion benchmark in backend/benches/"

# ── Final verdict ─────────────────────────────────────────────────────────
echo ""
if [ $FAILED -eq 1 ]; then
    echo "❌ P0/P1 violations found — merge blocked."
    echo "   P1 rules R-004 (stop-gradient), R-005 (broadcasting), R-006 (Hessian)"
    echo "   cannot be scripted — verify manually against the prohibited list in engineering-practices.md."
    exit 1
fi

echo "✅ P0 rules passed. P2/P3 flags above are advisory."
echo "   Manually verify P1 rules: R-002, R-004, R-005, R-006, R-016."
```

---

## A.3 Known Detection Gaps

| Rule | Gap | What Escapes | Human Mitigation |
|---|---|---|---|
| `R-001` | Concrete type in `tensor/` | `CpuBackend` named anywhere in `tensor/src/` | `grep -rn "CpuBackend\|CudaBackend" tensor/src/` — any hit is a violation |
| `R-001` | Aliased backend import | `use b = backend; b::CpuBackend` in `tensor/src/` | `tensor/` may depend on `backend/` for the trait; verify no concrete structs are used |
| `R-001` | Any workspace import in `algebra/` | `tensor::` or `backend::` in `algebra/src/` | Audit `algebra/Cargo.toml` — must list no symmetrica workspace crates |
| `R-002` | Renamed types | `struct Graph { ops: Vec<Box<dyn Backward>> }` | Review all new `struct` definitions in `algebra/`/`tensor/` |
| `R-007` | Method-only RNG | `thread_rng().gen::<f64>()` without `use rand::` | Flag any call containing `rng`, `random`, `gen` in `algebra/`/`tensor/` |
| `R-012` | False positives | `explanation`, `expect`, `export` contain `exp` | Script uses `\bexp\(` word-boundary — review flagged lines |
| `R-016` | Semantic fusion | Multi-op composite type not named `fuse*` | Review any new struct in `algebra/`/`tensor/` that wraps two or more operation types |
| `R-017` | Shape logic in op impls | Inline `a_rank + b_rank - 2*k` in `contract()` body | Any shape arithmetic outside `tensor/src/shape/` is a violation — search for arithmetic on `rank`/`shape` variables |

---

## A.4 Priority Quick-Reference for Agents

When encountering a rule violation, apply this decision tree:

```
Is the violation detected by the CI script?
├── YES → CI will block the PR automatically (P0). Fix before pushing.
└── NO  → Is it in the P1 list (R-002, R-004, R-005, R-006, R-016)?
           ├── YES → Do NOT proceed. Open a GitHub Issue tagged `architecture`.
           │         A human architect must review before merge.
           └── NO  → Is it in the P2 list (R-009 through R-019)?
                     ├── YES → Note it in the PR description. Does not block merge.
                     └── NO  → P3 advisory. Note if convenient.
```

**For AI agents specifically:** if you are uncertain which priority a potential
violation belongs to, default to P1 behaviour — stop and flag it. The cost of
a false P1 flag is one human review. The cost of a missed P0/P1 is an
architectural violation in the codebase.

---

*Last reviewed: 2026-03 — Principal Project Manager / Senior Principal DL Researcher / Principal Architect*
*Format: AGENTS.md — open standard stewarded by the Agentic AI Foundation / Linux Foundation*
*Next scheduled review: when any prohibited item is reconsidered, or at the next major milestone*
