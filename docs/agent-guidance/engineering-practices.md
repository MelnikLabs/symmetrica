# Engineering Practices

# Contributor Onboarding

**Estimated time to first meaningful PR: 2–3 days.**

If you are new to this codebase, complete these steps in order before writing
any code:

1. **Read this file completely.** Not skimming — reading. The prohibited list
   in [prohibited list](./engineering-practices.md) and the autodiff doctrine in [structural autodiff doctrine](./structural-autodiff.md) will save you from the most common
   wrong turns.

2. **Build and run the test suite:**
   ```bash
   rustup show                   # confirm pinned toolchain is active
   cargo build --workspace       # confirm clean build
   cargo test --workspace        # confirm all tests pass before you touch anything
   ```

3. **Read the crate READMEs** (`algebra/`, `tensor/`, `backend/`) to
   understand what is already implemented and what is in progress.

4. **Find an open Issue tagged `good-first-issue`** before inventing new work.
   Do not add features that are not tracked in the issue tracker.

5. **Open a Draft PR early.** Don't work in isolation for more than a day.
   Draft PRs allow async feedback before significant time is invested.

**For AI agents specifically:** steps 3 and 4 require reading files in the
repository. Use the file tree, not assumptions from training data. The codebase
is WIP and diverges from any prior snapshot an agent may have seen.

---

# Unsafe Code Policy

```rust
// Required format — no exceptions
unsafe {
    // SAFETY: `ptr` is non-null by Backend::alloc postcondition
    // (see backend/src/contract.rs:L42). Alignment guaranteed to 32 bytes
    // by CpuBackend::alloc. Exclusive access proven by borrow checker.
    *ptr = value;
}
```

Rules:
1. No `unsafe` without `// SAFETY:` — enforced by `clippy::undocumented_unsafe_blocks`
2. Minimize scope to the exact operations that require it
3. Never use `unsafe` to work around a design problem — fix the design
4. Raw pointers in `tensor/` must be wrapped in safe abstractions before
   crossing module boundaries
5. `unsafe` in `algebra/` or `tensor/` requires a prior design discussion;
   `unsafe` in `backend/` is permitted only in Layer 2 (concrete impls), never in the trait definition

---

# Testing Requirements

Every new feature must include all applicable categories. CI failures block
merge without exception.

```bash
cargo test --workspace              # full suite — CI gate
cargo test -p algebra -- --nocapture
cargo test proptest                 # algebraic law tests only
cargo test grad_check               # gradient check tests only
```

**Algebraic law tests (proptest)** — every new trait:
```rust
proptest! {
    #[test]
    fn ring_distributivity(a: f64, b: f64, c: f64) {
        let lhs = R64::mul(a, R64::add(b, c));
        let rhs = R64::add(R64::mul(a, b), R64::mul(a, c));
        // Relative error only — absolute thresholds are forbidden
        let rel_err = (lhs - rhs).abs() / (rhs.abs() + f64::EPSILON);
        prop_assert!(rel_err < 1e-10, "distributivity violated: {rel_err}");
    }
}
```

**Gradient check tests** — every `Differentiable` impl:
```rust
// Relative error, boundary values (0, ±∞, subnormals, kinks), 1e-4 tolerance
// See Numerical Reliability guidance for the full pattern.
```

**Shape/rank preservation** — every tensor operation:
```rust
// Const generic typing proves rank at compile time.
// Assert runtime dimensions explicitly in tests too.
```

**Numerical stability regression** — every stability-sensitive op:
```rust
// Must not NaN/overflow on large inputs, subnormals, or boundary values.
// See Numerical Reliability guidance for the log_sum_exp pattern.
```

---

# Documentation Standards

All public items must have doc comments. Trait docs must state: (1) the
algebraic structure modeled, (2) laws implementors must uphold, (3) a usage
example.

Use `# Errors`, `# Panics`, `# Safety` sections where applicable. Math
notation requires both LaTeX (backtick blocks) and a plain-English description.

```rust
/// A commutative ring over elements of type `Self::Element`.
///
/// # Laws
///
/// - **Associativity of +**: `(a + b) + c == a + (b + c)`
/// - **Commutativity of +**: `a + b == b + a`
/// - **Distributivity**: `a * (b + c) == a*b + a*c`
///
/// For floating-point impls, laws hold approximately. Implementors must
/// document tolerance and failure conditions.
///
/// Algebraic violations are not memory UB but produce mathematically
/// meaningless results with no runtime signal.
pub trait CommutativeRing: Ring { ... }
```

---

# Rust 2026 Feature Horizon

| Feature | Status | Crate | Policy |
|---|---|---|---|
| `adt_const_params` | RFC in progress, 2026 target | `tensor/` | In use — see [nightly const-parameter policy](./rust-and-type-system.md) |
| `min_generic_const_args` | Full prototype merged | `tensor/` | In use — annotate `// MGCA:` |
| `gen` keyword | Reserved in 2024 | all | Never use as identifier |
| Polonius borrow checker | Nightly | `tensor/` | Needed for some view lifetimes |
| `std::autodiff` / Enzyme | Nightly | — | **DO NOT USE** — see below |
| New trait solver | Rolling out | `algebra/` | Fixes coherence edge cases |
| `std::offload` (GPU) | Experimental nightly | Future `backend/` GPU layer | Watch; do not adopt yet |

When a feature stabilizes: remove its `#![feature(...)]` gate, update this
table. Do not adopt features not listed here without a design discussion first.

**On `std::autodiff` / Enzyme:** nightly ships `#[autodiff_forward]` and
`#[autodiff_reverse]` backed by LLVM Enzyme. These produce correct derivatives
— but at the LLVM-IR level, invisible to the type system, bypassing the entire
algebraic abstraction this codebase is built on. Using them here is the
equivalent of using `unsafe` to solve a type-safety problem. The answer is
always no.

---

# Prohibited List

Categorically forbidden. Do not introduce. Do not rationalize exceptions.

| Prohibited | Reason |
|---|---|
| `ndarray`, `nalgebra`, `faer` | Bypass algebraic type encoding |
| Runtime autodiff tape (`Vec<Box<dyn Op>>`) | Contradicts structural autodiff |
| `std::autodiff` / Enzyme | LLVM-level, bypasses type system |
| Implicit broadcasting | Silent shape changes hide research bugs |
| Kernel fusion outside `backend/` Layer 2 | Tensor semantics must remain pure; fusion is a backend optimization (see [kernel-fusion policy](./backend-performance.md)) |
| `async` in compute paths | Parallelism is backend dispatch, not async |
| Python bindings (premature) | Freezes API before semantics are stable |
| Mixed precision in `algebra/`/`tensor/` | Precision is a backend concern |
| Optimizer state or training loops | Belong in a higher-level library |
| Hidden RNG in `algebra/`/`tensor/` | Breaks referential transparency |
| Full Hessian matrix construction | O(n²) memory; use HVPs instead |
| Numerical stop-gradient (`* 0.0`) | Must be structural (type wrapper) |
| Unexplained `#[allow(...)]` | All suppressions require justification |

---

# PR Checklist

```bash
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --check
cargo doc --workspace --no-deps
```

**Correctness**
- [ ] No crate DAG violation — `algebra/` has zero workspace deps; `tensor/` depends on `algebra/` + `backend/` (trait only, never concrete types); `backend/` depends on `algebra/`
- [ ] No new `dyn Trait` in `algebra/` or `tensor/`
- [ ] All `unsafe` blocks have `// SAFETY:` comments
- [ ] No in-place operations in `algebra/` or on autodiff-participating tensors
- [ ] No hidden RNG in `algebra/` or `tensor/`
- [ ] `stop_gradient` is structural (type wrapper), not numerical (`* 0.0`)
- [ ] No implicit broadcasting introduced
- [ ] No full Hessian matrix constructed — HVPs only
- [ ] No kernel fusion in `algebra/` or `tensor/` — fusion is backend-only ([kernel-fusion policy](./backend-performance.md))

**Numerical**
- [ ] New `Differentiable` impls have gradient check tests (relative error, 1e-4)
- [ ] Boundary values tested: `0.0`, `±∞`, subnormals, kink points
- [ ] Non-differentiable ops marked `// NON-DIFFERENTIABLE: <reason>`
- [ ] Floating-point `Field`/`Ring` impls document approximate law satisfaction
- [ ] Numerically sensitive ops have stability regression tests with `// STABILITY:` comment

**Structure**
- [ ] New algebraic traits have `proptest` tests using relative error
- [ ] New tensor ops have shape/rank preservation tests via `tensor::shape` ([shape-algebra guidance](./rust-and-type-system.md))
- [ ] Autodiff composability annotated or non-differentiability documented
- [ ] PR description states the algebraic motivation, not just the code change

**Performance**
- [ ] No compile-time regression >10% (see [compile-time budget](./backend-performance.md))
- [ ] New `backend/` kernels have a `criterion` benchmark (see [benchmarking policy](./backend-performance.md))

---

# Change Process

Not all changes are equal. Use the right process for the scope of work:

| Change type | Process |
|---|---|
| Bug fix, test addition, doc improvement | Open PR directly |
| New operation in existing trait | Open PR with algebraic motivation in description |
| New trait in `algebra/` | Open Issue first, tag `design`, get acknowledgment before coding |
| New crate or workspace restructure | Open Issue, tag `architecture`, requires explicit sign-off |
| Anything in [prohibited list](./engineering-practices.md) Prohibited List | Open Issue, tag `architecture`. Do not begin without explicit approval. |
| Nightly feature not in [Rust feature policy](./engineering-practices.md) table | Open Issue, tag `nightly-feature`, before any use |

**Draft PRs are strongly encouraged** for anything beyond a trivial fix. The
cost of early feedback is zero. The cost of a large PR going in the wrong
direction is high.

**Commit message format:**
```
<crate>: <short imperative description>

<algebraic or mathematical motivation if non-obvious>
Relates to: #<issue>
```

Examples:
```
algebra: add Norm trait for normed vector spaces

tensor: fix rank arithmetic in contract() for RANK=0 edge case
Relates to: #47

backend: gate AVX-512 kernel behind cpu-avx512 feature flag
```

---

# Known Limitations and Active Rough Edges

**Read this before hitting a wall and assuming the codebase is broken.**

These are known issues that are deferred — not bugs to fix unsolicited:

- **`adt_const_params` ICE on complex ADTs:** The nightly feature can panic
  the compiler on some nested const generic patterns. If you hit an ICE,
  file a minimal reproducer as a GitHub Issue tagged `compiler-bug` and
  work around it with a simpler encoding temporarily.

- **Trait solver coherence failures in `algebra/`:** The new Rust trait solver
  (replacing Chalk) can reject some associated-type bounds that the old solver
  accepted, and vice versa. If you get unexpected coherence errors, note the
  nightly version in the issue and tag `trait-solver`.

- **`tensor/` view lifetimes with Polonius:** Some `TensorView` lifetime
  patterns require the Polonius borrow checker (`-Z polonius`) to compile.
  These patterns are intentionally kept and are not to be worked around by
  cloning — they will compile on stable once Polonius stabilizes.

- **No benchmark baselines yet:** `backend/benches/baselines/` is empty.
  The first PR to add a kernel benchmark should also commit the initial
  baseline using `cargo criterion --save-baseline main`.

- **`algebra/` trait hierarchy is incomplete:** Several standard algebraic
  structures (Lie algebras, Hilbert spaces, manifolds) are not yet represented.
  Do not add them without a design Issue — the hierarchy interactions are
  non-trivial.

---

Symmetrica encodes mathematics in the type system.

If a change weakens type-level guarantees,
introduces runtime autodiff,
relaxes algebraic laws,
compromises numerical stability,
or makes this codebase resemble a convenience wrapper —

**the change must be rejected.**

Correctness is structural.
Differentiation is algebraic.
Boundaries are inviolable.

---

*Last reviewed: 2026-03 — Principal Project Manager / Senior Principal DL Researcher / Principal Architect*
*Format: AGENTS.md — open standard stewarded by the Agentic AI Foundation / Linux Foundation*
*Next scheduled review: when any [prohibited list](./engineering-practices.md) item is reconsidered, or at next major milestone*
