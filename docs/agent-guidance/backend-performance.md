# Backend Execution and Performance

# Kernel Fusion Policy

Kernel fusion — combining multiple tensor operations into a single backend
kernel to reduce memory bandwidth — is a valid and important optimization.
It is also a source of subtle semantic violations if applied at the wrong
abstraction layer.

## The Invariant

**Kernel fusion is permitted only in `backend/` Layer 2 (concrete impls).**

`algebra/` and `tensor/` define *what* an operation means algebraically.
`backend/` defines *how* it executes. Fusion is an execution strategy, not
an algebraic identity. Mixing them destroys the abstraction boundary that
allows backends to be swapped without changing semantics.

## What Fusion May Not Do

A fused kernel is semantically correct only if it produces bitwise-identical
results to the sequential unfused kernels, modulo floating-point reassociation
that is explicitly documented. Fusion must never:

- Change the mathematical result of a sequence of operations in a way not
  covered by a documented stability trade-off
- Alter the differentiation semantics — the pullback of a fused kernel must
  equal the composition of the individual pullbacks
- Introduce implicit broadcasting that `tensor/` does not model
- Skip operations that have side-effects on shape or layout
- Violate the deterministic reduction order required for `deterministic_mode`

## Expressing Fusion

Fusion opportunities are expressed as pattern matches on sequences of
`Backend` method calls in Layer 2. They are never expressed as new traits
in `algebra/` or new types in `tensor/`:

```rust
// CORRECT — fusion lives entirely in backend/ Layer 2
impl CpuBackend {
    // Fused multiply-add: avoids a round-trip through memory
    // STABILITY: result equivalent to sequential mul then add, same rounding
    fn fused_mul_add<T: Scalar>(
        &self, a: &Storage<T>, b: &Storage<T>, c: &Storage<T>
    ) -> Storage<T> { ... }
}

// WRONG — fusion as a new tensor type or algebra trait
trait Fusable { fn fuse_with(self, other: impl TensorOp) -> FusedOp; }
```

## Autodiff Through Fused Kernels

Every fused kernel must have a verified pullback that agrees with the
composed pullback of its constituent operations. This must be tested with
a gradient check using the same relative-error standard as [gradient-check guidance](./numerical-reliability.md).

```rust
#[test]
fn grad_check_fused_mul_add() {
    // Verify: ∂(fused_mul_add)/∂a == ∂(a*b + c)/∂a (sequential)
    // Use relative error < 1e-4; see Numerical Reliability guidance.
}
```

---

# Compile-Time Budget Policy

Generic algebra systems compound compile times aggressively. Monomorphization
is the performance strategy — but unconstrained, it makes the codebase
unmaintainable. This section formalizes the governance rules stated informally
in [compile-time type-system constraints](./rust-and-type-system.md).

## Regression Threshold

**A compile-time regression of more than 10% on the full workspace build
(`cargo build --workspace`) must be investigated and resolved before merge.**

The 10% threshold applies to:
- Clean builds (`cargo build --workspace`)
- Incremental builds after a change to `algebra/`
- Test compilation (`cargo test --workspace --no-run`)

Regressions between 5% and 10% must be noted in the PR description with
an explanation. Regressions under 5% are advisory.

## Measuring Compile Time

Use `cargo build --timings` to identify which crates and generic
instantiations dominate compile time:

```bash
cargo build --workspace --timings
# Generates target/cargo-timings/cargo-timing.html
# Inspect for codegen units that dominate wall time
```

For fine-grained monomorphization analysis:
```bash
RUSTFLAGS="-Z print-mono-items=eager" cargo +nightly build 2>&1 | grep "MONO_ITEM"
```

Before submitting any PR that adds new generic parameters or blanket impls,
run both and confirm no significant regressions.

## Rules That Govern Compile Time

These are binding rules, not guidelines:

1. **Blanket impls over unconstrained generics are prohibited.** Every blanket
   impl must have at least one bound that limits the set of types it applies to.

2. **Sealed traits are required for all core `algebra/` traits.** Unsealed
   traits allow downstream impls that the compiler cannot prune, increasing
   monomorphization surface.

3. **Nested `Dual` types beyond second order require design review.**
   `Dual<Dual<T>>` (hyper-dual) is permitted for exact second derivatives.
   `Dual<Dual<Dual<T>>>` requires an Issue tagged `compile-budget` before use.

4. **New const generic parameters require justification.** Each additional
   const generic parameter multiplies the monomorphization space. Document
   the parameter's purpose and why a runtime value is insufficient.

5. **Compile-time regression PRs must include a fix, not just a note.**
   A PR that regresses compile time by >10% will not be merged until the
   regression is resolved, regardless of functional correctness.

## Known Expensive Patterns

| Pattern | Risk | Mitigation |
|---|---|---|
| `impl<T: Field> Foo for Bar<T>` (no other bounds) | High — instantiates for every `T` | Add bounding traits; seal the trait |
| Deeply nested associated types (`A::B::C::D`) | Medium — forces solver work at every use site | Introduce intermediate type aliases |
| `where [(); RANK_A + RANK_B]:` const bounds | Medium — combinatorial on rank values | Limit to ops where rank arithmetic is genuinely needed |
| `Dual<Dual<T>>` in non-hyper-dual contexts | High — doubles monomorphization depth | Restrict to [HVP policy](./structural-autodiff.md) hyper-dual use cases only |

---

# Benchmarking Policy

`backend/` has no formal benchmark suite at the time of this writing. This
section establishes the policy that new kernel work must follow.

## Criterion Benchmarks Are Mandatory for Backend Kernels

Every non-trivial kernel added to `backend/` Layer 2 must have a corresponding
`criterion` benchmark before the PR is merged. "Non-trivial" means any kernel
that:
- Performs a BLAS call
- Uses SIMD intrinsics
- Implements a reduction (sum, max, norm, etc.)
- Is a fused operation (see [kernel-fusion policy](./backend-performance.md))

Benchmarks live in `backend/benches/`. Each benchmark file corresponds to a
kernel family (e.g., `benches/matmul.rs`, `benches/reduce.rs`).

## Benchmark Structure

Use `criterion`'s standard group structure. Always bench across multiple
input sizes — a kernel that is fast at small sizes and slow at large sizes
(or vice versa) must have both visible:

```rust
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    for size in [64usize, 256, 1024, 4096] {
        group.bench_with_input(
            BenchmarkId::new("cpu_f64", size),
            &size,
            |b, &n| {
                let a = CpuBackend::alloc_random::<f64>([n, n]);
                let b_mat = CpuBackend::alloc_random::<f64>([n, n]);
                b.iter(|| black_box(CpuBackend::matmul(&a, &b_mat)))
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
```

## Performance Claims Require Benchmark Evidence

Do not make performance claims in PR descriptions, doc comments, or
`// STABILITY:` annotations without a benchmark result to support them.
"This is faster because SIMD" is not evidence. A `criterion` output
showing throughput improvement across representative sizes is.

## Regression Tracking

Benchmark baselines are committed to `backend/benches/baselines/` using
`cargo criterion --save-baseline <name>`. PRs that modify existing kernels
must run:

```bash
cargo criterion --baseline main --bench <kernel>
```

and include the comparison output in the PR description. A regression of
more than 5% in any benchmarked operation requires justification before merge.

## What Benchmarks Do Not Prove

Benchmarks measure throughput on the benchmarking machine. They do not
verify correctness, numerical stability, or behavior on other hardware
(especially regarding SIMD availability, cache sizes, or denormal handling).
Benchmark results accompany tests — they never replace them.

---
