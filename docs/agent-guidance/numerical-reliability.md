# Numerical Reliability

# Numerical Stability — Non-Negotiable

These rules exist because numerical errors in gradient-based research are
invisible in unit tests and catastrophic in multi-day training runs.

## Floating-Point Is Not a Field

`f32` and `f64` do NOT satisfy the algebraic laws of a mathematical field:

- **Non-associativity**: `(a + b) + c ≠ a + (b + c)` in general
- **Non-exact inverses**: `a + (-a)` can produce `-0.0` or NaN in edge cases
- **Non-distributivity**: `a * (b + c) ≠ a*b + a*c` in general

When implementing `Field` or `Ring` for floating-point types, the doc comment
**must** state which laws hold approximately, to what tolerance, and under
what conditions they fail. Do not silently implement a field trait for `f32`.

## No Silent NaN

If an operation can produce NaN, either encode invalid inputs in the type
system (making them unrepresentable) or return `Result`/`Option`.

```rust
// WRONG — negative input silently gives NaN, poisons entire gradient chain
fn sqrt_field(x: f64) -> f64 { x.sqrt() }

// CORRECT — invalid domain is explicit in the return type
fn sqrt_field(x: f64) -> Option<f64> {
    if x < 0.0 { None } else { Some(x.sqrt()) }
}
```

A single NaN in a gradient poisons the entire parameter update with no
runtime indication. Every potentially NaN-producing operation must be tested
at its boundary values.

## Stability Annotations Required

Operations prone to overflow, catastrophic cancellation, or instability must
carry a `// STABILITY:` comment explaining the technique used:

```rust
// WRONG — overflows for logits > ~710, common in neural net outputs
fn log_sum_exp(xs: &[f64]) -> f64 {
    xs.iter().map(|x| x.exp()).sum::<f64>().ln()
}

// CORRECT — max-shifted, stable for all finite inputs
fn log_sum_exp(xs: &[f64]) -> f64 {
    let max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    // STABILITY: shift by max before exp to prevent overflow; log cancels shift
    max + xs.iter().map(|x| (x - max).exp()).sum::<f64>().ln()
}
```

Always prefer stable standard library primitives:
- `f64::exp_m1()` over `f64::exp() - 1.0` (catastrophic cancellation near 0)
- `f64::ln_1p()` over `(1.0 + x).ln()` (catastrophic cancellation near 0)

Required techniques where applicable: log-sum-exp shifting, Kahan summation,
Cholesky over LU for PSD matrices, log-space arithmetic for probability products.

## Gradient Checks Mandatory

Every `Differentiable` implementation must include a gradient check test:

```rust
#[test]
fn grad_check_sigmoid() {
    for x in [-2.0, -0.5, 0.0, 0.5, 2.0_f64] {
        let eps       = 1e-5;
        let analytic  = sigmoid_derivative(x);
        let numerical = (sigmoid(x + eps) - sigmoid(x - eps)) / (2.0 * eps);
        // ALWAYS relative error — absolute thresholds fail for large/small x
        let rel_err   = (analytic - numerical).abs() / (numerical.abs() + 1e-8);
        assert!(rel_err < 1e-4,
            "grad check failed at x={x}: analytic={analytic}, numerical={numerical}");
    }
}
```

- Relative error tolerance: `1e-4` (standard in ML research)
- Absolute error thresholds are forbidden — they silently pass wrong gradients
- Tests must cover: `0.0`, `±∞`, subnormal inputs, kink points

## Denormals and Backend Performance

Subnormal floats can cause 10–100× slowdowns via microcode assists on some
hardware. In `backend/`, document any kernel that may encounter denormals and
whether flush-to-zero (FTZ) mode is acceptable. FTZ is never acceptable for
`algebra/` operations that participate in autodiff — it changes the math.

---
