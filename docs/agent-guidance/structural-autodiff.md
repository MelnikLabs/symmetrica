# Structural Autodiff

# Structural Autodiff Doctrine

## Runtime Tape Is Forbidden

The following are categorically prohibited — no exceptions:

```rust
// EVERY form of this is wrong for this codebase
struct Tape { ops: Vec<Box<dyn Op>> }        // runtime graph
enum Op { Add, Mul, ... }                     // dynamic dispatch on ops
fn backward(&self, tape: &mut Tape) { ... }  // mutating graph traversal
```

Autodiff must be structural and type-driven. If the gradient of an operation
cannot be expressed as a type-level morphism, the algebraic structure is
incomplete — fix the algebra, not the grad system.

## Forward Mode — Dual Numbers as Types

Forward mode (JVP) is implemented via dual numbers encoded structurally.
The nilpotent rule ε² = 0 must hold by construction, not by convention:

```rust
struct Dual<T: Field> {
    value:   T::Element,
    tangent: T::Element,   // dvalue/dinput carried forward
}

// Dual arithmetic: (a + εb)(c + εd) = ac + ε(ad + bc)
// ε² = 0 — the second tangent term vanishes structurally
impl<T: Ring> Mul for Dual<T> {
    type Output = Dual<T>;
    fn mul(self, rhs: Self) -> Dual<T> {
        Dual {
            value:   self.value * rhs.value,
            tangent: self.value * rhs.tangent + self.tangent * rhs.value,
        }
    }
}
```

## Reverse Mode — Structural Pullbacks

Reverse mode (VJP) is more efficient when `outputs ≪ inputs` (typical DL).
The correct encoding uses a **Pullback** type — not a tape, not heap allocation:

```rust
trait Pullback<W: VectorSpace> {
    type Cotangent: VectorSpace;
    fn pullback(&self, cotangent: W) -> Self::Cotangent;
}

trait Differentiable {
    type Domain:     VectorSpace;
    type Codomain:   VectorSpace;
    type Derivative: LinearMap<Domain = Self::Domain, Codomain = Self::Codomain>;
    fn diff(&self) -> Self::Derivative;
}
```

## Higher-Order Derivatives — HVPs, Never Full Hessians

Full Hessian matrices (`n×n`) must NOT be constructed by default. They are
O(n²) in memory and almost never required. All curvature-aware algorithms
— Newton-CG, L-BFGS, influence functions, Fisher information approximations
— work exclusively with Hessian-vector products (HVPs).

HVPs use **forward-over-reverse** (JVP applied to the gradient function):

```rust
// H(f)·v = ∂[x ↦ ∇f(x)] applied in direction v
// Cost: O(gradient) — same asymptotic complexity as first-order
trait HessianVectorProduct: Differentiable {
    fn hvp(&self, at: &Self::Domain, v: &Self::Domain) -> Self::Domain;
}
```

For exact second derivatives of scalar functions, use **hyper-dual numbers**
via type-level composition — do NOT add a `num_dual` dependency:

```rust
// Dual<Dual<T>> gives exact mixed partials (Hessian diagonal) for free
// ε₁² = 0, ε₂² = 0, ε₁ε₂ ≠ 0 — the mixed term IS the second derivative
type HyperDual<T> = Dual<Dual<T>>;
```

If you find yourself building an `n×n` matrix to represent curvature: stop,
and implement the HVP directly.

## Stop-Gradient — Structural, Not Numerical

Stop-gradient boundaries MUST be type-level wrappers. Zeroing a gradient
numerically (`* 0.0`) is forbidden — the multiplication is still in the
computation graph and breaks correctness in composed systems:

```rust
// CORRECT — cotangent is structurally zero, compiler eliminates it
struct StopGrad<T>(T);

impl<T: Differentiable> Differentiable for StopGrad<T> {
    type Derivative = Zero;  // gradient does not flow through
    fn diff(&self) -> Zero { Zero }
}

// WRONG — zero numerically but structurally present; breaks composition
let loss = (pred - target * 0.0).norm();
```

Required for: target networks (RL), EMA parameters, normalization statistics,
discrete latents (VQ-VAE), any value that must not receive gradient updates.

## Non-Smooth Operations

`ReLU`, `abs`, `sign`, `max`, `min` are not differentiable at kink points.

Rules:
1. Document the subgradient convention explicitly — frameworks disagree
   (0 vs 0.5 vs 1 at the kink), and the choice affects zero-initialization
   analysis and dead neuron detection in research.
2. Mark with `// NON-DIFFERENTIABLE: <reason and convention>`.
3. Silent subgradient choices are prohibited.

```rust
impl Differentiable for Relu {
    // NON-DIFFERENTIABLE: kink at x=0, subgradient = 0 (PyTorch convention)
    fn diff(&self, x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }
}
```

## Gradient Checkpointing — Structural Recompute Boundaries

For deep differentiable chains (transformers, long sequences), naive reverse
mode stores all intermediate activations at O(depth) memory. Checkpointing
recomputes activations during the backward pass at the cost of extra compute.

In structural autodiff, checkpointing is a **recompute boundary in the morphism
composition** — not a runtime flag or boolean option:

```rust
struct Checkpoint<F: SmoothMap> {
    f:           F,
    saved_input: Option<F::Domain>, // only input stored; intermediates dropped
}
```

Do not implement checkpointing as a `bool` parameter. It changes the
mathematical contract of the morphism and must be explicit in the type.

---
