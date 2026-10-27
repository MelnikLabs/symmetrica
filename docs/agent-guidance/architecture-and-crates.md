# Architecture and Crate Boundaries

# Project Identity

Symmetrica is a **type-driven tensor algebra** with **structural autodiff** and
**backend-independent semantics**, written in Rust.

It is **NOT**:
- A deep learning framework (not PyTorch, not JAX, not a wrapper around either)
- A numerical convenience library
- A runtime tape-based autodiff system
- A BLAS abstraction layer

It **IS**:
- The algebraic substrate that a DL framework could be built on top of
- A compile-time–verified tensor and morphism system
- A research-grade, numerically stable differentiation core occupying the layer
  between raw BLAS/LAPACK and a high-level autograd engine

All design decisions must reinforce this identity. If a change makes symmetrica
look more like PyTorch, it is the wrong change.

---

# Workspace Architecture

```
symmetrica/
├── algebra/    # Abstract algebraic traits and laws
├── tensor/     # Tensor<T, B: Backend> — parameterised over the trait, not any impl
│   └── shape/  # Compile-time shape algebra (see Rust, Types, and Shapes guidance)
└── backend/    # trait Backend (abstract interface) + concrete impls (CpuBackend, ...)
```

## Dependency Graph

```
            backend/ (trait Backend)
            /                      \
      (impls)                    (uses)
     /                                \
backend/ (CpuBackend, ...)        tensor/ (Tensor<T, B: Backend>)
```

`backend/` has two internally distinct layers (see [crate-level contracts](./architecture-and-crates.md)):
- **Layer 1** — `pub trait Backend` (abstract, no `unsafe`, no BLAS)
- **Layer 2** — `pub struct CpuBackend`, future `CudaBackend`, etc. (concrete impls)

`tensor/` depends on `backend/` for the `Backend` trait only. It must never
name a concrete impl type. Concrete backends are injected by callers at the use
site — `tensor/` is permanently unaware of which impl is active.

## Strict Dependency Rules

**Forbidden — reject any PR that introduces:**
- `algebra/` importing from `tensor/` or `backend/`
- `tensor/` naming any concrete backend type (`CpuBackend`, `CudaBackend`, etc.)
- `tensor/` importing anything from `backend/`'s Layer 2 modules
- Any cyclic dependency

**Permitted and required:**
- `backend/` importing from `algebra/`
- `tensor/` importing the `Backend` trait from `backend/`

This DAG is the entire abstraction guarantee. One violation unravels it.

---

# Project Scope and Current Status

## What is in scope now (WIP)

| Area | Status | Notes |
|---|---|---|
| `algebra/` trait hierarchy | 🔄 Active | Ring, Field, VectorSpace, Module |
| `tensor/` core types | 🔄 Active | Const-generic rank, owned + view |
| `tensor/shape` module | 🔄 Active | Compile-time shape algebra (see [shape-algebra guidance](./rust-and-type-system.md)) |
| `backend/` CPU impl | 🔄 Active | BLAS bindings, basic kernels |
| Forward mode autodiff (JVP) | 🔄 Active | Dual number encoding |
| Reverse mode autodiff (VJP) | 🔄 Active | Structural pullbacks |

## What is explicitly out of scope until further notice

These items do not have open issues and should not be started without a
formal decision recorded in the issue tracker:

- GPU backend (CUDA, Vulkan, Metal, `std::offload`)
- Python bindings (PyO3)
- Higher-level training APIs (optimizers, loss functions, data loaders)
- Model serialization / checkpoint format
- Distributed execution
- WASM backend

If you believe one of these should be started, open an Issue for discussion
first. Do not begin implementation speculatively.

## Decision log

Significant architectural decisions that are not self-evident from the code
should be recorded as GitHub Issues tagged `decision-record` and linked here.
Agents and new contributors must read open decision-record issues before
making changes to `algebra/` public traits, as in-flight decisions may affect
what is acceptable.

---

# Crate-Level Contracts

## `algebra/`

- Pure trait definitions and their algebraic laws — `Ring`, `Field`, `VectorSpace`, `Module`, `Differentiable`, etc.
- **Zero workspace dependencies** — `algebra/` is the root; nothing in this workspace imports into it
- No concrete types, no allocations, no I/O, no RNG
- Every trait must document: algebraic structure, laws, and a reference
- New traits require `proptest` property tests verifying the laws
- **Autodiff composability:** every operation trait must have either a
  `Differentiable` impl or a `// NON-DIFFERENTIABLE: <reason>` annotation
- **Forbidden:** `std::collections`, `Box`, `Vec`, `Arc`, any randomness source,
  any import from `tensor/` or `backend/`

## `tensor/`

- Defines `Tensor<T, const RANK: usize, B: Backend>` — parameterised over the
  **abstract `Backend` trait**, never over a concrete impl type
- Depends on `algebra/` and `backend/` (for the trait only)
- `B: Backend` is always a trait bound; concrete types (`CpuBackend`, etc.) are
  injected by callers and must never be named inside `tensor/`
- Shape arithmetic is defined and verified in the `tensor::shape` submodule (see [shape-algebra guidance](./rust-and-type-system.md))
- Contraction, transpose, and broadcast operations preserve algebraic structure
- Memory layout is part of the type contract ([memory-layout policy](./rust-and-type-system.md))
- `Tensor<T, N>` vs `TensorView<'a, T, N>` ownership distinction is enforced ([owned and borrowed tensor policy](./purity-and-ownership.md))

## `backend/`

`backend/` has two internally distinct layers. Keep them clean:

**Layer 1 — Abstract interface** (`backend/src/lib.rs` or `backend/src/trait.rs`):
- Defines `pub trait Backend` — the sole contract between `tensor/` and execution
- Depends on `algebra/` only at this layer
- No concrete structs, no SIMD, no BLAS calls, no `unsafe`
- Adding a method to `trait Backend` is a breaking change for all impls —
  requires a design Issue before any modification

**Layer 2 — Concrete implementations** (`backend/src/cpu/`, `backend/src/gpu/`):
- `struct CpuBackend` — current CPU/BLAS implementation
- Future: `struct CudaBackend`, `struct WasmBackend`, etc.
- The ONLY location where `unsafe`, SIMD, and BLAS bindings are permitted
- The ONLY location where kernel fusion may be applied (see [kernel-fusion policy](./backend-performance.md))
- All kernels gated behind Cargo features — never unconditionally compiled
- `alloc` implementations must document alignment (32-byte for AVX2, 64-byte for AVX-512)
- Alignment violations in SIMD are UB — tests will not catch them
- Reductions must support deterministic mode with stable parallel reduction order
- All non-trivial kernels must have a `criterion` benchmark (see [benchmarking policy](./backend-performance.md))
- GPU/WASM impls are out of scope — do not add without a tracked design Issue (see [project scope](./architecture-and-crates.md))

---

# Category-Theoretic Structure

Symmetrica's design is grounded in category theory. Making this structure
explicit helps contributors reason about what operations are valid, how
autodiff composes, and where new abstractions should live.

## The Category

Symmetrica defines an implicit category **Tens** where:

- **Objects** are vector spaces (types implementing `VectorSpace`)
- **Morphisms** are differentiable maps between vector spaces
  (types implementing `Differentiable<Domain = A, Codomain = B>`)
- **Composition** is function composition, verified associative at the type level
- **Identity** morphisms exist for every object (the identity map)

This is not a metaphor — it is the architectural invariant. Every trait and
type in `algebra/` and `tensor/` must fit this picture. If a proposed abstraction
does not correspond to an object, morphism, or functor in this category,
it does not belong in `algebra/`.

## Objects → Vector Spaces

Types implementing `VectorSpace` are the objects of **Tens**:

```rust
// VectorSpace: the objects
// Must satisfy: closure, associativity, commutativity of +,
//               scalar distributivity, identity (zero vector), inverses
pub trait VectorSpace: Module {
    type Scalar: Field;
    fn zero() -> Self;
    fn add(a: Self, b: Self) -> Self;
    fn scale(s: Self::Scalar, v: Self) -> Self;
}
```

Concrete tensors (`Tensor<T, N, B>`) are elements of vector space objects,
not objects themselves. The distinction matters: the space `ℝⁿ` is the
object; a particular vector in `ℝⁿ` is an element.

## Morphisms → Differentiable Maps

Morphisms in **Tens** are smooth maps `f: A → B` between vector spaces:

```rust
// Differentiable: the morphisms
// The derivative at a point is a linear map (element of the tangent space)
pub trait Differentiable {
    type Domain:     VectorSpace;
    type Codomain:   VectorSpace;
    // The derivative: a linear map from tangent(Domain) → tangent(Codomain)
    type Derivative: LinearMap<
        Domain   = <Self::Domain as VectorSpace>::Tangent,
        Codomain = <Self::Codomain as VectorSpace>::Tangent,
    >;
    fn apply(&self, x: Self::Domain) -> Self::Codomain;
    fn diff(&self, x: &Self::Domain) -> Self::Derivative;
}
```

Composition of morphisms must be valid in both directions: `f: A → B` and
`g: B → C` compose to `g ∘ f: A → C`. The chain rule then gives
`D(g ∘ f) = Dg · Df` — this is exactly the VJP pullback composition.

## Pullbacks → Cotangent Maps

The **cotangent** (dual) of a vector space `V` is `V* = Hom(V, ℝ)`. The
pullback (VJP) of a morphism `f: A → B` is the linear map
`f*: B* → A*` that carries cotangent vectors backward:

```rust
// Pullback: the cotangent functor applied to a morphism
// f* is contravariant: reverses the direction of arrows
pub trait Pullback<W: VectorSpace> {
    type Cotangent: VectorSpace;
    // f*(w) — apply the cotangent map to a cotangent vector w ∈ W*
    fn pullback(&self, cotangent: W) -> Self::Cotangent;
}
```

This is the categorical dual of forward-mode autodiff. Forward mode applies
the **pushforward** (tangent map, covariant); reverse mode applies the
**pullback** (cotangent map, contravariant). Both are morphisms in the
appropriate (co)tangent category.

## Functors → Structure-Preserving Maps

Operations that map one algebraic structure to another structurally are
**functors**. Examples in this codebase:

| Functor | Domain category | Codomain category | What it does |
|---|---|---|---|
| `Dual<_>` | **Ring** | **Ring** | Lifts a ring to its dual-number extension |
| `Gradient<_>` | **VectorSpace** | **VectorSpace** | Maps a space to its cotangent space |
| `Checkpoint<_>` | **Differentiable** | **Differentiable** | Adds recompute boundary without changing semantics |
| `StopGrad<_>` | **Differentiable** | **Differentiable** | Sets the derivative to zero morphism |

A functor must preserve composition: `F(g ∘ f) = F(g) ∘ F(f)`. When
implementing a new functor-like wrapper, verify this law in a `proptest` test.

## Implications for Contributors

This structure has direct practical implications:

- **New operations** must be expressible as morphisms (differentiable maps)
  or functors. If they cannot be, the algebraic structure is incomplete —
  extend it, don't bypass it.
- **Composition** is always valid when types match. If composition fails to
  typecheck, the types are wrong — fix the types, not the composition.
- **Autodiff correctness** reduces to chain-rule correctness in this category.
  A gradient bug is a morphism composition bug — reason about it categorically.
- **The prohibited list** ([prohibited list](./engineering-practices.md)) items are precisely the things that break
  this categorical structure: runtime tapes replace morphisms with dynamic
  dispatch; implicit broadcasting introduces morphisms that are not
  type-visible; full Hessians mistake a rank-2 tensor for a linear map.

---
