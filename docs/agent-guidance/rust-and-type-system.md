# Rust, Types, and Shapes

# Rust Edition & Nightly Policy

Symmetrica targets **Rust 2024 edition** (stable since 1.85). Nightly features
are permitted only when explicitly justified, tracked for stabilization, and
listed in [Rust feature policy](./engineering-practices.md).

## Breaking Changes in Effect (Rust 2024)

**`unsafe extern` is mandatory.** All FFI now requires:
```rust
// CORRECT
unsafe extern "C" {
    pub safe fn cblas_ddot(n: i32, x: *const f64, incx: i32,
                           y: *const f64, incy: i32) -> f64;
    pub unsafe fn free(p: *mut core::ffi::c_void);
}
// WRONG — does not compile in edition 2024
extern "C" { fn cblas_ddot(...) -> f64; }
```

**`unsafe fn` bodies require inner `unsafe {}` blocks:**
```rust
// CORRECT — explicit scope required
unsafe fn write_raw(ptr: *mut f32, val: f32) {
    unsafe { *ptr = val; }
}
```

**`gen` is a reserved keyword.** Never use it as an identifier anywhere.
Use `generate`, `produce`, or a domain-specific name instead.

**Tail expression drop order changed.** Temporaries in tail expressions are
dropped before local variables. In `backend/` where RAII guards control kernel
dispatch, never rely on the old implicit drop order.

---

# Type System Doctrine

## Invariants Belong in Types

Before adding any runtime check, ask: *"Can the compiler reject this instead?"*

Runtime shape assertions are design failures unless truly unavoidable. If you
find yourself writing `assert!(tensor.is_contiguous())`, the type system has
failed to enforce a contract it should own.

## Tensor Rank Encoding

Rank MUST be encoded in const generics. Dynamic rank is prohibited in core
tensor types.

```rust
// CORRECT — rank is a compile-time fact; B: Backend is the abstract trait from backend/
struct Tensor<T, const RANK: usize, B: Backend> {
    data:  B::Storage<T>,
    shape: [usize; RANK],
}

// WRONG — rank is runtime state, all static guarantees lost
struct Tensor<T, B: Backend> {
    data:  B::Storage<T>,
    shape: Vec<usize>,
}
```

`Backend` here is the abstract trait defined in `backend/`. `B` is never
`CpuBackend`, `CudaBackend`, or any other concrete type. Those are injected
by callers; `tensor/` is permanently unaware of which impl is used.

## `adt_const_params` — Layout as a Type (nightly, 2026 target)

Use `feature(adt_const_params)` to make layout a type-level invariant:

```rust
#![feature(adt_const_params)]

#[derive(PartialEq, Eq)]
struct Layout { row_major: bool }

struct Matrix<T, const L: Layout> { ... }
// Contiguous and strided are now distinct types — coercion is a compile error
```

Do NOT apply to types with privacy-sensitive or unsafe interior state. The RFC
governing permitted ADTs is still being finalized.

## `min_generic_const_args` (nightly, prototype)

Use only where it eliminates a meaningful type-level workaround. Every use
must carry a `// MGCA: <reason>` comment:

```rust
trait HasRank { const RANK: usize; }
struct Dense<T: HasRank> where [(); T::RANK]: { ... }
```

## Phantom Types for Algebraic Structure

Encode algebraic laws in types — never validate them at runtime:

```rust
// Scalar knows which field it lives in
struct Scalar<F: Field> {
    value:  F::Element,
    _field: PhantomData<F>,
}

// Gradient carries its originating space — mixing spaces is a compile error
struct Gradient<V: VectorSpace> {
    components: V::Dual,
    _space:     PhantomData<V>,
}
```

## Broadcasting Policy

**Implicit broadcasting is prohibited.** NumPy/PyTorch-style silent shape
expansion is a frequent source of research bugs that are invisible in small
tests. All broadcasting must be an explicit `broadcast_to` morphism visible
in the type signature, defined in `tensor::shape` (see [shape-algebra guidance](./rust-and-type-system.md)).

## Memory Layout Is Part of the Type

Contiguous row-major, contiguous column-major, and strided views are distinct
types. Silent layout coercion is never allowed.

- `Tensor<T, N>` — owned, layout determined at construction
- `TensorView<'a, T, N>` — borrowed, layout preserved from source
- Never copy data to resolve a lifetime conflict — fix the lifetime

## No `dyn Trait` in Hot Paths

`dyn Trait` is allowed only:
- At crate boundaries for error types
- For backend plugin registration during initialization only

Never use `Box<dyn Trait>` in `algebra/` or `tensor/`. Monomorphization is
the strategy — code size is a secondary concern.

## Compile-Time Explosion Governance

Unconstrained generics compound compile times aggressively. To prevent this:
- Blanket impls over unconstrained generics are prohibited
- Core `algebra/` traits must be sealed
- Nested `Dual` types beyond second order (`Dual<Dual<T>>`) require design review
- **Compile-time regressions over 10% must be investigated and resolved before merge**
  (see [compile-time budget](./backend-performance.md) for the full compile-time budget policy)
- Excessive monomorphization (binary size growth without perf justification)
  must be addressed before merge

## Associated Type Selection Policy — Regular vs GAT

The decision rule is single and non-negotiable:

> **Use a GAT if and only if the associated type's validity depends on a
> borrow from `Self`. Use a regular associated type for everything else.**

Mixing these up in either direction is wrong. A GAT where a regular type
suffices propagates lifetime noise through the entire trait hierarchy. A
regular associated type where a GAT is needed forces the wrong choice between
cloning (destroying zero-copy) and unsoundness (lying about lifetimes).

### Regular associated types — lifetime-independent

These describe *what kind of thing* a trait works with. They carry no borrow
from `Self` and are fully owned or statically known:

```rust
trait TensorAlgebra {
    type Scalar: Field;          // the numeric type — owns nothing
    type Gradient: VectorSpace;  // the cotangent element — fully owned
    type Shape: ShapeEncoding;   // compile-time rank/dimension info
    type Backend: Backend;       // the execution interface — a type, not a borrow
}
```

Adding a lifetime to any of these four is a design error. If `Gradient` seems
to need a lifetime, the gradient is not being modelled as a value — fix the
algebra.

### GATs — lifetime tied to a borrow from `Self`

These describe *something that borrows from `Self`* and cannot outlive it.
The lifetime must be present in the associated type because the associated
type's validity is a function of the source's lifetime:

```rust
trait TensorOps {
    // View borrows from self — cannot outlive the source tensor
    type View<'a>: TensorView where Self: 'a;

    // Iterator holds a reference into self — same constraint
    type Iter<'a>: Iterator where Self: 'a;

    // A borrowed slice of the underlying storage
    type Slice<'a>: StorageSlice where Self: 'a;

    // Zero-copy structural transform (transpose, reshape) — reinterprets
    // borrowed memory without allocating; must not outlive source
    type ZeroCopyTransform<'a>: TensorView where Self: 'a;
}
```

### Differentiable projections — own by default, GAT only if checkpointing

A differentiable projection is a morphism `f: A → B` with a structural
pullback `f*: B* → A*`. In structural autodiff, pullbacks are values — they
carry whatever forward-pass data they need **by value**, not by borrow:

```rust
// CORRECT — pullback owns its data; no lifetime required
trait Differentiable {
    type Domain:     VectorSpace;
    type Codomain:   VectorSpace;
    type Derivative: LinearMap; // fully owned — regular associated type
    fn diff(&self) -> Self::Derivative;
}
```

A GAT is required only if the projection *borrows* a forward activation rather
than owning it — which arises specifically in gradient checkpointing, where the
recompute boundary holds a reference to the saved input:

```rust
// CORRECT — Checkpoint borrows saved_input; projection lifetime is real
trait CheckpointedMap {
    type Projection<'a>: DifferentiableMap where Self: 'a;
    fn project<'a>(&'a self) -> Self::Projection<'a>;
}
```

If you find yourself adding a lifetime to `Derivative` outside of a
checkpointing context: stop. The pullback should own its data. If it cannot
own its data, the forward pass is retaining too much state — restructure it.

### Decision table

| Associated type | Use | Reason |
|---|---|---|
| `Scalar` | Regular | Lifetime-independent numeric type |
| `Gradient` | Regular | Fully owned cotangent value |
| `Shape` | Regular | Compile-time encoding, no borrow |
| `Backend` | Regular | Execution interface, no borrow from tensor |
| `View<'a>` | GAT | Borrows from source tensor |
| `Iter<'a>` | GAT | Holds reference into storage |
| `Slice<'a>` | GAT | Borrowed sub-storage |
| `ZeroCopyTransform<'a>` | GAT | Reinterprets borrowed memory |
| `Derivative` (autodiff) | Regular | Pullback owns its data |
| `Projection<'a>` (checkpoint only) | GAT | Borrows saved forward activation |

---

# Shape Algebra Module (`tensor::shape`)

Shape arithmetic in a rank-polymorphic tensor library is not trivial. Implicit
or ad-hoc shape reasoning is a primary source of silent bugs. All shape
operations must live in a single, well-typed module.

## Module Purpose

`tensor::shape` is the **sole location** for compile-time shape reasoning.
It encodes the rules for how ranks and dimensions transform under each tensor
operation — contraction, broadcast, transpose, reshape — as types and const
expressions, not as runtime assertions.

Any PR that introduces shape arithmetic outside this module will be rejected.

## Shape Encoding

A shape is an array of dimension sizes indexed by rank:

```rust
// Shape is a compile-time value — RANK is a const generic
type Shape<const RANK: usize> = [usize; RANK];

// ShapeEncoding: the trait that algebra/ and tensor/ operate over
pub trait ShapeEncoding {
    const RANK: usize;
    fn dims(&self) -> Shape<{ Self::RANK }>;
}
```

## Typed Shape Operations

Each operation on tensors has a corresponding shape-level morphism. The
output shape must be computable from input shapes **at compile time** wherever
rank is statically known.

### Contraction

Contraction over `K` shared indices reduces rank:

```rust
// contract: [A × K] ⊗ [K × B] → [A × B]
// Output rank = RANK_LEFT + RANK_RIGHT - 2 * NUM_CONTRACT_AXES
pub const fn contract_rank(left: usize, right: usize, k: usize) -> usize {
    left + right - 2 * k
}
```

Contracting over axes that do not exist or have mismatched sizes must be a
**compile error**, not a runtime panic. Dimension mismatches that cannot be
checked at compile time must return `Result`, never panic.

### Broadcast

Broadcasting is an explicit, typed morphism — never implicit (see [broadcasting policy](./rust-and-type-system.md)):

```rust
// broadcast_to: Shape<M> → Shape<N> where N >= M
// The shape difference is explicit in the type signature
pub fn broadcast_to<const M: usize, const N: usize>(
    src: Shape<M>, target: Shape<N>
) -> Result<Shape<N>, BroadcastError>
where [(); N - M]: // N >= M enforced at compile time
{ ... }
```

`BroadcastError` is returned (not panicked) when runtime dimension
compatibility fails (e.g., a size-3 axis cannot broadcast to size-5).

### Transpose

Transpose is a permutation of axes. The rank is preserved; the dimension
order changes:

```rust
// transpose: Shape<N> × Permutation<N> → Shape<N>
// Permutation validity (no repeats, all indices in range) is checked at construction
pub struct Permutation<const N: usize>([usize; N]);

impl<const N: usize> Permutation<N> {
    pub fn new(perm: [usize; N]) -> Result<Self, PermutationError> {
        // validate: all values in 0..N, no repeats
    }
}

pub fn transpose_shape<const N: usize>(
    shape: Shape<N>, perm: &Permutation<N>
) -> Shape<N> { ... }
```

### Reshape

Reshape is valid only when the total number of elements is preserved:

```rust
// reshape: Shape<M> → Shape<N> where product(M dims) == product(N dims)
// Validity is a runtime check because N dims are not statically constrained
pub fn reshape<const M: usize, const N: usize>(
    src: Shape<M>, target: Shape<N>
) -> Result<Shape<N>, ReshapeError> {
    // ReshapeError if element counts differ
}
```

## Shape Arithmetic in Tests

Every tensor operation test must assert the output shape explicitly, in
addition to value correctness:

```rust
#[test]
fn contract_output_shape() {
    // [3 × 4] · [4 × 5] → [3 × 5]
    let a: Tensor<f64, 2, _> = ...;
    let b: Tensor<f64, 2, _> = ...;
    let c = contract(&a, &b, 1);
    assert_eq!(c.shape(), [3, 5]);
}
```

---
