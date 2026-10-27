# Purity and Ownership

# Referential Transparency and Purity

`algebra/` and `tensor/` must be **referentially transparent** — same inputs,
same outputs, always. This is a correctness requirement for structural autodiff,
not a style preference.

## Randomness Must Be Explicit

No RNG in `algebra/` or `tensor/`. Stochastic operations take randomness as
an explicit argument:

```rust
// WRONG — hidden RNG breaks referential transparency and autodiff
fn dropout(x: Tensor<f64, 2>, rate: f64) -> Tensor<f64, 2> {
    x.map(|v| if rand::random::<f64>() > rate { v } else { 0.0 })
}

// CORRECT — mask is explicit; function is pure; derivative is well-defined
fn dropout(x: Tensor<f64, 2>, mask: Tensor<bool, 2>) -> Tensor<f64, 2> {
    x.zip_map(mask, |v, keep| if keep { v } else { 0.0 })
}
```

The derivative of a stochastic function is only well-defined when the stochastic
choices are fixed. A function that internally samples randomness cannot be
correctly differentiated through — it will silently produce wrong gradients.

`backend/` generates masks and noise and passes them as tensors. The compute
graph in `tensor/` and `algebra/` remains deterministic and pure.

## In-Place Operations

| Location | Status | Condition |
|---|---|---|
| `algebra/` | **Prohibited** | Always |
| `tensor/` | **Prohibited** | Always |
| `backend/` | Permitted | Only with proven exclusive ownership |

Autodiff-participating tensors must never be mutated in-place. The backward
pass requires forward-pass values to be intact.

---

# Ownership and Memory Model

## Owned vs Borrowed Tensors

- `Tensor<T, N>` — owned, allocates, has defined lifetime
- `TensorView<'a, T, N>` — borrowed view, zero-copy slice of owned data
- Slicing and transposing produce views, never silent copies
- Never copy data to resolve a lifetime conflict — fix the lifetime

## Gradient Accumulation vs Replacement

Be explicit about whether a gradient operation **accumulates** (`+=`) or
**replaces** (`=`). Conflating these is the source of a large class of
optimizer bugs — especially in models with shared parameters or when using
gradient checkpointing. Document this in every gradient-producing function.

---
