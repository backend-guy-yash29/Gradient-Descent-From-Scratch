# Gradient Computation for Multivariate Linear Regression — From Scratch

This document explains, from first principles, how gradient descent computes
weight updates when there are **multiple input columns (features)**, what the
`@` operator does, and why the transpose (`.T`) is needed. It is written
against the shapes used in `vanilla-gradient.py`.

---

## 1. The Setup

| Symbol | Meaning                          | Shape      |
|--------|----------------------------------|------------|
| `n`    | number of samples (rows)         | scalar     |
| `k`    | number of features (columns)     | scalar     |
| `X`    | input data matrix                | `(n, k)`   |
| `w`    | weight vector                    | `(k,)`     |
| `b`    | bias (intercept)                 | scalar     |
| `y`    | true output / target             | `(n,)`     |
| `y_pred` | predicted output                | `(n,)`   |
| `error`  | `y_pred - y`                    | `(n,)`     |

`X` looks like this, with one row per sample and one column per feature:

```
        feature_1   feature_2   ...   feature_k
sample1   x11         x12        ...    x1k
sample2   x21         x22        ...    x2k
  .        .           .                 .
  .        .           .                 .
samplen   xn1         xn2        ...    xnk
```

---

## 2. Prediction: Why `X @ w` Works

For a **single sample** `i`, the linear model is:

```
y_pred_i = x_i1*w1 + x_i2*w2 + ... + x_ik*wk + b
```

This is a **dot product** of row `i` of `X` with the weight vector `w`,
plus bias.

Doing this for *every* sample at once — without writing a loop — is exactly
what matrix-vector multiplication does:

```
y_pred = X @ w + b
```

### What `@` actually is

`@` is Python's matrix multiplication operator (`__matmul__`, introduced in
PEP 465). For `X` of shape `(n, k)` and `w` of shape `(k,)`:

```
X @ w  →  shape (n,)
```

Each output element is the dot product of one row of `X` with `w`:

```
(X @ w)[i] = sum_j( X[i, j] * w[j] )
```

So `@` replaces this nested loop:

```python
y_pred = []
for i in range(n):
    s = 0
    for j in range(k):
        s += X[i][j] * w[j]
    y_pred.append(s + b)
```

with a single vectorized expression — same math, no Python loops.

---

## 3. The Loss Function

Mean Squared Error over all `n` samples:

```
L = (1/n) * sum_i( error_i^2 )     where error_i = y_pred_i - y_i
```

This is a single scalar number describing how wrong the model is, on average,
across the whole dataset.

---

## 4. Deriving the Gradient (Calculus)

We need `dL/dw_j` — how much the loss changes if we nudge **one specific
weight** `w_j`. This tells gradient descent which direction to move `w_j`.

Since `y_pred_i` depends on `w_j` only through the term `x_ij * w_j`, the
chain rule gives:

```
dL/dw_j = d/dw_j [ (1/n) * sum_i( error_i^2 ) ]
        = (1/n) * sum_i( 2 * error_i * d(error_i)/dw_j )
        = (1/n) * sum_i( 2 * error_i * x_ij )
        = (2/n) * sum_i( x_ij * error_i )
```

**Key insight:** this derivative is different for every feature `j`, because
`x_ij` changes depending on which column you're looking at. With `k`
features, you get **`k` separate gradient values** — one per weight — not
one shared number.

```
dL/dw_1 = (2/n) * sum_i( x_i1 * error_i )
dL/dw_2 = (2/n) * sum_i( x_i2 * error_i )
   ...
dL/dw_k = (2/n) * sum_i( x_ik * error_i )
```

The bias has no feature index, so it collapses to one scalar:

```
dL/db = (2/n) * sum_i( error_i )
```

---

## 5. Why We Need the Transpose

We have:
- `error` → shape `(n,)` — one value per **sample**.
- `X`     → shape `(n, k)` — rows are samples, columns are features.

To compute `sum_i(x_ij * error_i)` for every feature `j` in one shot, we need
to line up each **column** of `X` with the `error` vector and dot them
together, once per column.

`X.T` (transpose) flips the matrix from `(n, k)` to `(k, n)`. Now each
**row** of `X.T` is one feature's values across all `n` samples:

```
X.T shape (k, n):

        sample1   sample2   ...   samplen
feat_1    x11       x21      ...    xn1
feat_2    x12       x22      ...    xn2
  .        .         .               .
feat_k    x1k       x2k      ...    xnk
```

Now `X.T @ error` (a `(k, n)` matrix times a `(n,)` vector) computes, **for
every feature row simultaneously**, the dot product of that row with
`error`:

```
(X.T @ error)[j] = sum_i( X.T[j, i] * error[i] )
                 = sum_i( x_ij * error_i )
```

That is *exactly* the sum inside `dL/dw_j` above — for all `j` at once,
no loop needed.

### Putting it together

```
grad_w = (2/n) * (X.T @ error)      # shape (k,) — one gradient per weight
grad_b = (2/n) * sum(error)         # scalar
```

---

## 6. Concrete Walkthrough (k=2 features, n=4 samples)

```
X = [[x11, x12],      w = [w1, w2]
     [x21, x22],
     [x31, x32],
     [x41, x42]]
```

**Step 1 — Predict:**
```
y_pred = X @ w
       = [ x11*w1 + x12*w2,
           x21*w1 + x22*w2,
           x31*w1 + x32*w2,
           x41*w1 + x42*w2 ]      shape (4,)
```

**Step 2 — Error:**
```
error = y_pred - y = [e1, e2, e3, e4]      shape (4,)
```

**Step 3 — Transpose X:**
```
X.T = [[x11, x21, x31, x41],
       [x12, x22, x32, x42]]              shape (2, 4)
```

**Step 4 — Dot each feature row with error:**
```
X.T @ error = [ x11*e1 + x21*e2 + x31*e3 + x41*e4 ,   <- feature 1's total
                x12*e1 + x22*e2 + x32*e3 + x42*e4 ]   <- feature 2's total
            shape (2,)
```

Multiply by `2/n` and you have `grad_w = [dL/dw1, dL/dw2]` — the exact two
numbers derived analytically in Section 4.

---

## 7. Shape Map (Mental Model)

```
X        (n, k)
w        (k,)
X @ w    (n, k) @ (k,)   →  (n,)        "per-sample" predictions

X.T      (k, n)
error    (n,)
X.T @ error   (k, n) @ (n,)  →  (k,)    "per-feature" gradient sums
```

Rule of thumb:
- Multiplying **by `w`** collapses the feature axis → you get one number
  **per sample**.
- Multiplying **by `X.T`** collapses the sample axis → you get one number
  **per feature**.

This symmetry is why the transpose shows up: forward pass walks samples,
backward pass (gradient) walks features, and transpose is what flips which
axis gets summed over.

---

## 8. The Bug This Explains

In `vanilla-gradient.py`, the current line is:

```python
grad_w = 2*np.mean(self.dataset.T @ error)
```

`self.dataset.T @ error` correctly produces the `(k,)` vector
`[dL/dw_1 * n/2, ..., dL/dw_k * n/2]` (unscaled). But wrapping it in
`np.mean(...)` **averages across the k feature-gradients into a single
scalar**, so every weight is updated by the same blended value instead of
its own derivative.

The fix is to normalize by the **sample count** `n`, not by averaging over
features:

```python
n = len(self.dataset)
grad_w = (2 / n) * (self.dataset.T @ error)   # stays shape (k,)
grad_b = (2 / n) * np.sum(error)
```

This preserves one independent gradient per weight, matching Section 4's
derivation exactly.
