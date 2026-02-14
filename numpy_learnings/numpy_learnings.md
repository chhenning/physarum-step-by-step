# NumPy Learnings

Personal notes and deeper explanations encountered while working through the NumPy conversion exercises.

---

## `np.add.at` — Handling Duplicate Index Collisions

**From:** Exercise 2 (Deposit)

### The Problem

When multiple particles occupy the same grid cell, you want to deposit pheromone for **each** particle. Say 3 particles are at cell `(5, 10)` — you want to add `5.0` three times, for a total of `15.0`.

### Why `trail_map[ys, xs] += 5.0` is wrong

Fancy indexing creates a **temporary copy**, does the addition, then writes back. When indices repeat, NumPy doesn't accumulate — it just writes the result of the **last** occurrence.

```python
import numpy as np

trail_map = np.zeros((10, 10))

# Three particles, all at the same cell (2, 3)
ys = np.array([2, 2, 2])
xs = np.array([3, 3, 3])

trail_map[ys, xs] += 5.0

print(trail_map[2, 3])  # 5.0  — NOT 15.0!
```

Under the hood, this expands to something like:

```python
temp = trail_map[ys, xs]   # fetches [0.0, 0.0, 0.0]
temp = temp + 5.0          # becomes [5.0, 5.0, 5.0]
trail_map[ys, xs] = temp   # writes 5.0 three times to the SAME cell
```

Each write overwrites the previous one. The cell ends up with `5.0` instead of `15.0`.

### The fix: `np.add.at`

```python
trail_map = np.zeros((10, 10))

np.add.at(trail_map, (ys, xs), 5.0)

print(trail_map[2, 3])  # 15.0  ✓
```

`np.add.at` is an **unbuffered** operation — it applies each addition immediately and sequentially, so duplicates accumulate correctly. It modifies `trail_map` in-place.

### The general pattern

`np.ufunc.at(array, indices, values)` — applies any ufunc (`add`, `multiply`, `maximum`, etc.) at the given indices without buffering. You'll mostly use `np.add.at` for this kind of scatter-accumulate pattern.

### In context of the simulation

```python
def deposit(trail_map, particles):
    # Extract all particle positions as integer arrays
    xs = particles[:, 0].astype(int) % WIDTH
    ys = particles[:, 1].astype(int) % HEIGHT

    # Every particle deposits, even if multiple share a cell
    np.add.at(trail_map, (ys, xs), DEPOSIT_AMOUNT)
```

This is the vectorized replacement for the original loop:

```python
# Old pure-Python version
for p in particles:
    x, y = int(p['x']) % WIDTH, int(p['y']) % HEIGHT
    trail_map[y][x] += DEPOSIT_AMOUNT
```

The loop naturally handles duplicates because each iteration mutates the same cell. `np.add.at` preserves that behavior in vectorized form.

---

## `px[:]` — In-Place Assignment vs. Rebinding

**From:** Exercise 3 (Move)

### The difference

```python
# Creates a NEW array, rebinds the name `px` to it
px = (px + np.cos(ph)) % WIDTH

# Writes into the EXISTING array's memory
px[:] = (px + np.cos(ph)) % WIDTH
```

### Why it matters

If another variable references the same array, the two behave differently:

```python
px = np.array([1.0, 2.0, 3.0])
ref = px  # ref points to the same array

px = px + 10       # px is now a NEW array; ref still holds [1, 2, 3]
print(ref)         # [1. 2. 3.]

px = np.array([1.0, 2.0, 3.0])
ref = px

px[:] = px + 10    # mutates the original array in-place
print(ref)         # [11. 12. 13.]  — ref sees the change
```

In the simulation, `px`, `py`, `ph` are passed into functions like `move()` and `sense()`. Using `px[:] =` ensures the caller's arrays get updated. If you used `px =` inside the function, you'd just rebind the local variable and the caller would never see the change.

### `[:]` is slice syntax

`px[:]` means "all elements" — it's a slice from start to end. NumPy interprets assignment to a slice as "write these values into the existing buffer."
