# NumPy Learning Path — From Python Grids to Arrays

You know the basics of NumPy. This guide bridges the gap to real-world usage by converting `step_07_pygame.py` from Python lists-of-lists and dicts to NumPy arrays, one operation at a time. Each exercise targets a specific function in the existing code.

**End goal:** A new step file where every grid and particle operation uses NumPy — no nested Python loops over cells or particles.

---

## What We're Replacing

Current data structures in `step_07_pygame.py`:

```python
# Trail map: list of lists (HEIGHT rows x WIDTH cols)
trail_map = [[0.0] * WIDTH for _ in range(HEIGHT)]
trail_map[y][x] += 5.0

# Particles: list of dicts
particles = [{"x": 1.5, "y": 2.3, "heading": 0.78}, ...]
for p in particles:
    p["x"] = (p["x"] + math.cos(p["heading"])) % WIDTH
```

Target data structures:

```python
# Trail map: 2D numpy array
trail_map = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
trail_map[y, x] += 5.0

# Particles: 3 separate 1D arrays (Structure of Arrays)
px = np.array([1.5, ...])   # x positions
py = np.array([2.3, ...])   # y positions
ph = np.array([0.78, ...])  # headings
```

---

## Exercise 1: The Trail Map — `np.zeros` and Scalar Operations

**Converts:** `create_trail_map()` and `decay()`

These are the easiest wins. Replace the list-of-lists with a 2D array, and replace the nested decay loop with a single operation.

### What to learn

- `np.zeros((rows, cols))` — creating arrays with a shape tuple
- Element-wise arithmetic: `trail_map *= 0.95` operates on every cell at once
- Indexing: `trail_map[y, x]` instead of `trail_map[y][x]` (both work, but the comma form is idiomatic)

### Exercise

```python
import numpy as np

# 1. Create the trail map
trail_map = np.zeros((HEIGHT, WIDTH), dtype=np.float64)

# 2. Rewrite decay() as a one-liner (no loops)
def decay(trail_map):
    trail_map *= DECAY_FACTOR
```

### Verify it works

Drop this into your existing step_07 — just change `create_trail_map()` and `decay()`. Everything else still uses Python loops. The simulation should behave identically.

### Key insight

NumPy's element-wise operations replace the double `for y / for x` loop. The `*=` operator modifies the array in place, same as the original.

---

## Exercise 2: Deposit — Integer Indexing and `np.add.at`

**Converts:** `deposit()`

This is your first encounter with a subtle NumPy gotcha: when multiple particles land on the same cell, naive indexing silently drops duplicates.

### What to learn

- Fancy (advanced) indexing: `trail_map[ys, xs]` selects multiple elements at once
- Why `trail_map[ys, xs] += 5.0` is **wrong** when indices repeat (it only adds once per unique pair)
- `np.add.at(trail_map, (ys, xs), DEPOSIT_AMOUNT)` — unbuffered addition that handles duplicates

### Exercise

```python
def deposit(px, py, trail_map):
    # Convert float positions to integer grid coordinates
    gx = px.astype(int) % WIDTH
    gy = py.astype(int) % HEIGHT

    # WRONG — if two particles share a cell, only one deposit counts:
    # trail_map[gy, gx] += DEPOSIT_AMOUNT

    # RIGHT — accumulates all deposits, even at duplicate indices:
    np.add.at(trail_map, (gy, gx), DEPOSIT_AMOUNT)
```

### Verify it works

Run with `ring` mode. Particles start on the same cells — if deposits look weaker than before, you hit the duplicate bug.

### Key insight

NumPy's fancy indexing is buffered: `a[idx] += 1` reads all values, adds 1, then writes all values back. Duplicate indices write over each other. `np.add.at` is the unbuffered alternative that accumulates correctly.

---

## Exercise 3: Move — Vectorized Trigonometry

**Converts:** `move()`

Replace the per-particle Python loop with array-wide trig.

### What to learn

- `np.cos(array)` and `np.sin(array)` — applied element-wise to entire arrays
- Modulo on arrays: `px % WIDTH` wraps all positions at once
- In-place vs. copy: `px += ...` modifies the original; `px = px + ...` creates a new array

### Exercise

```python
def move(px, py, ph):
    px[:] = (px + np.cos(ph)) % WIDTH
    py[:] = (py + np.sin(ph)) % HEIGHT
```

### Why `px[:]` instead of `px +=`?

Both work here. `px[:] = (px + np.cos(ph)) % WIDTH` makes it clear we're writing back into the same array. The `+=` form would need a separate modulo step. Pick whichever reads better to you.

### Key insight

`np.cos` on an array of 2000 headings is one call into optimized C code. The Python loop called `math.cos` 2000 times with interpreter overhead each time.

---

## Exercise 4: Spawn — Array Construction

**Converts:** `spawn_random()`, `spawn_ring()`, `spawn_center()`, `spawn_clusters()`

Replace list comprehensions that build dicts with NumPy array constructors.

### What to learn

- `np.random.uniform(low, high, size=N)` — random floats in a range
- `np.random.normal(mean, std, size=N)` — Gaussian-distributed values (replaces `random.gauss`)
- `np.linspace(0, 2*np.pi, N, endpoint=False)` — evenly spaced angles (replaces the `for i in range` loop in `spawn_ring`)
- `np.concatenate` — joining arrays (for `spawn_clusters`)

### Exercise

```python
def spawn_random():
    px = np.random.uniform(0, WIDTH, NUM_PARTICLES)
    py = np.random.uniform(0, HEIGHT, NUM_PARTICLES)
    ph = np.random.uniform(0, 2 * np.pi, NUM_PARTICLES)
    return px, py, ph

def spawn_ring():
    cx, cy = WIDTH / 2, HEIGHT / 2
    radius = min(WIDTH, HEIGHT) * 0.35
    angles = np.linspace(0, 2 * np.pi, NUM_PARTICLES, endpoint=False)
    px = cx + np.cos(angles) * radius
    py = cy + np.sin(angles) * radius
    ph = angles + np.pi  # face inward
    return px, py, ph

def spawn_center():
    cx, cy = WIDTH / 2, HEIGHT / 2
    px = np.random.normal(cx, 2, NUM_PARTICLES)
    py = np.random.normal(cy, 2, NUM_PARTICLES)
    ph = np.random.uniform(0, 2 * np.pi, NUM_PARTICLES)
    return px, py, ph

def spawn_clusters():
    half = NUM_PARTICLES // 2
    rest = NUM_PARTICLES - half
    px = np.concatenate([
        np.random.normal(WIDTH * 0.25, 2, half),
        np.random.normal(WIDTH * 0.75, 2, rest),
    ])
    py = np.random.normal(HEIGHT / 2, 2, NUM_PARTICLES)
    ph = np.random.uniform(0, 2 * np.pi, NUM_PARTICLES)
    return px, py, ph
```

### Key insight

NumPy's random module generates entire arrays at once. No list comprehensions, no loops, no dicts. The `spawn_ring` conversion is the most instructive — `np.linspace` + vectorized trig replaces an explicit loop with index arithmetic.

---

## Exercise 5: Diffuse — 2D Convolution with `scipy.ndimage`

**Converts:** `diffuse()`

This is the biggest performance win. The original is a quadruple-nested loop (y, x, dy, dx). NumPy replaces it with a single library call.

### What to learn

- `scipy.ndimage.uniform_filter(trail_map, size=3, mode='wrap')` — 3x3 mean filter with toroidal boundary
- Why `mode='wrap'` matters (matches the modulo wrapping in the original code)
- Alternative: `scipy.signal.convolve2d` with a `np.ones((3,3))/9` kernel — more explicit but same result

### Exercise

```python
from scipy.ndimage import uniform_filter

def diffuse(trail_map):
    # 3x3 mean filter with toroidal (wrap-around) boundaries
    return uniform_filter(trail_map, size=3, mode='wrap')
```

That's it. One line replaces 8 lines and a quadruple-nested loop.

### Alternative: manual with `np.roll`

If you want to understand what the convolution is actually doing:

```python
def diffuse_manual(trail_map):
    total = np.zeros_like(trail_map)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            total += np.roll(np.roll(trail_map, -dy, axis=0), -dx, axis=1)
    return total / 9.0
```

`np.roll` shifts the entire array, so `np.roll(trail_map, -1, axis=0)` shifts everything up by one row (with wrap-around). The 9 shifted copies get summed and averaged. This is still ~100x faster than the pure Python version because each `roll` and `+=` operates on the whole grid in C.

### Verify

This is where things can go subtly wrong. Compare the output of your NumPy diffuse against the original Python diffuse for a few frames to make sure they match. A quick sanity check:

```python
# After one diffuse pass on a trail_map with a single deposit at (10, 10):
# The 3x3 neighborhood around (10, 10) should all be DEPOSIT_AMOUNT / 9
```

### Key insight

Many "loop over every cell and look at neighbors" patterns are just convolutions. Recognizing this lets you replace O(H*W*9) Python operations with a single optimized call.

---

## Exercise 6: Sense — The Hardest Vectorization

**Converts:** `sense()`

This is the trickiest part. The original samples 3 trail map positions per particle, compares them, and updates headings with conditional logic. Vectorizing conditionals is the key skill here.

### What to learn

- Computing sensor positions for all particles at once (vectorized trig)
- Sampling the trail map at many positions: `trail_map[sys, sxs]` (fancy indexing for lookup)
- `np.where(condition, value_if_true, value_if_false)` — vectorized if/else
- Building complex branching logic from multiple `np.where` calls

### Exercise — build it in stages

**Stage A: Compute sensor positions (all particles, all 3 sensors)**

```python
def sense(px, py, ph, trail_map):
    # Sensor sample positions for all particles
    def sample_coords(angle_offset):
        angles = ph + angle_offset
        sx = (px + np.cos(angles) * SENSOR_DISTANCE).astype(int) % WIDTH
        sy = (py + np.sin(angles) * SENSOR_DISTANCE).astype(int) % HEIGHT
        return trail_map[sy, sx]  # fancy indexing: returns one value per particle

    val_left  = sample_coords(-SENSOR_ANGLE)
    val_front = sample_coords(0)
    val_right = sample_coords(SENSOR_ANGLE)
```

**Stage B: Vectorized decision logic**

The original logic:
```
if front >= left and front >= right: do nothing
elif left > right:  turn left
elif right > left:  turn right
else:               random turn
```

Translated to NumPy:

```python
    # Random turns for the tie-breaking case
    random_turns = np.where(
        np.random.randint(0, 2, len(px)) == 0,
        -ROTATION_ANGLE,
        ROTATION_ANGLE,
    )

    # Build the rotation amount for each particle
    front_is_best = (val_front >= val_left) & (val_front >= val_right)
    left_is_better = val_left > val_right
    right_is_better = val_right > val_left

    rotation = np.where(
        front_is_best,
        0.0,                           # front wins: no turn
        np.where(
            left_is_better,
            -ROTATION_ANGLE,            # left wins: turn left
            np.where(
                right_is_better,
                ROTATION_ANGLE,         # right wins: turn right
                random_turns,           # tie: random
            )
        )
    )

    ph[:] = ph + rotation
```

### Key insight

Vectorized branching replaces `if/elif/else` with boolean masks and `np.where`. Think of it as computing **all branches for all particles**, then selecting the right result per particle. It feels inside-out compared to scalar code, but it eliminates the particle loop entirely.

---

## Exercise 7: Render — `pygame.surfarray`

**Converts:** `draw()`

The rendering loop iterates over every cell and calls `screen.fill()` per cell. NumPy lets you build the entire pixel buffer as an array and blit it in one call.

### What to learn

- `np.clip(array, min, max)` — clamping values to a range
- Building a 3D array of shape `(WIDTH, HEIGHT, 3)` for RGB (note: pygame surfarray is transposed)
- `pygame.surfarray.blit_array(surface, array)` — push a NumPy array to the screen
- `np.kron` or `np.repeat` — upscaling by PIXEL_SCALE without a loop

### Exercise

```python
def draw(screen, trail_map):
    brightness = np.clip(trail_map / 10.0, 0.0, 1.0)

    # Build an (H, W, 3) RGB array
    r = (brightness * 140).astype(np.uint8)
    g = (brightness * 255).astype(np.uint8)
    b = (brightness * 100).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)

    # Scale up: repeat each pixel PIXEL_SCALE times in both dimensions
    scaled = np.repeat(np.repeat(rgb, PIXEL_SCALE, axis=0), PIXEL_SCALE, axis=1)

    # pygame surfarray expects (width, height, 3) — transpose the first two axes
    pygame.surfarray.blit_array(screen, scaled.transpose(1, 0, 2))
```

### Key insight

`pygame.surfarray` bridges NumPy and pygame. The transpose is necessary because NumPy arrays are `[row, col]` (y, x) but pygame surfaces are `[x, y]`. `np.repeat` handles the pixel scaling without any loops.

---

## Putting It All Together

Once all exercises are done, the main loop becomes:

```python
px, py, ph = MODES[mode]()
trail_map = np.zeros((HEIGHT, WIDTH), dtype=np.float64)

while running:
    sense(px, py, ph, trail_map)
    move(px, py, ph)
    deposit(px, py, trail_map)
    trail_map = diffuse(trail_map)
    decay(trail_map)
    draw(screen, trail_map)
```

No Python loops over particles or cells anywhere in the hot path.

---

## Integration Order (Recommended)

Convert one function at a time and verify the simulation still works after each change. This order minimizes the amount of code you need to change at once:

| Order | Function | Difficulty | Why this order |
|-------|----------|------------|----------------|
| 1 | `create_trail_map` + `decay` | Trivial | Array creation and scalar ops — the gentlest introduction |
| 2 | `deposit` | Easy | First use of fancy indexing; teaches `np.add.at` |
| 3 | `move` | Easy | Vectorized trig on the particle arrays |
| 4 | Spawn functions | Easy | Array constructors; requires switching particle storage from dicts to arrays |
| 5 | `diffuse` | Medium | Single library call, but need to understand `mode='wrap'` |
| 6 | `sense` | Hard | Vectorized conditionals with `np.where` — the conceptual leap |
| 7 | `draw` | Medium | `surfarray` + transpose convention |

After each step, run the simulation and confirm the visual output matches the original. If something looks wrong, compare intermediate values between the old and new implementations.

---

## Performance Comparison

With 2000 particles on a 320x240 grid, expect roughly:

| Version | Bottleneck | Typical frame time |
|---------|-----------|-------------------|
| Pure Python | `diffuse()` — quadruple nested loop over 76,800 cells | ~200-500ms |
| NumPy | Everything is fast; `sense()` is the relative bottleneck | ~5-15ms |

The speedup comes from replacing Python interpreter loops with calls into NumPy's compiled C/Fortran core. Each function call processes the entire array in one shot.

---

## NumPy Concepts Cheat Sheet

| Concept | Python (before) | NumPy (after) |
|---------|-----------------|---------------|
| Create grid | `[[0.0]*W for _ in range(H)]` | `np.zeros((H, W))` |
| Multiply all cells | `for y: for x: grid[y][x] *= k` | `grid *= k` |
| Trig on all particles | `for p: math.cos(p["heading"])` | `np.cos(headings)` |
| Add at indices | loop + `grid[y][x] += v` | `np.add.at(grid, (ys, xs), v)` |
| 3x3 mean filter | quadruple nested loop | `uniform_filter(grid, 3, mode='wrap')` |
| Conditional update | `if/elif/else` per particle | `np.where(cond, a, b)` |
| Sample grid at points | `grid[int(y)][int(x)]` per particle | `grid[ys, xs]` (fancy indexing) |
| Random arrays | `[random.uniform(a,b) for _ in range(N)]` | `np.random.uniform(a, b, N)` |

---

## Dependencies

Add `scipy` for the diffusion filter:

```bash
uv add scipy
```

NumPy comes as a dependency of both pygame and scipy, so no separate install needed.
