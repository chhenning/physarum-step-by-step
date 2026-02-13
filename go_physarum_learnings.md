# Learnings from fogleman/physarum (Go Implementation)

A deep dive into [fogleman/physarum](https://github.com/fogleman/physarum/tree/main/pkg/physarum) — a high-performance Go implementation of the physarum slime mold simulation — and what we can bring into our Python + pygame project.

## Overview

Michael Fogleman's physarum simulator runs at **1024x1024** resolution with **4 million particles** in real time. It achieves this through Go's native parallelism, flat memory layouts, lookup-table math, and power-of-2 grid tricks. Beyond raw speed, the codebase introduces several features our project doesn't yet cover: multi-species dynamics, separable blur, variable step distances, dynamic-range rendering, and color palettes.

Our Python project currently uses a 320x240 pygame grid, 2,000 particles, Python dicts for particles, nested lists for the trail map, and a brute-force 3x3 mean filter with a fixed `value / 10.0` brightness clamp. The Go version is a blueprint for where each of those decisions can evolve.

## Architecture Comparison

| Aspect | Our Python + Pygame Project | fogleman/physarum (Go) |
|---|---|---|
| Grid size | 320 x 240 | 1024 x 1024 |
| Particle count | 2,000 | ~4,000,000 |
| Particle data | `dict` with `x`, `y`, `heading` | `struct` with `X`, `Y`, `A` (angle), `C` (config index) |
| Trail map | `list[list[float]]` (2D nested) | `[]float32` (flat 1D array) |
| Diffusion | 3x3 mean filter, allocates new grid each frame | Separable box blur (H+V passes), double-buffered in-place |
| Wrapping | `% WIDTH` / `% HEIGHT` (modulo) | `& (W-1)` / `& (H-1)` (bitwise AND, power-of-2 dimensions) |
| Trig | `math.cos()` / `math.sin()` per call | Pre-computed lookup tables (65,536 entries) |
| Species | Single species | Multi-species with attraction/repulsion table |
| Rendering | Fixed `value / 10.0` brightness clamp, single color | Percentile normalization + gamma correction, multi-color palettes |
| Parallelism | Single-threaded | Goroutines across all CPU cores |

## Key Learnings

---

### 1. Multi-Species Simulation

**What the Go version does:**
Each particle carries a config index `C` (a `uint32`) that maps to a `Config` struct defining its behavior: sensor angle, sensor distance, rotation angle, step distance, deposition amount, and decay factor. Each species gets its own trail grid. An **attraction table** — an NxN matrix — controls how each species responds to every other species' trail:

```
// Attraction table (2 species example):
//          Species 0    Species 1
// Sp. 0 [  +1.05,       -0.82  ]   ← attracted to own trail, repelled by other
// Sp. 1 [  -1.12,       +0.97  ]
```

During the sense phase, each particle reads from a **combined grid** that blends all species' trails weighted by the attraction table row for that particle's species. Diagonal values (self) are positive (~1.0 with std 0.25), off-diagonal values (others) are negative (~-1.0 with std 0.25), creating self-attraction and inter-species repulsion.

**Why it matters:**
Multi-species is what produces the most visually stunning outputs — competing trail networks that carve out territories, merge at boundaries, and form dynamic equilibria. Single-species simulations always converge to the same general network topology; multi-species creates far richer emergent behavior.

**How to implement in Python:**
```python
# Each particle gets a species index
particle = {"x": ..., "y": ..., "heading": ..., "species": 0}

# Per-species config
configs = [
    {"sensor_angle": 0.78, "sensor_dist": 9.0, "rotation_angle": 0.78,
     "step_dist": 1.0, "deposit": 5.0, "decay": 0.1},
    {"sensor_angle": 0.45, "sensor_dist": 15.0, "rotation_angle": 0.45,
     "step_dist": 1.5, "deposit": 5.0, "decay": 0.1},
]

# Per-species trail grids
grids = [create_trail_map() for _ in configs]

# Attraction table
attraction = [
    [ 1.0, -0.8],   # species 0: attracted to self, repelled by species 1
    [-0.8,  1.0],    # species 1: repelled by species 0, attracted to self
]

# During sensing, build a combined grid for each species:
def combined_grid(species_index, grids, attraction):
    combined = create_trail_map()
    for j, grid in enumerate(grids):
        weight = attraction[species_index][j]
        for y in range(HEIGHT):
            for x in range(WIDTH):
                combined[y][x] += grid[y][x] * weight
    return combined
```

With numpy, the combined grid becomes a single vectorized operation: `combined = sum(a * g for a, g in zip(attraction[i], grids))`.

---

### 2. Separable Box Blur

**What the Go version does:**
Instead of our 3x3 kernel that touches 9 neighbors per cell, the Go version uses a **separable two-pass box blur**: first a horizontal pass across each row, then a vertical pass down each column. It uses a sliding window (running sum) so each pass is O(W*H) regardless of blur radius — adding the new pixel entering the window and subtracting the one leaving. The blur also supports a configurable **radius** and **iteration count**, and folds the **decay factor** into the final blur pass to save a separate traversal.

```go
// blur.go: single function that does H-pass then V-pass
func boxBlur(src, tmp []float32, w, h, r int, scale float32) {
    boxBlurH(src, tmp, w, h, r, 1)
    boxBlurV(tmp, src, w, h, r, scale)  // scale = decayFactor on last iteration
}
```

The key insight: the blur operates on **flat 1D arrays** (`[]float32`), not nested lists. The horizontal pass iterates with stride 1, the vertical pass with stride W.

**Why it matters:**
- Our 3x3 kernel is O(9 * W * H) per frame. A separable blur with radius R is O(W * H) per pass regardless of R — the sliding window makes the cost independent of radius.
- Configurable radius lets us control diffusion spread. Radius 1 ≈ our current 3x3, but radius 5 or 10 creates smoother, wider trails that dramatically change the visual character.
- Multiple iterations of box blur approximate a Gaussian blur (3 iterations is a good approximation), giving smoother diffusion without kernel computation.
- Folding decay into the blur saves iterating over the entire grid a second time.

**How to implement in Python:**

Pure Python version (significant speedup over our 3x3 even without numpy):
```python
def box_blur_h(src, dst, w, h, r):
    """Horizontal box blur pass on flat arrays."""
    d = 2 * r + 1
    for i in range(h):
        row_start = i * w
        running_sum = sum(src[row_start + (j % w)] for j in range(-r, r + 1))
        for j in range(w):
            dst[row_start + j] = running_sum / d
            running_sum += src[row_start + ((j + r + 1) % w)]
            running_sum -= src[row_start + ((j - r) % w)]
```

With numpy, even simpler: `scipy.ndimage.uniform_filter(grid, size=2*r+1, mode='wrap')` or a manual cumsum approach.

---

### 3. Trig Lookup Tables

**What the Go version does:**
Pre-computes 65,536 entries each for sin and cos at startup. The lookup function converts a radian angle to a table index using bitwise AND for wrapping:

```go
const trigTableSize = 65536    // must be power of 2
const trigTableMask = trigTableSize - 1
const trigFactor    = trigTableSize / (2 * math.Pi)

func sin(t float32) float32 {
    i := int(t*trigFactor + trigTableSize) & trigTableMask
    return sinTable[i]
}
```

Adding `trigTableSize` before masking handles negative angles without branching.

**Why it matters:**
In the hot loop, every particle calls sin/cos twice per frame (once for sensing, once for moving). With 4M particles that's 16M trig calls per frame. The lookup table replaces transcendental math with an integer multiply, add, bitwise AND, and array index — roughly 10x faster per call.

In Python, `math.sin`/`math.cos` are already C calls so the per-call overhead is smaller, but with numpy vectorization we can pre-compute trig for all particles in one shot, which is the Pythonic equivalent of this optimization.

**How to implement in Python:**

For a pure-Python loop approach:
```python
import math

TRIG_TABLE_SIZE = 65536
TRIG_TABLE_MASK = TRIG_TABLE_SIZE - 1
TRIG_FACTOR = TRIG_TABLE_SIZE / (2 * math.pi)

sin_table = [math.sin(i / TRIG_TABLE_SIZE * 2 * math.pi) for i in range(TRIG_TABLE_SIZE)]
cos_table = [math.cos(i / TRIG_TABLE_SIZE * 2 * math.pi) for i in range(TRIG_TABLE_SIZE)]

def fast_sin(t):
    return sin_table[int(t * TRIG_FACTOR + TRIG_TABLE_SIZE) & TRIG_TABLE_MASK]

def fast_cos(t):
    return cos_table[int(t * TRIG_FACTOR + TRIG_TABLE_SIZE) & TRIG_TABLE_MASK]
```

With numpy, this becomes unnecessary — `np.sin(headings)` and `np.cos(headings)` on the full particle array is already vectorized and faster than any Python-level lookup table.

---

### 4. Power-of-2 Grid Dimensions

**What the Go version does:**
Enforces that grid width and height are powers of 2 (e.g. 1024). This allows toroidal wrapping with bitwise AND instead of modulo:

```go
func (g *Grid) Index(x, y float32) int {
    i := int(x + float32(g.W)) & (g.W - 1)   // instead of: int(x) % g.W
    j := int(y + float32(g.H)) & (g.H - 1)
    return j*g.W + i
}
```

Adding `g.W` / `g.H` before masking handles negative values (similar trick to the trig table). `IsPowerOfTwo` is checked at grid creation and the program fatally exits if violated.

**Why it matters:**
Modulo (`%`) on modern CPUs takes ~20-40 cycles. Bitwise AND takes 1 cycle. When wrapping is done for every particle sense + move + deposit (potentially millions of operations per frame), this adds up. In Python, the difference is less dramatic since the interpreter overhead dominates, but with numpy integer arrays it can still matter.

**How to implement in Python:**
```python
# Use power-of-2 dimensions
WIDTH = 512   # was 320
HEIGHT = 256  # was 240

# Wrapping with bitwise AND
def wrap(val, size):
    return int(val + size) & (size - 1)

# In numpy (vectorized):
# x_wrapped = (x_int + WIDTH) & (WIDTH - 1)
```

With pygame, power-of-2 dimensions also play nicely with `PIXEL_SCALE` values — e.g. 512x256 at scale 2 gives a 1024x512 window.

---

### 5. Step Distance Parameter

**What the Go version does:**
Each species config includes a `StepDistance` field (range 0.2 to 2.0). The move function multiplies the heading direction by this value:

```go
// In the step function (conceptual):
p.X += cos(p.A) * config.StepDistance
p.Y += sin(p.A) * config.StepDistance
```

**Why it matters:**
Our implementation hardcodes a step distance of 1.0 (one cell per frame). Varying step distance changes the character of the network:
- **Small steps** (0.2–0.5): Tight, dense networks. Particles react to local gradients more precisely.
- **Large steps** (1.5–2.0): Spread-out, sparse networks. Particles can "jump" over thin trails and form longer-range connections.

With multi-species, different step distances create asymmetric dynamics — fast species explore while slow species consolidate.

**How to implement in Python:**
```python
STEP_DISTANCE = 1.0  # make this a tunable parameter

def move(p):
    p["x"] = (p["x"] + math.cos(p["heading"]) * STEP_DISTANCE) % WIDTH
    p["y"] = (p["y"] + math.sin(p["heading"]) * STEP_DISTANCE) % HEIGHT
```

This is the simplest change — just multiply by the step distance in the move function. With multi-species, it becomes per-config.

---

### 6. Parallelism Strategies

**What the Go version does:**
Uses goroutines and `sync.WaitGroup` to parallelize across CPU cores. Two main parallelization points:

1. **Grid combination**: Each species' combined grid is computed in a separate goroutine.
2. **Particle updates**: The particle array is split into chunks (one per CPU core), each chunk processed by a goroutine.

```go
// Particle update parallelism (conceptual):
numWorkers := runtime.NumCPU()
chunkSize := len(particles) / numWorkers
for w := 0; w < numWorkers; w++ {
    go func(start, end int) {
        for i := start; i < end; i++ {
            // sense, rotate, move, deposit for particles[i]
        }
        wg.Done()
    }(w*chunkSize, (w+1)*chunkSize)
}
wg.Wait()
```

**Why it matters:**
Python's GIL prevents true thread-based parallelism for CPU-bound work. But there are viable alternatives that can achieve similar speedups.

**How to implement in Python:**

Option A — **numpy vectorization** (best for Python, no parallelism needed):
```python
import numpy as np

# All particles as structured arrays
x = np.array([p["x"] for p in particles])
y = np.array([p["y"] for p in particles])
h = np.array([p["heading"] for p in particles])

# Vectorized move — operates on ALL particles at once
x = (x + np.cos(h) * step_distance) % WIDTH
y = (y + np.sin(h) * step_distance) % HEIGHT
```

Option B — **multiprocessing** for truly parallel work:
```python
from multiprocessing import Pool

def update_chunk(particles_chunk):
    for p in particles_chunk:
        sense(p, trail_map)
        move(p)
    return particles_chunk

with Pool() as pool:
    results = pool.map(update_chunk, chunks)
```

Option C — **numba JIT** for near-C speeds on the inner loop:
```python
from numba import njit

@njit
def update_particles(x, y, h, trail_data, width, height):
    for i in range(len(x)):
        # sense + move logic in compiled code
        ...
```

Numpy vectorization is the recommended first step — it eliminates Python loop overhead entirely and is often 100x faster than pure Python loops.

---

### 7. Dynamic Range Rendering

**What the Go version does:**
Instead of a fixed brightness clamp, the Go version computes the **99.9th percentile** of trail values and uses that as the maximum for normalization. It then applies **gamma correction** via a 65,536-entry lookup table:

```go
// From image.go:
maxValues[i] = float32(stat.Quantile(0.999, stat.Empirical, temp, nil))
maxValues[i] *= 1.5

// Gamma LUT:
makeGammaLookupTable := func(gamma float32) func(float32) float32 {
    table := make([]float32, 65536)
    for i := range table {
        t := float64(i) / 65535
        table[i] = float32(math.Pow(t, float64(gamma)))
    }
    ...
}
```

Each pixel is normalized to [0, 1] using the percentile range, then gamma-corrected, then multiplied by the species color.

**Why it matters:**
Our current `trail_to_color()` uses a fixed `min(value / 10.0, 1.0)` clamp. This means:
- Low-activity frames look almost black.
- High-activity frames are fully saturated.
- The visual dynamic range is fixed and doesn't adapt to changing trail intensity.

Percentile-based normalization ensures the full color range is always used, regardless of how much trail has accumulated. Gamma correction (gamma < 1) reveals faint trail structure that would otherwise be invisible, while gamma > 1 emphasizes only the strongest trails.

**How to implement in Python (pygame):**
```python
def draw_adaptive(screen, trail_map, gamma=0.5):
    # Flatten and find 99.9th percentile
    values = [trail_map[y][x] for y in range(HEIGHT) for x in range(WIDTH)]
    values.sort()
    max_val = values[int(len(values) * 0.999)] * 1.5
    if max_val == 0:
        max_val = 1.0

    for y in range(HEIGHT):
        for x in range(WIDTH):
            t = trail_map[y][x] / max_val
            t = max(0.0, min(1.0, t))
            t = t ** gamma  # gamma correction
            r = int(t * 140)
            g = int(t * 255)
            b = int(t * 100)
            rect = (x * PIXEL_SCALE, y * PIXEL_SCALE, PIXEL_SCALE, PIXEL_SCALE)
            screen.fill((r, g, b), rect)
```

With numpy: `max_val = np.percentile(grid, 99.9) * 1.5` — one line. The gamma-corrected grid can be converted to a pygame surface via `pygame.surfarray.blit_array()` for much faster rendering.

---

### 8. Color Palettes

**What the Go version does:**
Defines 8 named palettes, each with 5 RGB colors. Each species is assigned a color from the palette. When rendering, each species' normalized trail value is multiplied by its color and the results are additively blended:

```go
// Each species contributes its color weighted by trail intensity
for i, grid := range grids {
    t := normalized(grid[index])
    t = gammaCorrect(t)
    r += float32(palette[i].R) * t
    g += float32(palette[i].G) * t
    b += float32(palette[i].B) * t
}
```

This additive blending means overlapping species create new colors at their boundaries (e.g. red + blue species create purple where they compete).

**Why it matters:**
Color makes multi-species dynamics visible — you can see which species dominates which region, where they compete at boundaries, and how territories shift over time. It's the difference between a technical demo and genuinely beautiful generative art.

**How to implement in Python (pygame):**

```python
# Define palettes
PALETTES = {
    "warm":  [(255, 100, 50), (255, 200, 50), (200, 50, 100), (255, 150, 0), (200, 100, 150)],
    "cool":  [(50, 100, 255), (50, 200, 200), (100, 50, 200), (0, 150, 255), (150, 100, 200)],
    "neon":  [(255, 0, 128), (0, 255, 128), (128, 0, 255), (255, 255, 0), (0, 255, 255)],
}

# In the pygame render loop:
for y in range(HEIGHT):
    for x in range(WIDTH):
        r, g, b = 0.0, 0.0, 0.0
        for i, grid in enumerate(grids):
            t = normalize_and_gamma(grid[y][x], max_vals[i], gamma)
            color = palette[i]
            r += color[0] * t
            g += color[1] * t
            b += color[2] * t
        pixel = (min(255, int(r)), min(255, int(g)), min(255, int(b)))
        screen.fill(pixel, (x * PIXEL_SCALE, y * PIXEL_SCALE, PIXEL_SCALE, PIXEL_SCALE))
```

With numpy + pygame: build the full RGB array with vectorized operations, then blit the entire surface in one call using `pygame.surfarray.blit_array()`.

---

### 9. Double-Buffered Grids (Buffer Reuse)

**What the Go version does:**
Each `Grid` struct contains both `Data` and `Temp` arrays, pre-allocated at creation:

```go
type Grid struct {
    W, H int
    Data []float32
    Temp []float32
}
```

The box blur alternates between `Data` and `Temp` as source/destination. No new allocations happen during the simulation loop.

**Why it matters:**
Our `diffuse()` function creates a brand-new `[[0.0] * WIDTH for _ in range(HEIGHT)]` every single frame. That's `HEIGHT` list allocations + `WIDTH * HEIGHT` float allocations per frame. With Python's garbage collector, this creates memory pressure and GC pauses. Pre-allocating two buffers and swapping between them eliminates all per-frame allocation.

**How to implement in Python:**
```python
# Pre-allocate at startup
trail_map = [[0.0] * WIDTH for _ in range(HEIGHT)]
temp_map  = [[0.0] * WIDTH for _ in range(HEIGHT)]

def diffuse(src, dst):
    """Write blurred result from src into dst (no allocation)."""
    for y in range(HEIGHT):
        for x in range(WIDTH):
            total = 0.0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    total += src[(y + dy) % HEIGHT][(x + dx) % WIDTH]
            dst[y][x] = total / 9.0

# In main loop — alternate roles:
diffuse(trail_map, temp_map)
trail_map, temp_map = temp_map, trail_map  # swap references, zero allocation
```

With numpy this is even cleaner: `np.copyto(dst, blurred_src)` or use two numpy arrays and swap.

---

### 10. Random Grid Initialization

**What the Go version does:**
When creating a new grid, every cell is initialized with `rand.Float32()`:

```go
func NewGrid(w, h int) *Grid {
    data := make([]float32, w*h)
    for i := range data {
        data[i] = rand.Float32()
    }
    return &Grid{w, h, data, temp}
}
```

**Why it matters:**
Our trail map starts at all zeros. Particles must build up trails from scratch, which means the first ~50 frames show very little structure. Starting with random noise gives particles something to "react to" from frame 1, bootstrapping the self-organization process. The noise quickly gets overwhelmed by actual particle deposits but it provides the initial symmetry-breaking that accelerates pattern formation.

**How to implement in Python:**
```python
import random

def create_trail_map_with_noise():
    return [[random.random() for _ in range(WIDTH)] for _ in range(HEIGHT)]

# Or with numpy:
# trail_map = np.random.random((HEIGHT, WIDTH))
```

---

### 11. Weighted Direction Sensing

**What the Go version does:**
In addition to the standard `direction()` function (which uses simple if/else conditionals like our implementation), the Go version includes a `weightedDirection()` function that uses **probabilistic steering**:

```go
func weightedDirection(f, l, r float32) float32 {
    // Compute weights based on differences between sensor values
    // Randomly select a direction proportional to its weight
    w := [3]float32{abs(l - r), abs(f - r), abs(f - l)}
    d := [3]float32{0, -1, 1}  // straight, left, right
    // ... weighted random selection
}
```

Instead of always turning left when left > right, the particle turns left with a *probability* proportional to how much stronger the left signal is. When signals are similar, the particle might go straight even if one side is slightly stronger.

**Why it matters:**
Simple conditional steering creates sharp, angular networks because particles always commit fully to a direction. Probabilistic steering produces smoother, more organic-looking networks because particles explore more when signals are ambiguous. It's a more physically realistic model of chemotaxis.

**How to implement in Python:**
```python
def sense_weighted(p, trail_map):
    h = p["heading"]
    x, y = p["x"], p["y"]

    def sample(angle):
        sx = int(x + math.cos(angle) * SENSOR_DISTANCE) % WIDTH
        sy = int(y + math.sin(angle) * SENSOR_DISTANCE) % HEIGHT
        return trail_map[sy][sx]

    f = sample(h)
    l = sample(h - SENSOR_ANGLE)
    r = sample(h + SENSOR_ANGLE)

    # Weights based on signal differences
    w_straight = abs(l - r)   # go straight when left ≈ right
    w_left     = abs(f - r)   # turn left when front ≈ right (left is different)
    w_right    = abs(f - l)   # turn right when front ≈ left (right is different)

    total = w_straight + w_left + w_right
    if total == 0:
        return  # all signals equal, don't turn

    # Weighted random choice
    roll = random.random() * total
    if roll < w_straight:
        pass  # keep heading
    elif roll < w_straight + w_left:
        p["heading"] -= ROTATION_ANGLE
    else:
        p["heading"] += ROTATION_ANGLE
```

---

## Suggested Step Progression

These learnings map naturally onto new steps in our incremental teaching style:

| Step | Topic | Key Learnings Applied |
|---|---|---|
| **step_08** | Numpy vectorization | Flat arrays, vectorized move/sense/deposit (#6, #9) |
| **step_09** | Separable box blur | Replace 3x3 kernel with H+V sliding window, configurable radius (#2) |
| **step_10** | Dynamic rendering | Percentile normalization + gamma correction via `pygame.surfarray` (#7) |
| **step_11** | Performance tricks | Trig LUT, power-of-2 grids, buffer reuse (#3, #4, #9) |
| **step_12** | Variable parameters | Step distance, sensor distance as tunables, parameter explorer UI (#5) |
| **step_13** | Multi-species | Per-species configs, separate grids, attraction table (#1) |
| **step_14** | Color palettes | Species-colored rendering with additive blending (#8) |
| **step_15** | Weighted sensing | Probabilistic steering as alternative to conditionals (#11) |
| **step_16** | Random grid init & polish | Noise seeding, combined optimizations, final demo (#10) |

Each step builds on the previous ones and introduces exactly one new concept, maintaining the project's pedagogical philosophy.

## Reference

Source files in [`fogleman/physarum/pkg/physarum/`](https://github.com/fogleman/physarum/tree/main/pkg/physarum):

| File | What it covers |
|---|---|
| [`model.go`](https://github.com/fogleman/physarum/blob/main/pkg/physarum/model.go) | Core simulation: Model struct, Step(), direction(), weightedDirection(), goroutine parallelism |
| [`config.go`](https://github.com/fogleman/physarum/blob/main/pkg/physarum/config.go) | Config struct, RandomConfig(), RandomAttractionTable(), parameter ranges |
| [`grid.go`](https://github.com/fogleman/physarum/blob/main/pkg/physarum/grid.go) | Grid struct with Data/Temp buffers, power-of-2 Index(), BoxBlur() orchestration |
| [`blur.go`](https://github.com/fogleman/physarum/blob/main/pkg/physarum/blur.go) | Separable box blur: boxBlurH(), boxBlurV(), sliding window implementation |
| [`trig.go`](https://github.com/fogleman/physarum/blob/main/pkg/physarum/trig.go) | Sin/cos lookup tables (65,536 entries), bitwise index wrapping |
| [`particle.go`](https://github.com/fogleman/physarum/blob/main/pkg/physarum/particle.go) | Particle struct: X, Y, A (angle), C (config/species index) |
| [`image.go`](https://github.com/fogleman/physarum/blob/main/pkg/physarum/image.go) | Rendering: percentile normalization, gamma LUT, additive color blending |
| [`palette.go`](https://github.com/fogleman/physarum/blob/main/pkg/physarum/palette.go) | 8 pre-defined 5-color palettes, RandomPalette(), ShuffledPalette() |
| [`util.go`](https://github.com/fogleman/physarum/blob/main/pkg/physarum/util.go) | Helpers: Radians(), Degrees(), IsPowerOfTwo(), HexColor(), SavePNG() |
| [`run.go`](https://github.com/fogleman/physarum/blob/main/pkg/physarum/run.go) | Entry point: 1024x1024, 4M particles, 400 iterations, frame output |
