# Physarum Slime Mold Simulation — Learning Path

Start simple, build up. Each step adds one concept. All steps use plain Python with text/terminal output.

---

## Step 1: A Single Particle Moving on a Grid

**Goal:** Understand the most basic unit — a particle with position and heading.

- Create a 2D grid (e.g. 40x20) represented as a list of lists
- Create one particle with `x`, `y`, and `heading` (angle in degrees/radians)
- Each tick: move the particle forward by 1 step in its heading direction
- Mark visited cells with a character (e.g. `*`)
- Print the grid to the terminal each tick

**Key concepts:** Grid representation, particle state, trigonometry for movement (`cos`, `sin`), wrapping at boundaries.

---

## Step 2: Multiple Particles with Random Headings

**Goal:** Scale from 1 to many particles.

- Spawn N particles (e.g. 50) at random positions with random headings
- Each tick: all particles move forward
- Print the grid — cells with particles show `*`, empty cells show `.`
- Observe how they spread out uniformly (no intelligence yet)

**Key concepts:** Managing a collection of agents, random initialization.

---

## Step 3: The Trail Map (Deposit + Decay)

**Goal:** Introduce the continuum layer — a grid of floating-point intensities.

- Add a second grid: `trail_map` (same dimensions, initialized to 0.0)
- Each tick, after particles move, they **deposit** a value (e.g. +5.0) at their position
- After all deposits, apply **decay** to the entire trail map: `trail_map[y][x] *= 0.95`
- For display, map intensity to characters: ` ` → `.` → `o` → `O` → `#`
- Run for many ticks and observe trails forming and fading

**Key concepts:** Separating agent layer from trail layer, deposition, exponential decay.

---

## Step 4: Diffusion (Blurring the Trail Map)

**Goal:** Trails should spread to neighboring cells, not stay as sharp dots.

- Each tick, after deposition but before decay, apply a **3x3 mean filter** to the trail map
- For each cell, set its new value to the average of itself and its 8 neighbors
- Use a temporary copy of the grid to avoid read/write conflicts
- Observe how trails become smoother, broader blobs

**Key concepts:** Convolution / mean filter, neighbor access, double buffering.

---

## Step 5: Sensors — Particles Sense the Trail Map

**Goal:** This is the core feedback loop. Particles read the trail map to decide where to go.

- Give each particle 3 sensors at fixed offsets from its heading:
  - Front-left: `heading - sensor_angle`
  - Front: `heading`
  - Front-right: `heading + sensor_angle`
- Each sensor samples the trail map at a distance `sensor_distance` from the particle
- The particle turns toward the sensor with the **highest** trail value:
  - If front is highest (or tie): keep heading
  - If front-left is highest: rotate left by `rotation_angle`
  - If front-right is highest: rotate right by `rotation_angle`
- Then move forward as before

**Suggested starting parameters:**
- `sensor_angle`: 45 degrees
- `sensor_distance`: 3 cells
- `rotation_angle`: 45 degrees

**Key concepts:** Sensory feedback, chemotaxis, the sense-rotate-move loop.

---

## Step 6: Watch Emergence Happen

**Goal:** With all pieces in place, experiment and observe.

At this point you have the full basic algorithm:

```
each tick:
  1. for each particle:
       a. sense (read trail map at 3 sensor positions)
       b. rotate (turn toward strongest signal)
       c. move (step forward)
       d. deposit (add to trail map at new position)
  2. diffuse the trail map (3x3 mean filter)
  3. decay the trail map (multiply by decay factor)
  4. display
```

Try these experiments:
- **Ring start:** Place all particles in a circle facing inward → watch network formation
- **Center blob:** All particles start in the center with random headings → watch expansion
- **Random scatter:** Random positions, random headings → watch clustering emerge
- **Two clusters:** Two groups of particles → watch them find each other
- **Vary parameters:** Change sensor angle, distance, rotation angle, decay rate, deposit amount

---

## Step 7: Parameter Exploration

**Goal:** Understand what each parameter does.

| Parameter | Low Value Effect | High Value Effect |
|---|---|---|
| `sensor_angle` | Narrow search, tight paths | Wide search, broad patterns |
| `sensor_distance` | Local sensing, dense networks | Far sensing, sparse networks |
| `rotation_angle` | Gentle turns, smooth curves | Sharp turns, jagged paths |
| `decay_factor` | Trails vanish fast, less memory | Trails persist, strong highways |
| `deposit_amount` | Weak signal, less clustering | Strong signal, tight clustering |
| `num_particles` | Sparse, fragile networks | Dense, robust networks |

---

## Step 8: Pygame Visualization

**Goal:** Move from ASCII terminal output to real-time graphical rendering with pygame.

- Replace terminal printing with a pygame window
- Create a pixel surface where each grid cell maps to a `PIXEL_SCALE × PIXEL_SCALE` block
- Map trail intensity to brightness: `min(value / 10.0, 1.0)` → green-tinted color
- Handle pygame events (quit, keyboard) in the main loop
- Run at a target framerate with `pygame.time.Clock`

**Key concepts:** Pygame surface rendering, pixel scaling, event loop, framerate control.

---

## Step 9: NumPy Vectorization

**Goal:** Replace Python loops and dicts with NumPy arrays for a massive speedup.

- Store particle data as NumPy arrays: `x`, `y`, `heading` (each of shape `(N,)`)
- Replace the nested-list trail map with a 2D NumPy array: `np.zeros((HEIGHT, WIDTH))`
- Vectorize the move step: `x = (x + np.cos(heading) * step) % WIDTH` — operates on ALL particles at once
- Vectorize deposit: use `np.add.at(trail_map, (y_int, x_int), deposit_amount)` to handle duplicate positions correctly
- Vectorize sensing: compute all three sensor positions for all particles in parallel, sample the trail map with integer indexing
- Replace the 3x3 diffusion loop with `scipy.ndimage.uniform_filter` or manual NumPy slicing
- Pre-allocate a double buffer for diffusion: two NumPy arrays swapped each frame instead of allocating a new grid

**Why this matters:** Pure Python loops over thousands of particles are the main bottleneck. NumPy pushes those loops into optimized C code, typically achieving 50–100× speedups. This is the single most impactful optimization for Python.

**Key concepts:** Vectorized operations, structured arrays, `np.add.at` for scatter deposits, eliminating Python-level loops.

---

## Step 10: Separable Box Blur

**Goal:** Replace the 3x3 mean filter with a faster, more flexible separable box blur.

- Implement a **two-pass blur**: first horizontal across each row, then vertical down each column
- Use a **sliding window** (running sum) so each pass is O(W×H) regardless of blur radius — add the pixel entering the window, subtract the one leaving
- Support a configurable **radius** parameter: radius 1 ≈ our current 3x3, but radius 5 or 10 creates smoother, wider trails that dramatically change the visual character
- Support an **iteration count**: multiple iterations of box blur approximate a Gaussian blur (3 iterations is a good approximation)
- Fold the **decay factor** into the final blur pass to save a separate traversal over the entire grid

**With NumPy:** `scipy.ndimage.uniform_filter(grid, size=2*r+1, mode='wrap')` handles this in one call, or use a manual cumsum approach for more control.

**Key concepts:** Separable filters, sliding window algorithm, configurable blur radius, combining decay with diffusion.

---

## Step 11: Dynamic Range Rendering

**Goal:** Replace the fixed brightness clamp with adaptive percentile-based normalization and gamma correction.

- Instead of `min(value / 10.0, 1.0)`, compute the **99.9th percentile** of all trail values each frame and use that as the normalization maximum
- Multiply the percentile by a headroom factor (e.g. 1.5) to avoid full saturation
- Apply **gamma correction**: `normalized_value ** gamma`
  - Gamma < 1.0 (e.g. 0.5): reveals faint trail structure that would otherwise be invisible
  - Gamma > 1.0 (e.g. 2.0): emphasizes only the strongest trails
- Use `pygame.surfarray.blit_array()` to render the entire frame as a NumPy array in one call instead of filling individual rectangles

**Why this matters:** Fixed brightness clamping means low-activity frames look almost black and high-activity frames are fully saturated. Percentile normalization ensures the full color range is always used, regardless of how much trail has accumulated. Gamma correction adds artistic control over the visual character.

**With NumPy:** `max_val = np.percentile(grid, 99.9) * 1.5` — one line for the normalization reference. The entire gamma-corrected RGB array can be built with vectorized operations.

**Key concepts:** Percentile normalization, gamma correction, `pygame.surfarray`, adaptive rendering.

---

## Step 12: Variable Step Distance

**Goal:** Make step distance a tunable parameter that changes network character.

- Add a `STEP_DISTANCE` parameter (default 1.0) to the move function:
  ```python
  x = (x + cos(heading) * STEP_DISTANCE) % WIDTH
  y = (y + sin(heading) * STEP_DISTANCE) % HEIGHT
  ```
- Experiment with different values:
  - **Small steps** (0.2–0.5): Tight, dense networks. Particles react to local gradients more precisely.
  - **Large steps** (1.5–2.0): Spread-out, sparse networks. Particles can "jump" over thin trails and form longer-range connections.
- Make sensor distance independently tunable as well
- Consider building a simple parameter explorer UI: keyboard controls to adjust parameters in real time and see the effect immediately

**Key concepts:** Step distance as a tunable, relationship between movement scale and network topology, interactive parameter exploration.

---

## Step 13: Multi-Species Simulation

**Goal:** Introduce multiple species with independent behavior and inter-species interactions.

- Give each particle a `species` index (e.g. 0 or 1)
- Define **per-species configs**: each species gets its own sensor angle, sensor distance, rotation angle, step distance, deposit amount, and decay factor
- Create **per-species trail grids**: each species deposits onto its own grid
- Define an **attraction table** — an N×N matrix controlling how each species responds to every other species' trail:
  ```
  # 2-species example:
  #            Species 0    Species 1
  # Sp. 0 [    +1.0,        -0.8   ]   ← attracted to own, repelled by other
  # Sp. 1 [    -0.8,        +1.0   ]
  ```
  Diagonal values (self) are positive (~1.0), off-diagonal values (others) are negative (~-1.0), creating self-attraction and inter-species repulsion.
- During sensing, build a **combined grid** for each species by blending all trail grids weighted by that species' row in the attraction table:
  ```python
  combined = sum(weight * grid for weight, grid in zip(attraction[species_idx], grids))
  ```
- Particles sense from the combined grid but deposit onto their own species' grid

**Why this matters:** Multi-species is what produces the most visually stunning outputs — competing trail networks that carve out territories, merge at boundaries, and form dynamic equilibria. Single-species simulations always converge to the same general topology; multi-species creates far richer emergent behavior.

**Key concepts:** Per-species configuration, separate trail grids, attraction/repulsion table, combined sensing grid.

---

## Step 14: Color Palettes and Additive Blending

**Goal:** Render multi-species simulations with color — each species gets its own color, and overlapping trails blend.

- Define color palettes — sets of RGB colors, one per species:
  ```python
  PALETTES = {
      "warm": [(255, 100, 50), (255, 200, 50), (200, 50, 100)],
      "cool": [(50, 100, 255), (50, 200, 200), (100, 50, 200)],
      "neon": [(255, 0, 128), (0, 255, 128), (128, 0, 255)],
  }
  ```
- For each pixel, compute the final color by **additively blending** all species' contributions:
  ```python
  for i, grid in enumerate(species_grids):
      t = normalize_and_gamma(grid[y][x], max_vals[i], gamma)
      r += palette[i][0] * t
      g += palette[i][1] * t
      b += palette[i][2] * t
  pixel = (min(255, int(r)), min(255, int(g)), min(255, int(b)))
  ```
- Additive blending means overlapping species create new colors at boundaries (e.g. red + blue → purple where they compete)
- Use `pygame.surfarray.blit_array()` with a NumPy RGB array for efficient rendering

**Why this matters:** Color makes multi-species dynamics visible — you can see which species dominates which region, where they compete at boundaries, and how territories shift over time. It's the difference between a technical demo and genuinely beautiful generative art.

**Key concepts:** Color palettes, additive color blending, per-species normalization, full-frame surface rendering.

---

## Step 15: Weighted Sensing

**Goal:** Replace hard conditional steering with probabilistic direction selection for smoother, more organic networks.

- Instead of always turning toward the strongest sensor, compute **weights** based on differences between sensor values:
  ```python
  w_straight = abs(left - right)    # go straight when left ≈ right
  w_left     = abs(front - right)   # turn left when front ≈ right
  w_right    = abs(front - left)    # turn right when front ≈ left
  ```
- Make a **weighted random choice** among the three directions:
  ```python
  roll = random.random() * (w_straight + w_left + w_right)
  if roll < w_straight:
      pass  # keep heading
  elif roll < w_straight + w_left:
      heading -= ROTATION_ANGLE
  else:
      heading += ROTATION_ANGLE
  ```
- When signals are similar, the particle might go straight even if one side is slightly stronger — introducing natural exploration
- When one signal dominates, the particle almost certainly turns that way — preserving trail following

**Why this matters:** Simple conditional steering creates sharp, angular networks because particles always commit fully to a direction. Probabilistic steering produces smoother, more organic-looking networks because particles explore more when signals are ambiguous. It's a more physically realistic model of chemotaxis.

**Key concepts:** Probabilistic steering, weighted random selection, stochastic vs deterministic behavior, smooth vs angular network topology.

---

## Step 16: Random Grid Initialization and Final Polish

**Goal:** Bootstrap pattern formation and combine all optimizations into a polished final demo.

- Initialize the trail map with **random noise** instead of zeros:
  ```python
  trail_map = np.random.random((HEIGHT, WIDTH))
  ```
  This gives particles something to react to from frame 1, accelerating the self-organization process. The noise quickly gets overwhelmed by actual deposits but provides the initial symmetry-breaking.
- Combine all previous features into a cohesive demo:
  - Multi-species with color palettes and additive blending
  - Separable box blur with configurable radius
  - Percentile normalization and gamma correction
  - Weighted probabilistic sensing
  - Variable step distances per species
- Add command-line options for selecting palette, number of species, and spawn mode
- Generate random species configurations (with randomized attraction tables) for endlessly varied results

**Key concepts:** Noise seeding for symmetry breaking, combining optimizations, random configuration generation, generative art.

---

## References

- [Sage Jenson — Physarum](https://cargocollective.com/sagejenson/physarum)
- Jones, J. (2010). "Characteristics of pattern formation and evolution in approximations of Physarum transport networks"
- [Sebastian Lague — Slime Simulation (YouTube)](https://www.youtube.com/watch?v=X-iSQQgOd1A) — great visual explanation
