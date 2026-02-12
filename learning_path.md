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

## Future Enhancements (after the basics work)

Once the text version is working and you understand the dynamics:

1. **Visualization** — Use `pygame`, `matplotlib` animation, or `PIL` to render as images
2. **Performance** — Use `numpy` for the trail map operations (diffusion, decay)
3. **GPU** — Port to compute shaders (GLSL) or use `cupy` for massive particle counts
4. **Contagion** — Add Sage Jenson's infection mechanic (particles carry state that spreads)
5. **Color** — Map trail intensity to color gradients
6. **Food sources** — Place attractors on the map that emit their own trails
7. **Interactive** — Click to place food or spawn particles

---

## References

- [Sage Jenson — Physarum](https://cargocollective.com/sagejenson/physarum)
- Jones, J. (2010). "Characteristics of pattern formation and evolution in approximations of Physarum transport networks"
- [Sebastian Lague — Slime Simulation (YouTube)](https://www.youtube.com/watch?v=X-iSQQgOd1A) — great visual explanation
