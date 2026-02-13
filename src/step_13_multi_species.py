"""Step 13: Multi-species simulation — independent species with per-species
configs, separate trail grids, and an attraction/repulsion table.

Builds on Step 12. Major changes:
  - Each particle has a species index
  - SPECIES_CONFIGS: per-species sensor angle, sensor distance, rotation angle,
    step distance, deposit amount, decay factor, and color
  - Per-species trail grids: each species deposits onto its own grid
  - ATTRACTION table: NxN matrix controlling how each species responds to
    every other species' trail (positive = attracted, negative = repelled)
  - During sensing, a combined grid blends all species' trails weighted by
    the attraction table row for that species
  - Basic species-colored rendering (additive blend) so you can see territories

    python src/step_13_multi_species.py [random|ring|center|clusters]
"""

import math
import sys
import time

import numpy as np
import pygame

# --- Grid dimensions ---
WIDTH = 640 * 2
HEIGHT = 480 * 2

# --- Blur parameters ---
BLUR_RADIUS = 1
BLUR_ITERATIONS = 1

# --- Display ---
PIXEL_SCALE = 1
SCREEN_WIDTH = WIDTH * PIXEL_SCALE
SCREEN_HEIGHT = HEIGHT * PIXEL_SCALE
FPS = 60

# --- Rendering parameters ---
GAMMA = 0.45
PERCENTILE = 99.9
HEADROOM = 1.5

# --- Multi-species configuration (NEW in this step) ---

NUM_PARTICLES_PER_SPECIES = 2000

SPECIES_CONFIGS = [
    {
        "sensor_angle": math.radians(45),
        "sensor_distance": 9.0,
        "rotation_angle": math.radians(45),
        "step_distance": 1.0,
        "deposit": 5.0,
        "decay": 0.99,
        "color": (255, 80, 40),  # orange-red
    },
    {
        "sensor_angle": math.radians(30),
        "sensor_distance": 12.0,
        "rotation_angle": math.radians(30),
        "step_distance": 1.2,
        "deposit": 5.0,
        "decay": 0.95,
        "color": (40, 130, 255),  # blue
    },
]

NUM_SPECIES = len(SPECIES_CONFIGS)
NUM_PARTICLES = NUM_PARTICLES_PER_SPECIES * NUM_SPECIES

# Attraction table: how each species responds to every species' trail.
# Diagonal (self) is positive → attracted to own trail.
# Off-diagonal (other) is negative → repelled by other species' trail.
ATTRACTION = np.array(
    [
        [+1.0, -0.5],
        [-0.5, +1.0],
    ]
)


# --- Spawn modes ---
# Now return (px, py, ph, species) where species is an int array.


def spawn_random():
    px = np.random.uniform(0, WIDTH, NUM_PARTICLES)
    py = np.random.uniform(0, HEIGHT, NUM_PARTICLES)
    ph = np.random.uniform(0, 2 * np.pi, NUM_PARTICLES)
    species = np.repeat(np.arange(NUM_SPECIES), NUM_PARTICLES_PER_SPECIES)
    return px, py, ph, species


def spawn_ring():
    cx, cy = WIDTH / 2, HEIGHT / 2
    radius = min(WIDTH, HEIGHT) * 0.35
    angles = np.linspace(0, 2 * np.pi, NUM_PARTICLES, endpoint=False)
    px = cx + np.cos(angles) * radius
    py = cy + np.sin(angles) * radius
    ph = angles + np.pi
    species = np.repeat(np.arange(NUM_SPECIES), NUM_PARTICLES_PER_SPECIES)
    return px, py, ph, species


def spawn_center():
    cx, cy = WIDTH / 2, HEIGHT / 2
    px = np.random.normal(cx, 2, NUM_PARTICLES)
    py = np.random.normal(cy, 2, NUM_PARTICLES)
    ph = np.random.uniform(0, 2 * np.pi, NUM_PARTICLES)
    species = np.repeat(np.arange(NUM_SPECIES), NUM_PARTICLES_PER_SPECIES)
    return px, py, ph, species


def spawn_clusters():
    """Each species gets its own cluster."""
    positions_x = np.linspace(0.25, 0.75, NUM_SPECIES) * WIDTH
    px_parts, py_parts = [], []
    for cx in positions_x:
        px_parts.append(np.random.normal(cx, 3, NUM_PARTICLES_PER_SPECIES))
        py_parts.append(np.random.normal(HEIGHT / 2, 3, NUM_PARTICLES_PER_SPECIES))
    px = np.concatenate(px_parts)
    py = np.concatenate(py_parts)
    ph = np.random.uniform(0, 2 * np.pi, NUM_PARTICLES)
    species = np.repeat(np.arange(NUM_SPECIES), NUM_PARTICLES_PER_SPECIES)
    return px, py, ph, species


MODES = {
    "random": spawn_random,
    "ring": spawn_ring,
    "center": spawn_center,
    "clusters": spawn_clusters,
}


# --- Simulation functions ---


def combined_grid(species_idx, grids):
    """Build a combined sensing grid for one species.

    Blends all species' trail grids weighted by that species' row in the
    attraction table. Positive weights attract, negative weights repel.
    """
    weights = ATTRACTION[species_idx]
    result = np.zeros_like(grids[0])
    for w, g in zip(weights, grids):
        result += w * g
    return result


def sense_species(px, py, ph, mask, trail_map, cfg):
    """Sense and rotate particles selected by mask, reading from trail_map."""
    spx = px[mask]
    spy = py[mask]
    sph = ph[mask]
    sensor_angle = cfg["sensor_angle"]
    sensor_distance = cfg["sensor_distance"]
    rotation_angle = cfg["rotation_angle"]

    def sample(angle_offset):
        angles = sph + angle_offset
        sx = (spx + np.cos(angles) * sensor_distance).astype(int) % WIDTH
        sy = (spy + np.sin(angles) * sensor_distance).astype(int) % HEIGHT
        return trail_map[sy, sx]

    val_left = sample(-sensor_angle)
    val_front = sample(0)
    val_right = sample(sensor_angle)

    random_turns = np.where(
        np.random.randint(0, 2, spx.size) == 0,
        -rotation_angle,
        rotation_angle,
    )

    front_is_best = (val_front >= val_left) & (val_front >= val_right)
    left_is_better = val_left > val_right
    right_is_better = val_right > val_left

    rotation = np.where(
        front_is_best,
        0.0,
        np.where(
            left_is_better,
            -rotation_angle,
            np.where(right_is_better, rotation_angle, random_turns),
        ),
    )

    ph[mask] = sph + rotation


def move_species(px, py, ph, mask, step_distance):
    """Move particles selected by mask."""
    sph = ph[mask]
    px[mask] = (px[mask] + np.cos(sph) * step_distance) % WIDTH
    py[mask] = (py[mask] + np.sin(sph) * step_distance) % HEIGHT


def deposit_species(px, py, mask, grid, deposit_amount):
    """Deposit onto a species-specific grid for particles selected by mask."""
    gx = px[mask].astype(int) % WIDTH
    gy = py[mask].astype(int) % HEIGHT
    np.add.at(grid, (gy, gx), deposit_amount)


def separable_box_blur(grid, radius):
    d = 2 * radius + 1
    h, w = grid.shape

    padded = np.pad(grid, ((0, 0), (radius, radius)), mode="wrap")
    cs = np.zeros((h, padded.shape[1] + 1))
    cs[:, 1:] = np.cumsum(padded, axis=1)
    blurred = (cs[:, d:] - cs[:, :w]) / d

    padded = np.pad(blurred, ((radius, radius), (0, 0)), mode="wrap")
    cs = np.zeros((padded.shape[0] + 1, w))
    cs[1:, :] = np.cumsum(padded, axis=0)
    blurred = (cs[d:, :] - cs[:h, :]) / d

    return blurred


def blur_and_decay(grid, radius, iterations, decay_factor):
    for _ in range(iterations):
        grid = separable_box_blur(grid, radius)
    grid *= decay_factor
    return grid


# --- Rendering (NEW: additive species-colored blending) ---


def draw(screen, grids):
    """Render all species' grids with additive color blending.

    Each species' trail intensity is normalized and gamma-corrected independently,
    then multiplied by that species' color. Results are summed and clamped to 255.
    """
    rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float64)

    for s, cfg in enumerate(SPECIES_CONFIGS):
        grid = grids[s]
        max_val = np.percentile(grid, PERCENTILE) * HEADROOM
        if max_val < 1e-10:
            max_val = 1.0

        normalized = np.clip(grid / max_val, 0.0, 1.0)
        corrected = normalized**GAMMA

        color = cfg["color"]
        rgb[:, :, 0] += corrected * color[0]
        rgb[:, :, 1] += corrected * color[1]
        rgb[:, :, 2] += corrected * color[2]

    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)

    scaled = np.repeat(np.repeat(rgb_u8, PIXEL_SCALE, axis=0), PIXEL_SCALE, axis=1)
    pygame.surfarray.blit_array(screen, scaled.transpose(1, 0, 2))


# --- Main ---


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "random"
    if mode not in MODES:
        print(f"Unknown mode '{mode}'. Choose from: {', '.join(MODES)}")
        sys.exit(1)

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"Physarum — {mode}")
    clock = pygame.time.Clock()

    px, py, ph, species = MODES[mode]()
    grids = [np.zeros((HEIGHT, WIDTH), dtype=np.float64) for _ in range(NUM_SPECIES)]

    # Pre-compute boolean masks for each species (avoids recomputing every frame)
    masks = [species == s for s in range(NUM_SPECIES)]

    tick = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        start = time.time()

        # Build combined sensing grids and sense per species
        for s, cfg in enumerate(SPECIES_CONFIGS):
            cg = combined_grid(s, grids)
            sense_species(px, py, ph, masks[s], cg, cfg)

        # Move and deposit per species
        for s, cfg in enumerate(SPECIES_CONFIGS):
            move_species(px, py, ph, masks[s], cfg["step_distance"])
            deposit_species(px, py, masks[s], grids[s], cfg["deposit"])

        # Blur and decay each species' grid independently
        for s, cfg in enumerate(SPECIES_CONFIGS):
            grids[s] = blur_and_decay(
                grids[s], BLUR_RADIUS, BLUR_ITERATIONS, cfg["decay"]
            )

        draw(screen, grids)
        pygame.display.flip()

        elapsed = (time.time() - start) * 1000
        pygame.display.set_caption(
            f"Physarum — {mode}  |  tick={tick}  "
            f"species={NUM_SPECIES}  particles={NUM_PARTICLES}  "
            f"{elapsed:.0f}ms"
        )

        clock.tick(FPS)
        tick += 1

    pygame.quit()


if __name__ == "__main__":
    main()
