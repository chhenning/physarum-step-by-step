"""Step 15: Weighted sensing — probabilistic steering for smoother, more
organic network topology.

Builds on Step 14 (color palettes). Changes:
  - sense_species() uses weighted random selection instead of hard conditionals
  - Weights are derived from differences between sensor readings:
      w_straight = |left - right|   (go straight when sides are balanced)
      w_left     = |front - right|  (turn left when front ≈ right)
      w_right    = |front - left|   (turn right when front ≈ left)
  - When signals are similar, particles explore more freely
  - When one signal dominates, particles still follow it reliably
  - Networks appear smoother and more organic compared to step 14

    python src/step_15_weighted_sensing.py [mode] [palette]
    python src/step_15_weighted_sensing.py random neon
    python src/step_15_weighted_sensing.py clusters fire
"""

import math
import sys
import time

import numpy as np
import pygame

# --- Grid dimensions ---
WIDTH = 640
HEIGHT = 480

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

# --- Color palettes ---

PALETTES = {
    "warm": [(255, 100, 50), (255, 200, 50), (200, 50, 100)],
    "cool": [(50, 100, 255), (50, 200, 200), (100, 50, 200)],
    "neon": [(255, 0, 128), (0, 255, 128), (128, 0, 255)],
    "fire": [(255, 60, 20), (255, 160, 0), (255, 240, 80)],
    "ocean": [(0, 80, 200), (0, 200, 180), (100, 220, 255)],
    "pastel": [(255, 150, 150), (150, 255, 150), (150, 150, 255)],
    "sunset": [(255, 60, 80), (255, 150, 50), (180, 80, 200)],
    "forest": [(40, 180, 60), (160, 200, 40), (80, 120, 40)],
}

PALETTE_NAMES = list(PALETTES.keys())

# --- Multi-species configuration ---

NUM_PARTICLES_PER_SPECIES = 1000

SPECIES_CONFIGS = [
    {
        "sensor_angle": math.radians(45),
        "sensor_distance": 9.0,
        "rotation_angle": math.radians(45),
        "step_distance": 1.0,
        "deposit": 5.0,
        "decay": 0.99,
    },
    {
        "sensor_angle": math.radians(30),
        "sensor_distance": 12.0,
        "rotation_angle": math.radians(30),
        "step_distance": 1.2,
        "deposit": 5.0,
        "decay": 0.95,
    },
    {
        "sensor_angle": math.radians(60),
        "sensor_distance": 7.0,
        "rotation_angle": math.radians(50),
        "step_distance": 0.8,
        "deposit": 5.0,
        "decay": 0.97,
    },
]

NUM_SPECIES = len(SPECIES_CONFIGS)
NUM_PARTICLES = NUM_PARTICLES_PER_SPECIES * NUM_SPECIES

# Attraction table: self-attraction on diagonal, repulsion off-diagonal.
ATTRACTION = np.array(
    [
        [+1.0, -0.5, -0.5],
        [-0.5, +1.0, -0.5],
        [-0.5, -0.5, +1.0],
    ]
)


# --- Spawn modes ---


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
    positions_x = np.linspace(0.2, 0.8, NUM_SPECIES) * WIDTH
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
    weights = ATTRACTION[species_idx]
    result = np.zeros_like(grids[0])
    for w, g in zip(weights, grids):
        result += w * g
    return result


def sense_species(px, py, ph, mask, trail_map, cfg):
    """Weighted probabilistic steering (NEW in this step).

    Instead of always turning toward the strongest sensor, compute weights
    from the differences between sensor values and make a weighted random
    choice.  When signals are similar the particle explores freely; when
    one signal dominates it almost certainly turns that way.
    """
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

    # Compute weights from sensor differences.
    # w_straight is large when left ≈ right (sides balanced → go straight)
    # w_left is large when front ≈ right (front and right similar → turn left)
    # w_right is large when front ≈ left (front and left similar → turn right)
    w_straight = np.abs(val_left - val_right)
    w_left = np.abs(val_front - val_right)
    w_right = np.abs(val_front - val_left)

    # Add a small epsilon to avoid division by zero when all sensors read
    # the same value (all weights would be zero).  This gives uniform
    # probability in the fully-ambiguous case.
    epsilon = 1e-10
    total = w_straight + w_left + w_right + epsilon

    # Cumulative thresholds for weighted random selection.
    threshold_straight = w_straight / total
    threshold_left = (w_straight + w_left) / total

    roll = np.random.uniform(0.0, 1.0, spx.size)

    rotation = np.where(
        roll < threshold_straight,
        0.0,
        np.where(roll < threshold_left, -rotation_angle, rotation_angle),
    )

    ph[mask] = sph + rotation


def move_species(px, py, ph, mask, step_distance):
    sph = ph[mask]
    px[mask] = (px[mask] + np.cos(sph) * step_distance) % WIDTH
    py[mask] = (py[mask] + np.sin(sph) * step_distance) % HEIGHT


def deposit_species(px, py, mask, grid, deposit_amount):
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


# --- Rendering ---


def draw(screen, grids, palette_colors):
    rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float64)

    for s in range(NUM_SPECIES):
        grid = grids[s]
        max_val = np.percentile(grid, PERCENTILE) * HEADROOM
        if max_val < 1e-10:
            max_val = 1.0

        normalized = np.clip(grid / max_val, 0.0, 1.0)
        corrected = normalized**GAMMA

        color = palette_colors[s]
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

    palette_name = sys.argv[2] if len(sys.argv) > 2 else "neon"
    if palette_name not in PALETTES:
        print(f"Unknown palette '{palette_name}'. Choose from: {', '.join(PALETTES)}")
        sys.exit(1)

    palette_idx = PALETTE_NAMES.index(palette_name)

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    px, py, ph, species = MODES[mode]()
    grids = [np.zeros((HEIGHT, WIDTH), dtype=np.float64) for _ in range(NUM_SPECIES)]
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
                elif event.key == pygame.K_p:
                    palette_idx = (palette_idx + 1) % len(PALETTE_NAMES)

        start = time.time()

        for s, cfg in enumerate(SPECIES_CONFIGS):
            cg = combined_grid(s, grids)
            sense_species(px, py, ph, masks[s], cg, cfg)

        for s, cfg in enumerate(SPECIES_CONFIGS):
            move_species(px, py, ph, masks[s], cfg["step_distance"])
            deposit_species(px, py, masks[s], grids[s], cfg["deposit"])

        for s, cfg in enumerate(SPECIES_CONFIGS):
            grids[s] = blur_and_decay(
                grids[s], BLUR_RADIUS, BLUR_ITERATIONS, cfg["decay"]
            )

        current_palette = PALETTES[PALETTE_NAMES[palette_idx]]
        draw(screen, grids, current_palette)
        pygame.display.flip()

        elapsed = (time.time() - start) * 1000
        pygame.display.set_caption(
            f"Physarum — {mode}  |  tick={tick}  "
            f"palette={PALETTE_NAMES[palette_idx]}  "
            f"species={NUM_SPECIES}  "
            f"{elapsed:.0f}ms  [P=next palette]"
        )

        clock.tick(FPS)
        tick += 1

    pygame.quit()


if __name__ == "__main__":
    main()
