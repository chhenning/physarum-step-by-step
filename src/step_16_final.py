"""Step 16: Random grid initialization and final polish — the capstone demo.

Builds on Step 15 (weighted sensing). Changes:
  - Trail grids start with random noise so particles react from frame 1
  - generate_random_configs(n) creates randomized species parameters
  - generate_random_attraction(n) creates randomized attraction matrices
  - Third CLI argument selects num_species (1-4)
  - R key regenerates random configs and restarts the simulation

    python src/step_16_final.py [mode] [palette] [num_species]
    python src/step_16_final.py ring fire 2
    python src/step_16_final.py random random 4
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
PIXEL_SCALE = 2
SCREEN_WIDTH = WIDTH * PIXEL_SCALE
SCREEN_HEIGHT = HEIGHT * PIXEL_SCALE
FPS = 30

# --- Rendering parameters ---
GAMMA = 0.45
PERCENTILE = 99.0
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

NUM_PARTICLES_PER_SPECIES = 200

DEFAULT_SPECIES_CONFIGS = [
    {
        "sensor_angle": math.radians(30),
        "sensor_distance": 9.0,
        "rotation_angle": math.radians(25),
        "step_distance": 1.0,
        "deposit": 1.0,
        "decay": 0.9,
    },
    {
        "sensor_angle": math.radians(25),
        "sensor_distance": 12.0,
        "rotation_angle": math.radians(20),
        "step_distance": 1.2,
        "deposit": 1.0,
        "decay": 0.9,
    },
    {
        "sensor_angle": math.radians(20),
        "sensor_distance": 7.0,
        "rotation_angle": math.radians(15),
        "step_distance": 0.8,
        "deposit": 1.0,
        "decay": 0.9,
    },
]

DEFAULT_ATTRACTION = np.array(
    [
        [+1.0, -0.5, -0.5],
        [-0.5, +1.0, -0.5],
        [-0.5, -0.5, +1.0],
    ]
)


# --- Random configuration generation ---


def generate_random_configs(n):
    configs = []
    for _ in range(n):
        configs.append(
            {
                "sensor_angle": math.radians(np.random.uniform(15, 45)),
                "sensor_distance": np.random.uniform(5.0, 15.0),
                "rotation_angle": math.radians(np.random.uniform(10, 35)),
                "step_distance": np.random.uniform(0.6, 1.5),
                "deposit": np.random.uniform(0.8, 1.5),
                "decay": np.random.uniform(0.85, 0.95),
            }
        )
    return configs


def generate_random_attraction(n):
    matrix = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = np.random.uniform(-0.8, -0.3)
    return matrix


# --- Spawn modes ---


def spawn_random(num_particles, num_species):
    px = np.random.uniform(0, WIDTH, num_particles)
    py = np.random.uniform(0, HEIGHT, num_particles)
    ph = np.random.uniform(0, 2 * np.pi, num_particles)
    species = np.repeat(np.arange(num_species), NUM_PARTICLES_PER_SPECIES)
    return px, py, ph, species


def spawn_ring(num_particles, num_species):
    cx, cy = WIDTH / 2, HEIGHT / 2
    radius = min(WIDTH, HEIGHT) * 0.35
    angles = np.linspace(0, 2 * np.pi, num_particles, endpoint=False)
    px = cx + np.cos(angles) * radius
    py = cy + np.sin(angles) * radius
    ph = angles + np.pi
    species = np.repeat(np.arange(num_species), NUM_PARTICLES_PER_SPECIES)
    return px, py, ph, species


def spawn_center(num_particles, num_species):
    cx, cy = WIDTH / 2, HEIGHT / 2
    px = np.random.normal(cx, 2, num_particles)
    py = np.random.normal(cy, 2, num_particles)
    ph = np.random.uniform(0, 2 * np.pi, num_particles)
    species = np.repeat(np.arange(num_species), NUM_PARTICLES_PER_SPECIES)
    return px, py, ph, species


def spawn_clusters(num_particles, num_species):
    positions_x = np.linspace(0.2, 0.8, num_species) * WIDTH
    px_parts, py_parts = [], []
    for cx in positions_x:
        px_parts.append(np.random.normal(cx, 3, NUM_PARTICLES_PER_SPECIES))
        py_parts.append(np.random.normal(HEIGHT / 2, 3, NUM_PARTICLES_PER_SPECIES))
    px = np.concatenate(px_parts)
    py = np.concatenate(py_parts)
    ph = np.random.uniform(0, 2 * np.pi, num_particles)
    species = np.repeat(np.arange(num_species), NUM_PARTICLES_PER_SPECIES)
    return px, py, ph, species


MODES = {
    "random": spawn_random,
    "ring": spawn_ring,
    "center": spawn_center,
    "clusters": spawn_clusters,
}


# --- Simulation functions ---


def combined_grid(species_idx, grids, attraction):
    weights = attraction[species_idx]
    result = np.zeros_like(grids[0])
    for w, g in zip(weights, grids):
        result += w * g
    return result


def sense_species(px, py, ph, mask, trail_map, cfg):
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

    w_straight = np.abs(val_left - val_right)
    w_left = np.abs(val_front - val_right)
    w_right = np.abs(val_front - val_left)

    epsilon = 1e-10
    total = w_straight + w_left + w_right + epsilon

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


def draw(screen, grids, palette_colors, num_species):
    rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float64)

    for s in range(num_species):
        grid = grids[s]
        max_val = np.percentile(grid, PERCENTILE) * HEADROOM
        if max_val < 1e-10:
            max_val = 1.0

        normalized = np.clip(grid / max_val, 0.0, 1.0)
        corrected = normalized**GAMMA

        color = palette_colors[s % len(palette_colors)]
        rgb[:, :, 0] += corrected * color[0]
        rgb[:, :, 1] += corrected * color[1]
        rgb[:, :, 2] += corrected * color[2]

    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)

    scaled = np.repeat(np.repeat(rgb_u8, PIXEL_SCALE, axis=0), PIXEL_SCALE, axis=1)
    pygame.surfarray.blit_array(screen, scaled.transpose(1, 0, 2))


# --- Initialization helper ---


def init_simulation(mode, num_species, use_random_configs):
    num_particles = NUM_PARTICLES_PER_SPECIES * num_species

    if use_random_configs or num_species != 3:
        configs = generate_random_configs(num_species)
        attraction = generate_random_attraction(num_species)
    else:
        configs = DEFAULT_SPECIES_CONFIGS
        attraction = DEFAULT_ATTRACTION

    px, py, ph, species = MODES[mode](num_particles, num_species)
    grids = [
        np.random.random((HEIGHT, WIDTH)) for _ in range(num_species)
    ]
    masks = [species == s for s in range(num_species)]

    return px, py, ph, species, grids, masks, configs, attraction


# --- Main ---


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "random"
    if mode not in MODES:
        print(f"Unknown mode '{mode}'. Choose from: {', '.join(MODES)}")
        sys.exit(1)

    palette_arg = sys.argv[2] if len(sys.argv) > 2 else "random"
    if palette_arg == "random":
        palette_idx = np.random.randint(len(PALETTE_NAMES))
    elif palette_arg in PALETTES:
        palette_idx = PALETTE_NAMES.index(palette_arg)
    else:
        print(f"Unknown palette '{palette_arg}'. Choose from: {', '.join(PALETTES)}, random")
        sys.exit(1)

    num_species = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    num_species = max(1, min(4, num_species))

    use_random_configs = num_species != 3

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    px, py, ph, species, grids, masks, configs, attraction = init_simulation(
        mode, num_species, use_random_configs
    )

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
                elif event.key == pygame.K_r:
                    px, py, ph, species, grids, masks, configs, attraction = (
                        init_simulation(mode, num_species, True)
                    )
                    tick = 0

        start = time.time()

        for s, cfg in enumerate(configs):
            cg = combined_grid(s, grids, attraction)
            sense_species(px, py, ph, masks[s], cg, cfg)

        for s, cfg in enumerate(configs):
            move_species(px, py, ph, masks[s], cfg["step_distance"])
            deposit_species(px, py, masks[s], grids[s], cfg["deposit"])

        for s, cfg in enumerate(configs):
            grids[s] = blur_and_decay(
                grids[s], BLUR_RADIUS, BLUR_ITERATIONS, cfg["decay"]
            )

        current_palette = PALETTES[PALETTE_NAMES[palette_idx]]
        draw(screen, grids, current_palette, num_species)
        pygame.display.flip()

        elapsed = (time.time() - start) * 1000
        pygame.display.set_caption(
            f"Physarum — {mode}  |  tick={tick}  "
            f"palette={PALETTE_NAMES[palette_idx]}  "
            f"species={num_species}  "
            f"{elapsed:.0f}ms  [P=palette R=regenerate]"
        )

        clock.tick(FPS)
        tick += 1

    pygame.quit()


if __name__ == "__main__":
    main()
