"""Step 18: Realtime fogleman-style simulation.

Builds on Step 16 (realtime multi-species) with Step 17 improvements:
  - Classic Jones sensing logic (fogleman's direction() function)
  - Aggressive decay: 0.1 (keeps 10%) for sharp, high-contrast trails
  - Fogleman-scale parameter ranges (scaled for 640x480)
  - Float32 throughout for consistency and performance
  - ~25,000 particles per species (100k total with 4 species)
  - BLUR_ITERATIONS = 2 matching fogleman
  - Fogleman color palette added
  - Performance: per-species slice views (no boolean masks), scipy
    uniform_filter blur, bincount deposit, tensordot species mixing,
    and the modern NumPy Generator RNG

    python src/step_18_realtime.py [mode] [palette] [num_species] [save_gif]
    python src/step_18_realtime.py random fire 4
    python src/step_18_realtime.py ring neon 2
"""

from collections import deque
import math
import sys
import time

import imageio

import numpy as np
from scipy.ndimage import uniform_filter

import pygame

rng = np.random.default_rng()


# --- Grid dimensions ---
WIDTH = 320
HEIGHT = 240

# --- Blur parameters ---
BLUR_RADIUS = 1
BLUR_ITERATIONS = 2

# --- Display ---
PIXEL_SCALE = 1
SCREEN_WIDTH = WIDTH * PIXEL_SCALE
SCREEN_HEIGHT = HEIGHT * PIXEL_SCALE
FPS = 15

# --- Rendering parameters ---
GAMMA = np.float32(0.45)
PERCENTILE = 99.0
HEADROOM = np.float32(1.5)

# --- Particles per species ---
NUM_PARTICLES_PER_SPECIES = 25_000

# --- Color palettes ---

FOGLEMAN_COLORS = [
    (0xC9, 0x3C, 0x00),  # #C93C00 — orange-red
    (0xE8, 0x88, 0x01),  # #E88801 — amber
    (0x73, 0x00, 0x46),  # #730046 — magenta-purple
    (0xBF, 0xBB, 0x11),  # #BFBB11 — yellow-green
]

PALETTES = {
    "fogleman": FOGLEMAN_COLORS,
    "warm": [(255, 100, 50), (255, 200, 50), (200, 50, 100), (255, 150, 80)],
    "cool": [(50, 100, 255), (50, 200, 200), (100, 50, 200), (80, 150, 255)],
    "neon": [(255, 0, 128), (0, 255, 128), (128, 0, 255), (255, 255, 0)],
    "fire": [(255, 60, 20), (255, 160, 0), (255, 240, 80), (200, 40, 0)],
    "ocean": [(0, 80, 200), (0, 200, 180), (100, 220, 255), (40, 120, 200)],
    "pastel": [(255, 150, 150), (150, 255, 150), (150, 150, 255), (200, 200, 150)],
    "sunset": [(255, 60, 80), (255, 150, 50), (180, 80, 200), (255, 120, 40)],
    "forest": [(40, 180, 60), (160, 200, 40), (80, 120, 40), (200, 255, 80)],
}

PALETTE_NAMES = list(PALETTES.keys())

frames = deque(maxlen=100)


# --- Random configuration generation (fogleman-scale, scaled for 640x480) ---


def generate_random_configs(n):
    """Generate species configs with fogleman-scale parameter ranges.

    Fogleman's ranges (1024x1024): SensorDistance 20-65
    Scaled for 640x480 (factor ~0.625): SensorDistance 12-40
    Angles are resolution-independent.
    """
    configs = []
    for _ in range(n):
        configs.append(
            {
                "sensor_angle": np.float32(rng.uniform(0.6, 1.4)),
                "sensor_distance": np.float32(rng.uniform(12.0, 40.0)),
                "rotation_angle": np.float32(rng.uniform(0.3, 1.4)),
                "step_distance": np.float32(rng.uniform(1.0, 1.7)),
                # "deposit": np.float32(5.0),
                "deposit": np.float32(rng.uniform(3.0, 6.0)),
                # "decay": np.float32(0.1),
                "decay": np.float32(rng.uniform(0.05, 0.2)),
            }
        )
    return configs


def generate_random_attraction(n):
    """Generate attraction matrix: positive diagonal, negative off-diagonal."""
    matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = rng.uniform(0.5, 1.3)
            else:
                matrix[i, j] = rng.uniform(-1.3, -0.4)
    return matrix


# --- Spawn modes ---


def spawn_random(num_particles, num_species):
    px = rng.uniform(0, WIDTH, num_particles).astype(np.float32)
    py = rng.uniform(0, HEIGHT, num_particles).astype(np.float32)
    ph = rng.uniform(0, 2 * np.pi, num_particles).astype(np.float32)
    return px, py, ph


def spawn_ring(num_particles, num_species):
    cx, cy = np.float32(WIDTH / 2), np.float32(HEIGHT / 2)
    radius = np.float32(min(WIDTH, HEIGHT) * 0.35)
    angles = np.linspace(0, 2 * np.pi, num_particles, endpoint=False).astype(np.float32)
    px = cx + np.cos(angles) * radius
    py = cy + np.sin(angles) * radius
    ph = angles + np.float32(np.pi)
    return px, py, ph


def spawn_center(num_particles, num_species):
    cx, cy = np.float32(WIDTH / 2), np.float32(HEIGHT / 2)
    px = rng.normal(cx, 2, num_particles).astype(np.float32)
    py = rng.normal(cy, 2, num_particles).astype(np.float32)
    ph = rng.uniform(0, 2 * np.pi, num_particles).astype(np.float32)
    return px, py, ph


def spawn_clusters(num_particles, num_species):
    positions_x = np.linspace(0.2, 0.8, num_species) * WIDTH
    px_parts, py_parts = [], []
    for cx in positions_x:
        px_parts.append(
            rng.normal(cx, 3, NUM_PARTICLES_PER_SPECIES).astype(np.float32)
        )
        py_parts.append(
            rng.normal(HEIGHT / 2, 3, NUM_PARTICLES_PER_SPECIES).astype(
                np.float32
            )
        )
    px = np.concatenate(px_parts)
    py = np.concatenate(py_parts)
    ph = rng.uniform(0, 2 * np.pi, num_particles).astype(np.float32)
    return px, py, ph


MODES = {
    "random": spawn_random,
    "ring": spawn_ring,
    "center": spawn_center,
    "clusters": spawn_clusters,
}


# --- Simulation functions ---


def sense_species(spx, spy, sph, trail_map, cfg):
    """Classic Jones (2010) sensing — fogleman's direction() logic.

    if C > L and C > R → go straight
    if C < L and C < R → random left or right
    if L < R           → turn right (+rotation_angle)
    if R < L           → turn left  (-rotation_angle)

    Operates on per-species slice views; updates sph in place.
    """
    sensor_angle = cfg["sensor_angle"]
    sensor_distance = cfg["sensor_distance"]
    rotation_angle = cfg["rotation_angle"]

    def sample(angle_offset):
        angles = sph + angle_offset
        sx = (spx + np.cos(angles) * sensor_distance).astype(np.int32) % WIDTH
        sy = (spy + np.sin(angles) * sensor_distance).astype(np.int32) % HEIGHT
        return trail_map[sy, sx]

    C = sample(0)
    L = sample(-sensor_angle)
    R = sample(sensor_angle)

    # Default: no rotation
    da = np.zeros_like(sph)

    # Center strongest → go straight (da stays 0)
    # Center weakest → random left or right
    both_stronger = (C < L) & (C < R)
    signs = rng.integers(0, 2, size=np.count_nonzero(both_stronger), dtype=np.int8)
    da[both_stronger] = rotation_angle * (signs * 2 - 1).astype(np.float32)

    # Left weaker than right → turn right (but not if center is strongest or weakest)
    neither = ~((C > L) & (C > R)) & ~both_stronger
    da[neither & (L < R)] = rotation_angle
    da[neither & (R < L)] = -rotation_angle

    sph += da


def move_species(spx, spy, sph, step_distance):
    spx += np.cos(sph) * step_distance
    spx %= WIDTH
    spy += np.sin(sph) * step_distance
    spy %= HEIGHT


def deposit_species(spx, spy, grid, deposit_amount):
    gx = spx.astype(np.int32) % WIDTH
    gy = spy.astype(np.int32) % HEIGHT
    counts = np.bincount(gy * WIDTH + gx, minlength=WIDTH * HEIGHT)
    grid += counts.astype(np.float32).reshape(HEIGHT, WIDTH) * deposit_amount


def blur_and_decay(grid, radius, iterations, decay_factor):
    size = 2 * radius + 1
    for _ in range(iterations):
        grid = uniform_filter(grid, size=size, mode="wrap")
    grid *= decay_factor
    return grid


# --- Rendering ---


def draw(screen, grids, palette_colors, num_species):
    rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)

    for s in range(num_species):
        grid = grids[s]
        max_val = np.float32(np.percentile(grid, PERCENTILE)) * HEADROOM
        if max_val < 1e-10:
            max_val = np.float32(1.0)

        normalized = np.clip(grid / max_val, 0.0, 1.0)
        corrected = normalized**GAMMA

        color = palette_colors[s % len(palette_colors)]
        rgb[:, :, 0] += corrected * np.float32(color[0])
        rgb[:, :, 1] += corrected * np.float32(color[1])
        rgb[:, :, 2] += corrected * np.float32(color[2])

    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)

    if PIXEL_SCALE > 1:
        rgb_u8 = np.repeat(np.repeat(rgb_u8, PIXEL_SCALE, axis=0), PIXEL_SCALE, axis=1)
    pygame.surfarray.blit_array(screen, rgb_u8.transpose(1, 0, 2))


# --- Initialization helper ---


def init_simulation(mode, num_species):
    num_particles = NUM_PARTICLES_PER_SPECIES * num_species

    configs = generate_random_configs(num_species)
    attraction = generate_random_attraction(num_species)

    # Particles are contiguous per-species blocks, so slices are zero-copy views
    px, py, ph = MODES[mode](num_particles, num_species)
    grids = rng.random((num_species, HEIGHT, WIDTH), dtype=np.float32) * np.float32(0.1)
    slices = [
        slice(s * NUM_PARTICLES_PER_SPECIES, (s + 1) * NUM_PARTICLES_PER_SPECIES)
        for s in range(num_species)
    ]

    # Print config summary
    print(f"\nSpecies configs ({num_species} species, {num_particles} particles):")
    for i, cfg in enumerate(configs):
        print(
            f"  [{i}] StepDist={cfg['step_distance']:.2f}  "
            f"SensorDist={cfg['sensor_distance']:.1f}  "
            f"SensorAngle={math.degrees(cfg['sensor_angle']):.0f}°  "
            f"RotAngle={math.degrees(cfg['rotation_angle']):.0f}°  "
            f"Deposit={cfg['deposit']:.1f}  "
            f"Decay={cfg['decay']:.2f}"
        )
    print()

    return px, py, ph, grids, slices, configs, attraction


# --- Main ---


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "random"
    if mode not in MODES:
        print(f"Unknown mode '{mode}'. Choose from: {', '.join(MODES)}")
        sys.exit(1)

    palette_arg = sys.argv[2] if len(sys.argv) > 2 else "fogleman"
    if palette_arg == "random":
        palette_idx = int(rng.integers(len(PALETTE_NAMES)))
    elif palette_arg in PALETTES:
        palette_idx = PALETTE_NAMES.index(palette_arg)
    else:
        print(
            f"Unknown palette '{palette_arg}'. Choose from: {', '.join(PALETTES)}, random"
        )
        sys.exit(1)

    num_species = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    num_species = max(1, min(4, num_species))

    save_gif = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    px, py, ph, grids, slices, configs, attraction = init_simulation(
        mode, num_species
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
                    px, py, ph, grids, slices, configs, attraction = (
                        init_simulation(mode, num_species)
                    )
                    tick = 0

        start = time.time()

        # Sense — all combined grids at once: (S,S) @ (S,H,W) -> (S,H,W)
        combined = np.tensordot(attraction, grids, axes=1)
        for s, cfg in enumerate(configs):
            sl = slices[s]
            sense_species(px[sl], py[sl], ph[sl], combined[s], cfg)

        for s, cfg in enumerate(configs):
            sl = slices[s]
            move_species(px[sl], py[sl], ph[sl], cfg["step_distance"])
            deposit_species(px[sl], py[sl], grids[s], cfg["deposit"])

        for s, cfg in enumerate(configs):
            grids[s] = blur_and_decay(
                grids[s], BLUR_RADIUS, BLUR_ITERATIONS, cfg["decay"]
            )

        current_palette = PALETTES[PALETTE_NAMES[palette_idx]]
        draw(screen, grids, current_palette, num_species)
        pygame.display.flip()

        # --- Capture frame ---
        if save_gif:
            frame = pygame.surfarray.array3d(screen)
            frame = np.transpose(
                frame, (1, 0, 2)
            )  # Pygame uses (width,height), GIF needs (height,width)
            frames.append(frame)
        # --- Capture frame done ---

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

    if save_gif:
        imageio.mimsave("output.gif", list(frames), fps=30)


if __name__ == "__main__":
    main()
