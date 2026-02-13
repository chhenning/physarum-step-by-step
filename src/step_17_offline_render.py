"""Step 17: Offline high-resolution rendering — fogleman-style output.

Builds on Step 16 (final). Changes:
  - No realtime display — runs simulation headless and saves a PNG
  - 1920x1080 resolution with ~1M particles per species
  - Fogleman-scale parameter ranges (large sensor distances, wide angles)
  - Random config generation matching fogleman's Go implementation
  - Configurable number of simulation steps (default 500)
  - Progress output during simulation

    python src/step_17_offline_render.py [--steps 500] [--species 4] [--output out.png]
    python src/step_17_offline_render.py --steps 1000 --species 3 --output render.png
"""

import argparse
import math
import time

import numpy as np
from PIL import Image

# --- Grid dimensions (fogleman uses 1024x1024) ---
WIDTH = 1024
HEIGHT = 1024

# --- Blur parameters ---
BLUR_RADIUS = 1
BLUR_ITERATIONS = 2

# --- Rendering parameters ---
GAMMA = 0.45
PERCENTILE = 99.0
HEADROOM = 1.5

# --- Default particle count (fogleman uses 1<<22 = 4M total) ---
PARTICLES_PER_SPECIES = 1_048_576  # ~1M per species (4M total with 4 species)

# --- Color palettes (fogleman-style, from the output) ---
FOGLEMAN_COLORS = [
    (0xC9, 0x3C, 0x00),  # #C93C00 — orange-red
    (0xE8, 0x88, 0x01),  # #E88801 — amber
    (0x73, 0x00, 0x46),  # #730046 — magenta-purple
    (0xBF, 0xBB, 0x11),  # #BFBB11 — yellow-green
]

PALETTE_PRESETS = {
    "fogleman": FOGLEMAN_COLORS,
    "neon": [(255, 0, 128), (0, 255, 128), (128, 0, 255), (255, 255, 0)],
    "ocean": [(0, 80, 200), (0, 200, 180), (100, 220, 255), (40, 120, 200)],
    "fire": [(255, 60, 20), (255, 160, 0), (255, 240, 80), (200, 40, 0)],
    "forest": [(40, 180, 60), (160, 200, 40), (80, 120, 40), (200, 255, 80)],
}


# --- Random configuration generation (fogleman-scale ranges) ---


def generate_random_configs(n):
    """Generate species configs with fogleman-scale parameter ranges.

    From fogleman's output:
        SensorDistance:  24 — 64
        SensorAngle:    36 — 76 degrees (0.6 — 1.3 radians)
        RotationAngle:  18 — 76 degrees (0.3 — 1.3 radians)
        StepDistance:    1.2 — 1.6
        DecayFactor:    0.1 (grid *= 0.1, keeps 10% — very aggressive)
    """
    configs = []
    for _ in range(n):
        configs.append(
            {
                "sensor_angle": np.random.uniform(0.6, 1.4),
                "sensor_distance": np.random.uniform(20.0, 65.0),
                "rotation_angle": np.random.uniform(0.3, 1.4),
                "step_distance": np.random.uniform(1.0, 1.7),
                "deposit": 5.0,
                "decay": 0.1,
            }
        )
    return configs


def generate_random_attraction(n):
    """Generate attraction matrix: positive diagonal, negative off-diagonal."""
    matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = np.random.uniform(0.5, 1.3)
            else:
                matrix[i, j] = np.random.uniform(-1.3, -0.4)
    return matrix


# --- Spawn ---


def spawn_random(num_particles):
    px = np.random.uniform(0, WIDTH, num_particles)
    py = np.random.uniform(0, HEIGHT, num_particles)
    ph = np.random.uniform(0, 2 * np.pi, num_particles)
    return px, py, ph


# --- Simulation functions ---


def combined_grid(species_idx, grids, attraction):
    weights = attraction[species_idx]
    result = np.zeros_like(grids[0])
    for w, g in zip(weights, grids):
        result += w * g
    return result


def sense_species(px, py, ph, mask, trail_map, cfg):
    """Fogleman-style sensing: classic Jones (2010) direction logic.

    if C > L and C > R → go straight
    if C < L and C < R → random left or right
    if L < R           → turn right (+rotation_angle)
    if R < L           → turn left  (-rotation_angle)
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

    C = sample(0)
    L = sample(-sensor_angle)
    R = sample(sensor_angle)

    # Default: no rotation
    da = np.zeros_like(sph)

    # Center strongest → go straight (da stays 0)
    # Center weakest → random left or right
    both_stronger = (C < L) & (C < R)
    random_sign = (np.random.randint(0, 2, size=spx.size) * 2 - 1).astype(np.float64)
    da = np.where(both_stronger, rotation_angle * random_sign, da)

    # Left weaker than right → turn right (but not if center is strongest or weakest)
    neither = ~((C > L) & (C > R)) & ~both_stronger
    da = np.where(neither & (L < R), rotation_angle, da)
    da = np.where(neither & (R < L), -rotation_angle, da)

    ph[mask] = sph + da


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


# --- Rendering to image ---


def render_image(grids, palette_colors, num_species):
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
    return Image.fromarray(rgb_u8, "RGB")


# --- Main ---


def main():
    global WIDTH, HEIGHT

    parser = argparse.ArgumentParser(description="Physarum offline renderer")
    parser.add_argument(
        "--steps", type=int, default=400, help="Simulation steps (default: 400)"
    )
    parser.add_argument(
        "--species", type=int, default=4, help="Number of species 1-4 (default: 4)"
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=PARTICLES_PER_SPECIES,
        help=f"Particles per species (default: {PARTICLES_PER_SPECIES})",
    )
    parser.add_argument(
        "--palette",
        choices=list(PALETTE_PRESETS.keys()),
        default="fogleman",
        help="Color palette (default: fogleman)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="physarum_render.png",
        help="Output filename (default: physarum_render.png)",
    )
    parser.add_argument(
        "--width", type=int, default=WIDTH, help=f"Grid width (default: {WIDTH})"
    )
    parser.add_argument(
        "--height", type=int, default=HEIGHT, help=f"Grid height (default: {HEIGHT})"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    WIDTH = args.width
    HEIGHT = args.height

    num_species = max(1, min(4, args.species))
    num_particles = args.particles * num_species

    if args.seed is not None:
        np.random.seed(args.seed)

    # Generate configs
    configs = generate_random_configs(num_species)
    attraction = generate_random_attraction(num_species)
    palette_colors = PALETTE_PRESETS[args.palette]

    # Print config summary (like fogleman's output)
    print(f"{args.output}")
    print(f"{num_particles} particles")
    print()
    print("Species configs:")
    for i, cfg in enumerate(configs):
        print(
            f"  [{i}] StepDistance={cfg['step_distance']:.3f}  "
            f"SensorDist={cfg['sensor_distance']:.1f}  "
            f"SensorAngle={math.degrees(cfg['sensor_angle']):.1f}°  "
            f"RotAngle={math.degrees(cfg['rotation_angle']):.1f}°  "
            f"Deposit={cfg['deposit']:.1f}  "
            f"Decay={cfg['decay']:.2f}"
        )
    print()
    print("Attraction matrix:")
    for row in attraction:
        print("  " + "  ".join(f"{v:+.3f}" for v in row))
    print()
    print(
        f"Palette: {args.palette} — {[f'#{r:02X}{g:02X}{b:02X}' for r, g, b in palette_colors[:num_species]]}"
    )
    print()

    # Spawn particles
    px, py, ph = spawn_random(num_particles)
    species = np.repeat(np.arange(num_species), args.particles)
    masks = [species == s for s in range(num_species)]

    # Initialize grids with random noise
    grids = [np.random.random((HEIGHT, WIDTH)) * 0.1 for _ in range(num_species)]

    # Run simulation
    t_start = time.time()
    for step in range(args.steps):
        # Sense
        for s, cfg in enumerate(configs):
            cg = combined_grid(s, grids, attraction)
            sense_species(px, py, ph, masks[s], cg, cfg)

        # Move & deposit
        for s, cfg in enumerate(configs):
            move_species(px, py, ph, masks[s], cfg["step_distance"])
            deposit_species(px, py, masks[s], grids[s], cfg["deposit"])

        # Blur & decay
        for s, cfg in enumerate(configs):
            grids[s] = blur_and_decay(
                grids[s], BLUR_RADIUS, BLUR_ITERATIONS, cfg["decay"]
            )

        # Progress
        elapsed = time.time() - t_start
        if (step + 1) % 10 == 0 or step == 0:
            rate = (step + 1) / elapsed
            eta = (args.steps - step - 1) / rate
            print(
                f"  step {step + 1}/{args.steps}  "
                f"({elapsed:.1f}s elapsed, ~{eta:.0f}s remaining, {rate:.1f} steps/s)"
            )

    total_time = time.time() - t_start
    print()
    print(f"Simulation complete: {total_time:.1f}s")

    # Render and save
    print(f"Rendering image...")
    img = render_image(grids, palette_colors, num_species)
    img.save(args.output)
    print(f"Saved: {args.output} ({WIDTH}x{HEIGHT})")


if __name__ == "__main__":
    main()
