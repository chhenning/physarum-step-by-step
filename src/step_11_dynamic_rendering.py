"""Step 11: Dynamic range rendering — adaptive brightness with percentile
normalization and gamma correction.

Builds on Step 10 (separable box blur). The only change is in rendering:
  - draw() now computes the 99.9th percentile of trail values each frame
    and uses that (× 1.5 headroom) as the normalization maximum
  - a GAMMA constant controls the brightness curve:
      gamma < 1.0 (e.g. 0.45) → reveals faint trail structure
      gamma > 1.0 (e.g. 2.0)  → emphasizes only the strongest trails

The fixed "value / 10.0" clamp from earlier steps is gone. The full color
range is now always used, regardless of how much trail has accumulated.

    python src/step_11_dynamic_rendering.py [random|ring|center|clusters]
"""

import math
import sys
import time

import numpy as np
import pygame

# --- Grid dimensions ---
WIDTH = 320
HEIGHT = 240

# --- Number of particles ---
NUM_PARTICLES = 2000

# --- Trail parameters ---
DEPOSIT_AMOUNT = 5.0
DECAY_FACTOR = 0.95

# --- Blur parameters ---
BLUR_RADIUS = 1
BLUR_ITERATIONS = 1

# --- Sensor parameters ---
SENSOR_ANGLE = math.radians(45)
SENSOR_DISTANCE = 3
ROTATION_ANGLE = math.radians(45)

# --- Display ---
PIXEL_SCALE = 4
SCREEN_WIDTH = WIDTH * PIXEL_SCALE
SCREEN_HEIGHT = HEIGHT * PIXEL_SCALE
FPS = 60

# --- Rendering parameters (NEW in this step) ---
GAMMA = 0.45  # < 1 reveals faint trails; > 1 emphasizes strong trails
PERCENTILE = 99.9  # use the 99.9th percentile as the brightness ceiling
HEADROOM = 1.5  # multiply percentile by this to avoid full saturation


# --- Spawn modes (unchanged) ---


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
    ph = angles + np.pi
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
    px = np.concatenate(
        [
            np.random.normal(WIDTH * 0.25, 2, half),
            np.random.normal(WIDTH * 0.75, 2, rest),
        ]
    )
    py = np.random.normal(HEIGHT / 2, 2, NUM_PARTICLES)
    ph = np.random.uniform(0, 2 * np.pi, NUM_PARTICLES)
    return px, py, ph


MODES = {
    "random": spawn_random,
    "ring": spawn_ring,
    "center": spawn_center,
    "clusters": spawn_clusters,
}


# --- Simulation functions (unchanged from step 10) ---


def create_trail_map():
    return np.zeros((HEIGHT, WIDTH), dtype=np.float64)


def sense(px, py, ph, trail_map):
    def sample(angle_offset):
        angles = ph + angle_offset
        sx = (px + np.cos(angles) * SENSOR_DISTANCE).astype(int) % WIDTH
        sy = (py + np.sin(angles) * SENSOR_DISTANCE).astype(int) % HEIGHT
        return trail_map[sy, sx]

    val_left = sample(-SENSOR_ANGLE)
    val_front = sample(0)
    val_right = sample(SENSOR_ANGLE)

    random_turns = np.where(
        np.random.randint(0, 2, len(px)) == 0,
        -ROTATION_ANGLE,
        ROTATION_ANGLE,
    )

    front_is_best = (val_front >= val_left) & (val_front >= val_right)
    left_is_better = val_left > val_right
    right_is_better = val_right > val_left

    rotation = np.where(
        front_is_best,
        0.0,
        np.where(
            left_is_better,
            -ROTATION_ANGLE,
            np.where(
                right_is_better,
                ROTATION_ANGLE,
                random_turns,
            ),
        ),
    )

    ph[:] = ph + rotation


def move(px, py, ph):
    px[:] = (px + np.cos(ph)) % WIDTH
    py[:] = (py + np.sin(ph)) % HEIGHT


def deposit(px, py, trail_map):
    gx = px.astype(int) % WIDTH
    gy = py.astype(int) % HEIGHT
    np.add.at(trail_map, (gy, gx), DEPOSIT_AMOUNT)


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


def blur_and_decay(trail_map, radius, iterations, decay_factor):
    for _ in range(iterations):
        trail_map = separable_box_blur(trail_map, radius)
    trail_map *= decay_factor
    return trail_map


# --- Adaptive rendering (NEW in this step) ---


def draw(screen, trail_map):
    """Render with percentile normalization and gamma correction.

    Instead of the fixed `value / 10.0` clamp from earlier steps:
      1. Find the 99.9th percentile of trail values as the brightness ceiling.
         Multiply by HEADROOM (1.5) so the very brightest pixels aren't fully
         saturated — this preserves detail in the hottest spots.
      2. Normalize all values to [0, 1] using that ceiling.
      3. Apply gamma correction: value ** GAMMA.
         gamma < 1 compresses highs and expands lows → faint trails become visible.
         gamma > 1 compresses lows and expands highs → only strong trails show.
      4. Map the corrected values to RGB and blit via surfarray.

    The result: brightness adapts automatically to whatever trail intensity
    exists in the current frame. Early frames (low trail) and late frames
    (high trail) both use the full color range.
    """
    # Adaptive normalization: use percentile as ceiling
    max_val = np.percentile(trail_map, PERCENTILE) * HEADROOM
    if max_val < 1e-10:
        max_val = 1.0  # avoid division by zero on empty frames

    # Normalize to [0, 1] and apply gamma correction
    normalized = np.clip(trail_map / max_val, 0.0, 1.0)
    corrected = normalized ** GAMMA

    # Map to greenish RGB
    r = (corrected * 140).astype(np.uint8)
    g = (corrected * 255).astype(np.uint8)
    b = (corrected * 100).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)

    scaled = np.repeat(np.repeat(rgb, PIXEL_SCALE, axis=0), PIXEL_SCALE, axis=1)
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

    px, py, ph = MODES[mode]()
    trail_map = create_trail_map()
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

        sense(px, py, ph, trail_map)
        move(px, py, ph)
        deposit(px, py, trail_map)
        trail_map = blur_and_decay(trail_map, BLUR_RADIUS, BLUR_ITERATIONS, DECAY_FACTOR)

        draw(screen, trail_map)
        pygame.display.flip()

        elapsed = (time.time() - start) * 1000
        pygame.display.set_caption(
            f"Physarum — {mode}  |  tick={tick}  "
            f"gamma={GAMMA}  "
            f"{elapsed:.0f}ms"
        )

        clock.tick(FPS)
        tick += 1

    pygame.quit()


if __name__ == "__main__":
    main()
