"""Step 10: Separable box blur — replace scipy's uniform_filter with a hand-rolled
two-pass blur using a cumsum sliding window. Configurable radius and iteration
count, with decay folded into the final pass.

Builds on Step 9 (NumPy vectorization). The only changes are in diffusion:
  - separable_box_blur() replaces scipy.ndimage.uniform_filter
  - BLUR_RADIUS and BLUR_ITERATIONS are new tunable constants
  - decay is folded into blur_and_decay(), eliminating a separate grid traversal

Try different radii to see how trail character changes:
  radius 1  = tight networks (same as 3x3 mean filter)
  radius 5  = smoother, wider trails
  radius 10 = very diffuse, soft glow

    python src/step_10_separable_box_blur.py [random|ring|center|clusters]
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

# --- Blur parameters (NEW in this step) ---
BLUR_RADIUS = 1  # radius 1 → 3x3 window (matches step 9); try 3, 5, 10
BLUR_ITERATIONS = 1  # 3 iterations ≈ Gaussian blur

# --- Sensor parameters ---
SENSOR_ANGLE = math.radians(45)
SENSOR_DISTANCE = 3
ROTATION_ANGLE = math.radians(45)

# --- Display ---
PIXEL_SCALE = 4
SCREEN_WIDTH = WIDTH * PIXEL_SCALE
SCREEN_HEIGHT = HEIGHT * PIXEL_SCALE
FPS = 60


# --- Spawn modes (unchanged from step 9) ---


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


# --- Simulation functions ---


def create_trail_map():
    return np.zeros((HEIGHT, WIDTH), dtype=np.float64)


def sense(px, py, ph, trail_map):
    """Sample three sensors per particle and rotate toward the strongest trail."""

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
    """Advance each particle one step along its heading (toroidal wrapping)."""
    px[:] = (px + np.cos(ph)) % WIDTH
    py[:] = (py + np.sin(ph)) % HEIGHT


def deposit(px, py, trail_map):
    """Add pheromone at each particle's grid cell."""
    gx = px.astype(int) % WIDTH
    gy = py.astype(int) % HEIGHT
    np.add.at(trail_map, (gy, gx), DEPOSIT_AMOUNT)


# --- Separable box blur (NEW in this step) ---


def separable_box_blur(grid, radius):
    """Two-pass box blur: horizontal then vertical, using cumsum sliding window.

    Instead of touching 9 neighbors per cell (3x3 kernel), this uses a running
    sum that slides across each row and then each column. The cost is O(W*H)
    per pass regardless of radius — adding the entering pixel and subtracting
    the leaving pixel.

    With numpy we implement the sliding window via cumulative sums:
      padded_cumsum[i + window] - padded_cumsum[i]  =  sum of window elements

    Wrapping is handled by np.pad(mode='wrap') before each pass.
    """
    d = 2 * radius + 1  # window diameter
    h, w = grid.shape

    # --- Horizontal pass: blur along each row ---
    # Pad left/right with wrap-around values so the sliding window
    # sees the correct neighbors at the grid edges.
    padded = np.pad(grid, ((0, 0), (radius, radius)), mode="wrap")
    # Prepend a zero column so that cs[:, j+d] - cs[:, j] gives the
    # sum of d elements starting at padded column j.
    cs = np.zeros((h, padded.shape[1] + 1))
    cs[:, 1:] = np.cumsum(padded, axis=1)
    blurred = (cs[:, d:] - cs[:, :w]) / d

    # --- Vertical pass: blur along each column ---
    padded = np.pad(blurred, ((radius, radius), (0, 0)), mode="wrap")
    cs = np.zeros((padded.shape[0] + 1, w))
    cs[1:, :] = np.cumsum(padded, axis=0)
    blurred = (cs[d:, :] - cs[:h, :]) / d

    return blurred


def blur_and_decay(trail_map, radius, iterations, decay_factor):
    """Apply separable box blur for N iterations, folding decay into the result.

    Multiple iterations of box blur approximate a Gaussian blur:
      1 iteration = box blur (flat/uniform)
      2 iterations ≈ triangular blur
      3 iterations ≈ Gaussian blur (smooth bell curve)

    Decay is multiplied once at the end rather than traversing the grid
    separately — one less full-grid operation per frame.
    """
    for _ in range(iterations):
        trail_map = separable_box_blur(trail_map, radius)

    # Fold decay into the result (saves a separate decay pass)
    trail_map *= decay_factor
    return trail_map


# --- Rendering (unchanged from step 9) ---


def draw(screen, trail_map):
    brightness = np.clip(trail_map / 10.0, 0.0, 1.0)

    r = (brightness * 140).astype(np.uint8)
    g = (brightness * 255).astype(np.uint8)
    b = (brightness * 100).astype(np.uint8)
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
            f"blur_r={BLUR_RADIUS} iter={BLUR_ITERATIONS}  "
            f"{elapsed:.0f}ms"
        )

        clock.tick(FPS)
        tick += 1

    pygame.quit()


if __name__ == "__main__":
    main()
