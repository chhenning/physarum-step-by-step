"""Step 12: Variable step distance — tunable movement and sensor range with
real-time keyboard controls.

Builds on Step 11 (dynamic rendering). Changes:
  - STEP_DISTANCE scales how far particles move each tick
  - SENSOR_DISTANCE is now a float for finer control
  - Keyboard controls adjust both parameters live:
      Up/Down    — step distance  ±0.1
      Left/Right — sensor distance ±1.0
      R          — reset both to defaults

Try the extremes:
  small step (0.2–0.5)  → tight, dense networks
  large step (1.5–2.0)  → sparse, long-range connections
  small sensor (1–2)    → local sensing, dense webs
  large sensor (9–15)   → far sensing, sparse highways

    python src/step_12_variable_step_distance.py [random|ring|center|clusters]
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
NUM_PARTICLES = 4000

# --- Trail parameters ---
DEPOSIT_AMOUNT = 5.0
DECAY_FACTOR = 0.98

# --- Blur parameters ---
BLUR_RADIUS = 1
BLUR_ITERATIONS = 1

# --- Sensor parameters ---
SENSOR_ANGLE = math.radians(45)
ROTATION_ANGLE = math.radians(45)

# --- Defaults for tunable parameters (NEW in this step) ---
DEFAULT_STEP_DISTANCE = 1.0
DEFAULT_SENSOR_DISTANCE = 3.0

# --- Display ---
PIXEL_SCALE = 4
SCREEN_WIDTH = WIDTH * PIXEL_SCALE
SCREEN_HEIGHT = HEIGHT * PIXEL_SCALE
FPS = 60

# --- Rendering parameters ---
GAMMA = 0.45
PERCENTILE = 99.9
HEADROOM = 1.5


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


# --- Simulation functions ---


def create_trail_map():
    return np.zeros((HEIGHT, WIDTH), dtype=np.float64)


def sense(px, py, ph, trail_map, sensor_distance):
    """Sample three sensors per particle. sensor_distance is now a parameter."""

    def sample(angle_offset):
        angles = ph + angle_offset
        sx = (px + np.cos(angles) * sensor_distance).astype(int) % WIDTH
        sy = (py + np.sin(angles) * sensor_distance).astype(int) % HEIGHT
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


def move(px, py, ph, step_distance):
    """Advance particles by step_distance along their heading (NEW: was hardcoded to 1.0)."""
    px[:] = (px + np.cos(ph) * step_distance) % WIDTH
    py[:] = (py + np.sin(ph) * step_distance) % HEIGHT


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


# --- Rendering (unchanged from step 11) ---


def draw(screen, trail_map):
    max_val = np.percentile(trail_map, PERCENTILE) * HEADROOM
    if max_val < 1e-10:
        max_val = 1.0

    normalized = np.clip(trail_map / max_val, 0.0, 1.0)
    corrected = normalized**GAMMA

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

    # Mutable parameters — adjusted with keyboard (NEW in this step)
    step_distance = DEFAULT_STEP_DISTANCE
    sensor_distance = DEFAULT_SENSOR_DISTANCE

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                # Step distance: Up/Down arrows in ±0.1 increments
                elif event.key == pygame.K_UP:
                    step_distance = min(step_distance + 0.1, 3.0)
                elif event.key == pygame.K_DOWN:
                    step_distance = max(step_distance - 0.1, 0.1)
                # Sensor distance: Left/Right arrows in ±1.0 increments
                elif event.key == pygame.K_RIGHT:
                    sensor_distance = min(sensor_distance + 1.0, 20.0)
                elif event.key == pygame.K_LEFT:
                    sensor_distance = max(sensor_distance - 1.0, 1.0)
                # Reset both to defaults
                elif event.key == pygame.K_r:
                    step_distance = DEFAULT_STEP_DISTANCE
                    sensor_distance = DEFAULT_SENSOR_DISTANCE

        start = time.time()

        sense(px, py, ph, trail_map, sensor_distance)
        move(px, py, ph, step_distance)
        deposit(px, py, trail_map)
        trail_map = blur_and_decay(
            trail_map, BLUR_RADIUS, BLUR_ITERATIONS, DECAY_FACTOR
        )

        draw(screen, trail_map)
        pygame.display.flip()

        elapsed = (time.time() - start) * 1000
        pygame.display.set_caption(
            f"Physarum — {mode}  |  tick={tick}  "
            f"step={step_distance:.1f}  sensor={sensor_distance:.0f}  "
            f"{elapsed:.0f}ms  "
            f"[Up/Down=step  Left/Right=sensor  R=reset]"
        )

        clock.tick(FPS)
        tick += 1

    pygame.quit()


if __name__ == "__main__":
    main()
