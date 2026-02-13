"""Step 9: NumPy-accelerated Physarum with pygame visualization.

Replaces the pure-Python simulation from Step 8 with vectorized NumPy arrays
for particle state and trail computation. Uses scipy's uniform_filter for
diffusion. Rendering is fully vectorized via surfarray.

    python src/step_09_numpy_vectorization.py [random|ring|center|clusters]
"""

import math
import sys
import time

import numpy as np
import pygame
from scipy.ndimage import uniform_filter

# --- Grid dimensions ---
WIDTH = 320
HEIGHT = 240

# --- Number of particles ---
NUM_PARTICLES = 2000

# --- Trail parameters ---
DEPOSIT_AMOUNT = 5.0
DECAY_FACTOR = 0.95

# --- Sensor parameters ---
SENSOR_ANGLE = math.radians(45)
SENSOR_DISTANCE = 3
ROTATION_ANGLE = math.radians(45)

# --- Display ---
PIXEL_SCALE = 4  # each grid cell = PIXEL_SCALE x PIXEL_SCALE screen pixels
SCREEN_WIDTH = WIDTH * PIXEL_SCALE
SCREEN_HEIGHT = HEIGHT * PIXEL_SCALE
FPS = 60


# --- Spawn modes ---


def spawn_random():
    """Scatter all particles uniformly across the grid with random headings."""
    px = np.random.uniform(0, WIDTH, NUM_PARTICLES)
    py = np.random.uniform(0, HEIGHT, NUM_PARTICLES)
    ph = np.random.uniform(0, 2 * np.pi, NUM_PARTICLES)
    return px, py, ph


def spawn_ring():
    """Place particles in a ring around the center, all facing inward."""
    cx, cy = WIDTH / 2, HEIGHT / 2
    radius = min(WIDTH, HEIGHT) * 0.35
    angles = np.linspace(0, 2 * np.pi, NUM_PARTICLES, endpoint=False)
    px = cx + np.cos(angles) * radius
    py = cy + np.sin(angles) * radius
    ph = angles + np.pi
    return px, py, ph


def spawn_center():
    """Cluster all particles tightly at the grid center with random headings."""
    cx, cy = WIDTH / 2, HEIGHT / 2
    px = np.random.normal(cx, 2, NUM_PARTICLES)
    py = np.random.normal(cy, 2, NUM_PARTICLES)
    ph = np.random.uniform(0, 2 * np.pi, NUM_PARTICLES)
    return px, py, ph


def spawn_clusters():
    """Split particles into two tight clusters at 25% and 75% of grid width."""
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
    """Create a zero-initialized trail map of shape (HEIGHT, WIDTH)."""
    return np.zeros((HEIGHT, WIDTH), dtype=np.float64)


def sense(px, py, ph, trail_map):
    """Sample three sensors per particle and rotate toward the strongest trail.

    Each particle probes the trail map at left, front, and right sensor
    positions. The heading array ph is updated in-place based on which
    sensor reads the highest value. Ties are broken randomly.
    """

    def sample(angle_offset):
        """Return trail values at sensor positions offset from each particle's heading."""
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
    """Add pheromone to the trail map at each particle's grid cell.

    Uses np.add.at for unbuffered accumulation so that multiple particles
    in the same cell each contribute their deposit.
    """
    gx = px.astype(int) % WIDTH
    gy = py.astype(int) % HEIGHT
    np.add.at(trail_map, (gy, gx), DEPOSIT_AMOUNT)


def diffuse(trail_map):
    """Apply a 3x3 mean filter with toroidal (wrap-around) boundaries.

    Returns a new array; the original is not modified.
    """
    return uniform_filter(trail_map, size=3, mode="wrap")


def decay(trail_map):
    """Reduce all trail values by DECAY_FACTOR (in-place multiplication)."""
    trail_map *= DECAY_FACTOR


# --- Rendering ---


def draw(screen, trail_map):
    """Render the trail map onto the pygame surface as scaled RGB pixels.

    Maps trail intensity to a greenish color, scales up by PIXEL_SCALE,
    and blits directly via surfarray (no per-pixel Python loop).
    """
    brightness = np.clip(trail_map / 10.0, 0.0, 1.0)

    r = (brightness * 140).astype(np.uint8)
    g = (brightness * 255).astype(np.uint8)
    b = (brightness * 100).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)

    scaled = np.repeat(np.repeat(rgb, PIXEL_SCALE, axis=0), PIXEL_SCALE, axis=1)

    # surfarray expects (width, height, 3) so transpose the spatial axes
    pygame.surfarray.blit_array(screen, scaled.transpose(1, 0, 2))


# --- Main ---


def main():
    """Run the simulation loop: sense, move, deposit, diffuse, decay, render."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "random"
    if mode not in MODES:
        print(f"Unknown mode '{mode}'. Choose from: {', '.join(MODES)}")
        sys.exit(1)

    # Start up all pygame subsystems (video, audio, etc.)
    pygame.init()
    # Create a window with the given pixel dimensions
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    # Set the text shown in the window's title bar
    pygame.display.set_caption(f"Physarum — {mode}")
    # Create a clock to control the frame rate
    clock = pygame.time.Clock()

    px, py, ph = MODES[mode]()
    trail_map = create_trail_map()
    tick = 0

    running = True
    while running:
        # Check all user inputs (clicks, keys, window close)
        for event in pygame.event.get():
            # User clicked the window's close button
            if event.type == pygame.QUIT:
                running = False
            # User pressed a key on the keyboard
            elif event.type == pygame.KEYDOWN:
                # The key pressed was Escape
                if event.key == pygame.K_ESCAPE:
                    running = False

        start = time.time()

        sense(px, py, ph, trail_map)
        move(px, py, ph)
        deposit(px, py, trail_map)
        trail_map = diffuse(trail_map)
        decay(trail_map)

        draw(screen, trail_map)
        # Push the newly drawn frame to the screen (swap front/back buffers)
        pygame.display.flip()

        elapsed = (time.time() - start) * 1000
        pygame.display.set_caption(
            f"Physarum — {mode}  |  tick={tick}  particles={NUM_PARTICLES}  {elapsed:.0f}ms"
        )

        # Wait just enough to cap the loop at FPS frames per second
        clock.tick(FPS)
        tick += 1

    # Shut down all pygame subsystems and close the window
    pygame.quit()


if __name__ == "__main__":
    main()
