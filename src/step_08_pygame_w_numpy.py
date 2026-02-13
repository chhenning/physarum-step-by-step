"""Step 7: Pygame visualization — render the trail map as colored pixels.

Same simulation as Step 6 but replaces ASCII output with a pygame window.
Each trail cell maps to a pixel (scaled up) with brightness based on intensity.

    python src/step_07_pygame.py [random|ring|center|clusters]
"""

import math
import random
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
    particles = [
        {
            "x": random.uniform(0, WIDTH),
            "y": random.uniform(0, HEIGHT),
            "heading": random.uniform(0, 2 * math.pi),
        }
        for _ in range(NUM_PARTICLES)
    ]

    return particles


def spawn_ring():
    cx, cy = WIDTH / 2, HEIGHT / 2
    radius = min(WIDTH, HEIGHT) * 0.35
    particles = []
    for i in range(NUM_PARTICLES):
        angle = 2 * math.pi * i / NUM_PARTICLES
        x = cx + math.cos(angle) * radius
        y = cy + math.sin(angle) * radius
        heading = angle + math.pi
        particles.append({"x": x, "y": y, "heading": heading})
    return particles


def spawn_center():
    cx, cy = WIDTH / 2, HEIGHT / 2
    return [
        {
            "x": cx + random.gauss(0, 2),
            "y": cy + random.gauss(0, 2),
            "heading": random.uniform(0, 2 * math.pi),
        }
        for _ in range(NUM_PARTICLES)
    ]


def spawn_clusters():
    particles = []
    for _ in range(NUM_PARTICLES // 2):
        particles.append(
            {
                "x": WIDTH * 0.25 + random.gauss(0, 2),
                "y": HEIGHT / 2 + random.gauss(0, 2),
                "heading": random.uniform(0, 2 * math.pi),
            }
        )
    for _ in range(NUM_PARTICLES - NUM_PARTICLES // 2):
        particles.append(
            {
                "x": WIDTH * 0.75 + random.gauss(0, 2),
                "y": HEIGHT / 2 + random.gauss(0, 2),
                "heading": random.uniform(0, 2 * math.pi),
            }
        )
    return particles


MODES = {
    "random": spawn_random,
    "ring": spawn_ring,
    "center": spawn_center,
    "clusters": spawn_clusters,
}


# --- Simulation functions ---


def create_trail_map():
    return np.zeros((HEIGHT, WIDTH), dtype=np.float64)


def sense(p, trail_map):
    h = p["heading"]
    x, y = p["x"], p["y"]

    def sample(angle):
        sx = int(x + math.cos(angle) * SENSOR_DISTANCE) % WIDTH
        sy = int(y + math.sin(angle) * SENSOR_DISTANCE) % HEIGHT
        return trail_map[sy][sx]

    val_left = sample(h - SENSOR_ANGLE)
    val_front = sample(h)
    val_right = sample(h + SENSOR_ANGLE)

    if val_front >= val_left and val_front >= val_right:
        pass
    elif val_left > val_right:
        p["heading"] -= ROTATION_ANGLE
    elif val_right > val_left:
        p["heading"] += ROTATION_ANGLE
    else:
        p["heading"] += random.choice([-1, 1]) * ROTATION_ANGLE


def move(p):
    p["x"] = (p["x"] + math.cos(p["heading"])) % WIDTH
    p["y"] = (p["y"] + math.sin(p["heading"])) % HEIGHT


def deposit(particles, trail_map):
    for p in particles:
        gx, gy = int(p["x"]) % WIDTH, int(p["y"]) % HEIGHT
        trail_map[gy][gx] += DEPOSIT_AMOUNT


def diffuse(trail_map):
    new_map = create_trail_map()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            total = 0.0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nx = (x + dx) % WIDTH
                    ny = (y + dy) % HEIGHT
                    total += trail_map[ny][nx]
            new_map[y][x] = total / 9.0
    return new_map


def decay(trail_map):
    trail_map *= DECAY_FACTOR


# --- Rendering ---


def trail_to_color(value):
    """Map trail intensity to a greenish-white color."""
    brightness = min(value / 10.0, 1.0)
    r = int(brightness * 140)
    g = int(brightness * 255)
    b = int(brightness * 100)
    return (r, g, b)


def draw(screen, trail_map):
    """Render the trail map onto the pygame surface."""
    for y in range(HEIGHT):
        for x in range(WIDTH):
            color = trail_to_color(trail_map[y][x])
            rect = (x * PIXEL_SCALE, y * PIXEL_SCALE, PIXEL_SCALE, PIXEL_SCALE)
            # Draw a filled rectangle (one grid cell) with the given color
            screen.fill(color, rect)


# --- Main ---


def main():
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

    particles = MODES[mode]()
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

        for p in particles:
            sense(p, trail_map)
        for p in particles:
            move(p)
        deposit(particles, trail_map)
        trail_map = diffuse(trail_map)
        decay(trail_map)

        # Clear the entire screen to black before drawing the new frame
        screen.fill((0, 0, 0))
        draw(screen, trail_map)
        # Push the newly drawn frame to the screen (swap front/back buffers)
        pygame.display.flip()

        elapsed = (time.time() - start) * 1000
        # Update the title bar with live stats
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
