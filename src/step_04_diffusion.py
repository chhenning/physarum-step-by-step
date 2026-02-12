"""Step 4: Diffusion — blurring the trail map.

New concept: After deposit but before decay, apply a 3x3 mean filter
to the trail map. This spreads trail values to neighboring cells,
simulating how chemicals diffuse through a medium.
"""

import math
import os
import random
import time

# Grid dimensions
WIDTH = 80
HEIGHT = 60

# Number of particles
NUM_PARTICLES = 20

# Trail parameters
DEPOSIT_AMOUNT = 5.0
DECAY_FACTOR = 0.99

# Map intensity to characters (low → high)
INTENSITY_CHARS = " .:-=+*#%@"


def create_particle():
    """Create a particle at a random position with a random heading."""
    return {
        "x": random.uniform(0, WIDTH),
        "y": random.uniform(0, HEIGHT),
        "heading": random.uniform(0, 2 * math.pi),
    }


def create_trail_map():
    """Create a 2D grid of floats, initialized to 0."""
    return [[0.0] * WIDTH for _ in range(HEIGHT)]


def move(p):
    """Move particle one step in its heading direction, wrap at edges."""
    p["x"] = (p["x"] + math.cos(p["heading"])) % WIDTH
    p["y"] = (p["y"] + math.sin(p["heading"])) % HEIGHT


def deposit(particles, trail_map):
    """Each particle deposits onto the trail map at its position."""
    for p in particles:
        gx, gy = int(p["x"]) % WIDTH, int(p["y"]) % HEIGHT
        trail_map[gy][gx] += DEPOSIT_AMOUNT


def diffuse(trail_map):
    """Apply a 3x3 mean filter to the trail map.

    Each cell becomes the average of itself and its 8 neighbors.
    Uses a copy to avoid read/write conflicts (double buffering).
    Wraps at edges using modulo.
    """
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
    """Multiply every cell by the decay factor."""
    for y in range(HEIGHT):
        for x in range(WIDTH):
            trail_map[y][x] *= DECAY_FACTOR


def intensity_to_char(value):
    """Map a float intensity to a display character."""
    index = int(value)
    index = min(index, len(INTENSITY_CHARS) - 1)
    return INTENSITY_CHARS[index]


def draw(trail_map):
    """Print the trail map to the terminal."""
    os.system("clear")

    used_chars = set()

    for y in range(HEIGHT):
        row = ""
        for x in range(WIDTH):
            c = intensity_to_char(trail_map[y][x])
            used_chars.add(c)
            row += c
        print(row)

    return list(used_chars)


# Create particles and trail map
particles = [create_particle() for _ in range(NUM_PARTICLES)]
trail_map = create_trail_map()

# Main loop
for tick in range(300):

    start = time.time()

    # 1. Move all particles
    for p in particles:
        move(p)

    # 2. Deposit onto trail map
    deposit(particles, trail_map)

    # 3. Diffuse the trail map (blur)
    trail_map = diffuse(trail_map)

    # 4. Decay the trail map
    decay(trail_map)

    # 5. Display
    used_chars = draw(trail_map)

    end = time.time()

    print(
        f"\ntick={tick}  particles={NUM_PARTICLES}  decay={DECAY_FACTOR} time={(end-start)*1000:.0f}ms"
    )

    print(used_chars)

    time.sleep(0.08)
