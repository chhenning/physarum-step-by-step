"""Step 3: Trail map with deposit and decay.

New concept: A separate 2D grid of float intensities (the trail map).
Particles deposit onto it, and it decays each tick.
The display now shows trail intensity instead of just particle positions.
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
DECAY_FACTOR = 0.95  # multiply each cell by this every tick

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
    for y in range(HEIGHT):
        row = ""
        for x in range(WIDTH):
            row += intensity_to_char(trail_map[y][x])
        print(row)


# Create particles and trail map
particles = [create_particle() for _ in range(NUM_PARTICLES)]
trail_map = create_trail_map()

# Main loop
for tick in range(300):
    # 1. Move all particles
    for p in particles:
        move(p)

    # 2. Deposit onto trail map
    deposit(particles, trail_map)

    # 3. Decay the trail map
    decay(trail_map)

    # 4. Display
    draw(trail_map)
    print(f"\ntick={tick}  particles={NUM_PARTICLES}  decay={DECAY_FACTOR}")

    time.sleep(0.08)
