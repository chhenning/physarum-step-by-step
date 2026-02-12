"""Step 2: Multiple particles with random headings."""

import math
import os
import random
import time

# Grid dimensions
WIDTH = 80
HEIGHT = 60

# Number of particles
NUM_PARTICLES = 80


def create_particle():
    """Create a particle at a random position with a random heading."""
    return {
        "x": random.uniform(0, WIDTH),
        "y": random.uniform(0, HEIGHT),
        "heading": random.uniform(0, 2 * math.pi),
    }


def move(p):
    """Move particle one step in its heading direction, wrap at edges."""
    p["x"] = (p["x"] + math.cos(p["heading"])) % WIDTH
    p["y"] = (p["y"] + math.sin(p["heading"])) % HEIGHT


def draw(particles):
    """Print the grid to the terminal."""
    os.system("clear")

    # Build a set of occupied cells for fast lookup
    occupied = set()
    for p in particles:
        occupied.add((int(p["x"]), int(p["y"])))

    for y in range(HEIGHT):
        row = ""
        for x in range(WIDTH):
            if (x, y) in occupied:
                row += "@"
            else:
                row += "."
        print(row)


# Create all particles
particles = [create_particle() for _ in range(NUM_PARTICLES)]

# Main loop
for tick in range(300):
    draw(particles)
    print(f"\ntick={tick}  particles={NUM_PARTICLES}")

    for p in particles:
        move(p)

    time.sleep(0.08)
