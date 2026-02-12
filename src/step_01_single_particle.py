"""Step 1: A single particle moving on a grid."""

import math
import os
import time

# Grid dimensions
WIDTH = 60
HEIGHT = 30

# Particle state
particle = {
    "x": WIDTH / 2,
    "y": HEIGHT / 2,
    "heading": math.radians(10),  # angle in radians
}

# Track visited cells - so that the trail can be visualized
visited = set()


def move(p):
    """Move particle one step in its heading direction, wrap at edges."""
    p["x"] = (p["x"] + math.cos(p["heading"])) % WIDTH
    p["y"] = (p["y"] + math.sin(p["heading"])) % HEIGHT


def draw(p):
    """Print the grid to the terminal."""
    os.system("clear")
    px, py = int(p["x"]), int(p["y"])
    visited.add((px, py))

    for y in range(HEIGHT):
        row = ""
        for x in range(WIDTH):
            if x == px and y == py:
                row += "@"  # current position
            elif (x, y) in visited:
                row += "*"  # trail
            else:
                row += "."
        print(row)


# Main loop
for tick in range(200):
    draw(particle)
    print(
        f"\ntick={tick}  x={particle['x']:.1f}  y={particle['y']:.1f}  heading={math.degrees(particle['heading']):.0f}°"
    )
    move(particle)
    time.sleep(0.08)
