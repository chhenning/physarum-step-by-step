"""Step 5: Sensors — particles sense the trail map and steer.

New concept: Each particle has 3 sensors (front-left, front, front-right).
Before moving, a particle samples the trail map at each sensor position
and turns toward the strongest signal. This closes the feedback loop:
particles deposit trails → trails guide particles → emergent networks.
"""

import math
import os
import random
import time

# Grid dimensions
WIDTH = 80
HEIGHT = 60

# Number of particles
NUM_PARTICLES = 150

# Trail parameters
DEPOSIT_AMOUNT = 5.0
DECAY_FACTOR = 0.95

# Sensor parameters
SENSOR_ANGLE = math.radians(45)  # offset from heading for left/right sensors
SENSOR_DISTANCE = 3  # how far ahead the sensors look
ROTATION_ANGLE = math.radians(45)  # how sharply particles turn

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


def sense(p, trail_map):
    """Sample the trail map at 3 sensor positions, rotate toward strongest.

    Sensors:
        FL (front-left)   = heading - sensor_angle
        F  (front)        = heading
        FR (front-right)  = heading + sensor_angle

    Each sensor reads the trail value at SENSOR_DISTANCE ahead.
    The particle rotates toward whichever sensor reads the highest value.
    """
    h = p["heading"]
    x, y = p["x"], p["y"]

    # Sample trail at each sensor position
    def sample(angle):
        sx = int(x + math.cos(angle) * SENSOR_DISTANCE) % WIDTH
        sy = int(y + math.sin(angle) * SENSOR_DISTANCE) % HEIGHT
        return trail_map[sy][sx]

    val_left = sample(h - SENSOR_ANGLE)
    val_front = sample(h)
    val_right = sample(h + SENSOR_ANGLE)

    # Decide rotation
    if val_front >= val_left and val_front >= val_right:
        pass  # keep heading, front is strongest (or tied)
    elif val_left > val_right:
        p["heading"] -= ROTATION_ANGLE  # turn left
    elif val_right > val_left:
        p["heading"] += ROTATION_ANGLE  # turn right
    else:
        # left == right and both > front: pick randomly
        p["heading"] += random.choice([-1, 1]) * ROTATION_ANGLE


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
    """Apply a 3x3 mean filter to the trail map."""
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
    for y in range(HEIGHT):
        row = ""
        for x in range(WIDTH):
            row += intensity_to_char(trail_map[y][x])
        print(row)


# Create particles and trail map
particles = [create_particle() for _ in range(NUM_PARTICLES)]
trail_map = create_trail_map()

# Main loop
for tick in range(500):
    start = time.time()

    # 1. Sense: each particle reads the trail map and adjusts heading
    for p in particles:
        sense(p, trail_map)

    # 2. Move all particles
    for p in particles:
        move(p)

    # 3. Deposit onto trail map
    deposit(particles, trail_map)

    # 4. Diffuse the trail map (blur)
    trail_map = diffuse(trail_map)

    # 5. Decay the trail map
    decay(trail_map)

    # 6. Display
    draw(trail_map)

    end = time.time()
    print(
        f"\ntick={tick}  particles={NUM_PARTICLES}  decay={DECAY_FACTOR}  time={(end - start) * 1000:.0f}ms"
    )

    time.sleep(0.05)
