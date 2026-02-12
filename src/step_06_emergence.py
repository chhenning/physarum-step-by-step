"""Step 6: Emergence — experiment with different starting configurations.

Same algorithm as Step 5. The only change is HOW particles are spawned.
Run with a mode argument:

    python step_06_emergence.py random     (default — random scatter)
    python step_06_emergence.py ring       (circle facing inward)
    python step_06_emergence.py center     (blob in the middle)
    python step_06_emergence.py clusters   (two groups that find each other)
"""

import math
import os
import random
import sys
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
SENSOR_ANGLE = math.radians(45)
SENSOR_DISTANCE = 3
ROTATION_ANGLE = math.radians(45)

# Map intensity to characters (low → high)
INTENSITY_CHARS = " .:-=+*#%@"


# --- Spawn modes ---


def spawn_random():
    """Random positions, random headings."""
    return [
        {
            "x": random.uniform(0, WIDTH),
            "y": random.uniform(0, HEIGHT),
            "heading": random.uniform(0, 2 * math.pi),
        }
        for _ in range(NUM_PARTICLES)
    ]


def spawn_ring():
    """Particles placed in a circle, all facing inward toward the center."""
    cx, cy = WIDTH / 2, HEIGHT / 2
    radius = min(WIDTH, HEIGHT) * 0.35
    particles = []
    for i in range(NUM_PARTICLES):
        angle = 2 * math.pi * i / NUM_PARTICLES
        x = cx + math.cos(angle) * radius
        y = cy + math.sin(angle) * radius
        heading = angle + math.pi  # face inward
        particles.append({"x": x, "y": y, "heading": heading})
    return particles


def spawn_center():
    """All particles clumped in the center with random headings."""
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
    """Two separate groups — watch them find each other."""
    particles = []
    # Left cluster
    for _ in range(NUM_PARTICLES // 2):
        particles.append(
            {
                "x": WIDTH * 0.25 + random.gauss(0, 2),
                "y": HEIGHT / 2 + random.gauss(0, 2),
                "heading": random.uniform(0, 2 * math.pi),
            }
        )
    # Right cluster
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


# --- Simulation functions (unchanged from step 5) ---


def create_trail_map():
    return [[0.0] * WIDTH for _ in range(HEIGHT)]


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
    for y in range(HEIGHT):
        for x in range(WIDTH):
            trail_map[y][x] *= DECAY_FACTOR


def intensity_to_char(value):
    index = int(value)
    index = min(index, len(INTENSITY_CHARS) - 1)
    return INTENSITY_CHARS[index]


def draw(trail_map):
    os.system("clear")
    for y in range(HEIGHT):
        row = ""
        for x in range(WIDTH):
            row += intensity_to_char(trail_map[y][x])
        print(row)


# --- Main ---

mode = sys.argv[1] if len(sys.argv) > 1 else "random"
if mode not in MODES:
    print(f"Unknown mode '{mode}'. Choose from: {', '.join(MODES)}")
    sys.exit(1)

particles = MODES[mode]()
trail_map = create_trail_map()

for tick in range(500):
    start = time.time()

    for p in particles:
        sense(p, trail_map)
    for p in particles:
        move(p)
    deposit(particles, trail_map)
    trail_map = diffuse(trail_map)
    decay(trail_map)
    draw(trail_map)

    end = time.time()
    print(
        f"\ntick={tick}  mode={mode}  particles={NUM_PARTICLES}  time={(end - start) * 1000:.0f}ms"
    )

    time.sleep(0.05)
