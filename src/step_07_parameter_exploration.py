"""Step 7: Parameter exploration — see how each parameter changes the simulation.

Same algorithm as Step 6. The change is named parameter PRESETS that let you
compare different behaviors side by side. Each preset emphasizes one parameter
extreme described in the learning path's parameter table.

    python src/step_07_parameter_exploration.py              (default preset)
    python src/step_07_parameter_exploration.py tight        (narrow sensor angle)
    python src/step_07_parameter_exploration.py wide         (wide sensor angle)
    python src/step_07_parameter_exploration.py smooth       (small rotation angle)
    python src/step_07_parameter_exploration.py jagged       (large rotation angle)
    python src/step_07_parameter_exploration.py persistent   (high decay factor — trails last)
    python src/step_07_parameter_exploration.py fading       (low decay factor — trails vanish)
    python src/step_07_parameter_exploration.py dense        (many particles)
    python src/step_07_parameter_exploration.py sparse       (few particles)
    python src/step_07_parameter_exploration.py farsight     (large sensor distance)
    python src/step_07_parameter_exploration.py nearsight    (small sensor distance)
"""

import math
import os
import random
import sys
import time

# Grid dimensions
WIDTH = 80
HEIGHT = 60

# Map intensity to characters (low → high)
INTENSITY_CHARS = " .:-=+*#%@"


# --- Parameter presets ---
# Each preset is a dict of simulation parameters.
# "default" matches Step 6's original values.

PRESETS = {
    "default": {
        "num_particles": 150,
        "sensor_angle": math.radians(45),
        "sensor_distance": 3,
        "rotation_angle": math.radians(45),
        "deposit_amount": 5.0,
        "decay_factor": 0.95,
        "description": "Balanced baseline (same as Step 6)",
    },
    # --- sensor_angle ---
    "tight": {
        "num_particles": 150,
        "sensor_angle": math.radians(15),
        "sensor_distance": 3,
        "rotation_angle": math.radians(45),
        "deposit_amount": 5.0,
        "decay_factor": 0.95,
        "description": "Narrow sensor angle → tight, focused paths",
    },
    "wide": {
        "num_particles": 150,
        "sensor_angle": math.radians(80),
        "sensor_distance": 3,
        "rotation_angle": math.radians(45),
        "deposit_amount": 5.0,
        "decay_factor": 0.95,
        "description": "Wide sensor angle → broad, spread-out patterns",
    },
    # --- rotation_angle ---
    "smooth": {
        "num_particles": 150,
        "sensor_angle": math.radians(45),
        "sensor_distance": 3,
        "rotation_angle": math.radians(10),
        "deposit_amount": 5.0,
        "decay_factor": 0.95,
        "description": "Small rotation angle → gentle turns, smooth curves",
    },
    "jagged": {
        "num_particles": 150,
        "sensor_angle": math.radians(45),
        "sensor_distance": 3,
        "rotation_angle": math.radians(80),
        "deposit_amount": 5.0,
        "decay_factor": 0.95,
        "description": "Large rotation angle → sharp turns, jagged paths",
    },
    # --- decay_factor ---
    "persistent": {
        "num_particles": 150,
        "sensor_angle": math.radians(45),
        "sensor_distance": 3,
        "rotation_angle": math.radians(45),
        "deposit_amount": 5.0,
        "decay_factor": 0.99,
        "description": "High decay factor → trails persist, strong highways",
    },
    "fading": {
        "num_particles": 150,
        "sensor_angle": math.radians(45),
        "sensor_distance": 3,
        "rotation_angle": math.radians(45),
        "deposit_amount": 5.0,
        "decay_factor": 0.85,
        "description": "Low decay factor → trails vanish fast, less memory",
    },
    # --- num_particles ---
    "dense": {
        "num_particles": 400,
        "sensor_angle": math.radians(45),
        "sensor_distance": 3,
        "rotation_angle": math.radians(45),
        "deposit_amount": 5.0,
        "decay_factor": 0.95,
        "description": "Many particles → dense, robust networks",
    },
    "sparse": {
        "num_particles": 40,
        "sensor_angle": math.radians(45),
        "sensor_distance": 3,
        "rotation_angle": math.radians(45),
        "deposit_amount": 5.0,
        "decay_factor": 0.95,
        "description": "Few particles → sparse, fragile networks",
    },
    # --- sensor_distance ---
    "farsight": {
        "num_particles": 150,
        "sensor_angle": math.radians(45),
        "sensor_distance": 9,
        "rotation_angle": math.radians(45),
        "deposit_amount": 5.0,
        "decay_factor": 0.95,
        "description": "Large sensor distance → far sensing, sparse networks",
    },
    "nearsight": {
        "num_particles": 150,
        "sensor_angle": math.radians(45),
        "sensor_distance": 1,
        "rotation_angle": math.radians(45),
        "deposit_amount": 5.0,
        "decay_factor": 0.95,
        "description": "Small sensor distance → local sensing, dense networks",
    },
}


# --- Simulation functions (unchanged from step 6) ---


def create_trail_map():
    return [[0.0] * WIDTH for _ in range(HEIGHT)]


def sense(p, trail_map, sensor_angle, sensor_distance, rotation_angle):
    h = p["heading"]
    x, y = p["x"], p["y"]

    def sample(angle):
        sx = int(x + math.cos(angle) * sensor_distance) % WIDTH
        sy = int(y + math.sin(angle) * sensor_distance) % HEIGHT
        return trail_map[sy][sx]

    val_left = sample(h - sensor_angle)
    val_front = sample(h)
    val_right = sample(h + sensor_angle)

    if val_front >= val_left and val_front >= val_right:
        pass
    elif val_left > val_right:
        p["heading"] -= rotation_angle
    elif val_right > val_left:
        p["heading"] += rotation_angle
    else:
        p["heading"] += random.choice([-1, 1]) * rotation_angle


def move(p):
    p["x"] = (p["x"] + math.cos(p["heading"])) % WIDTH
    p["y"] = (p["y"] + math.sin(p["heading"])) % HEIGHT


def deposit(particles, trail_map, deposit_amount):
    for p in particles:
        gx, gy = int(p["x"]) % WIDTH, int(p["y"]) % HEIGHT
        trail_map[gy][gx] += deposit_amount


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


def decay(trail_map, decay_factor):
    for y in range(HEIGHT):
        for x in range(WIDTH):
            trail_map[y][x] *= decay_factor


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

preset_name = sys.argv[1] if len(sys.argv) > 1 else "default"
if preset_name not in PRESETS:
    print(f"Unknown preset '{preset_name}'. Available presets:")
    for name, cfg in PRESETS.items():
        print(f"  {name:12s} — {cfg['description']}")
    sys.exit(1)

cfg = PRESETS[preset_name]

# Spawn particles randomly (focus is on parameter effects, not spawn modes)
particles = [
    {
        "x": random.uniform(0, WIDTH),
        "y": random.uniform(0, HEIGHT),
        "heading": random.uniform(0, 2 * math.pi),
    }
    for _ in range(cfg["num_particles"])
]
trail_map = create_trail_map()

for tick in range(500):
    start = time.time()

    for p in particles:
        sense(
            p,
            trail_map,
            cfg["sensor_angle"],
            cfg["sensor_distance"],
            cfg["rotation_angle"],
        )
    for p in particles:
        move(p)
    deposit(particles, trail_map, cfg["deposit_amount"])
    trail_map = diffuse(trail_map)
    decay(trail_map, cfg["decay_factor"])
    draw(trail_map)

    end = time.time()
    # Parameter dashboard
    print(f"\n[{preset_name}] {cfg['description']}")
    print(
        f"tick={tick}  particles={cfg['num_particles']}  "
        f"sensor_angle={math.degrees(cfg['sensor_angle']):.0f}deg  "
        f"sensor_dist={cfg['sensor_distance']}  "
        f"rotation={math.degrees(cfg['rotation_angle']):.0f}deg  "
        f"decay={cfg['decay_factor']}  "
        f"deposit={cfg['deposit_amount']}  "
        f"time={int((end - start) * 1000)}ms"
    )

    time.sleep(0.05)
