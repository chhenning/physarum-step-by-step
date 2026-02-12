# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Physarum slime mold simulation — a step-by-step educational project demonstrating emergent behavior through agent-based modeling. Pure Python with terminal ASCII visualization. Each `src/step_XX_*.py` file builds incrementally on the previous one.

## Setup and Running

```bash
# Setup environment
source scripts/setup.sh   # activates .venv and sets PYTHONPATH=.

# Run any step directly
python src/step_01_single_particle.py
python src/step_06_emergence.py [random|ring|center|clusters]
```

Uses **uv** for dependency management. Python >=3.13 required. Single dependency: pygame (for future visualization).

## Architecture

**Incremental step progression:**
1. `step_01` — Single particle on a grid with heading-based movement
2. `step_02` — Multiple particles with random initialization
3. `step_03` — Trail map: deposition and exponential decay
4. `step_04` — Diffusion via 3x3 mean filter (double-buffered)
5. `step_05` — Sensors and chemotaxis (sense → rotate → move feedback loop)
6. `step_06` — Emergence with configurable spawn modes

**Each step file is self-contained** with this structure: constants → helper functions → main loop → ASCII rendering.

**Core algorithm (steps 5-6):** sense 3 directions → rotate toward strongest trail → move forward → deposit pheromone → diffuse → decay → render.

**Data structures:**
- Particles: dicts with `x`, `y`, `heading` (radians)
- Trail map: list-of-lists indexed as `trail_map[y][x]`
- Toroidal wrapping via modulo on all positions

## Conventions

- No test suite or linter configured; VS Code uses Black formatter
- `main.py` is a placeholder — run individual step files directly
- `learning_path.md` contains the full tutorial guide and parameter tuning reference
