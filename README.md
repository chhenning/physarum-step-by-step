# Physarum Step by Step

Physarum slime mold simulation in Python — built step by step from scratch, from a single moving particle to emergent network behavior.

The inspiration came from `Michael Fogleman`'s [repo](https://github.com/fogleman/physarum). Here the algorithm is implemented as an offline renderer in Go.
It takes about 13 secs to generate a 1024x1024 png.

This implementation tries to make it realtime using Python and numpy. Here is what it looks like on a low-resolution and frame rate:

![Slime Mold Algo in Action](output.gif)

