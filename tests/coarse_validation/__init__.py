"""Coarse validation checks for early worm-model sanity.

These tests are intentionally lightweight and represent the first engineering
validation layer for Stage 1. They check that the worm model has the correct
first-order directionality:

- attractive odor activates AWA/AWC
- repellent odor activates AWB/ADL
- downstream output produces the expected movement direction

These are sanity checks, not final biological validation against published
calcium-imaging or receptor-level datasets.
"""
