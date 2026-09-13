"""Compare TimeF against what a dataset already ships, under the library a user already has.

Holds each dataset in two representations, reads both with both readers over four tasks, and
measures the size of each representation on disk. Runs on demand. It is never part of
continuous integration, because timings from two machines are not comparable.
"""

# This docstring is the governing text of ADR-0016. `tests/test_ci_boundary.py` is the mechanical
# half: it audits the workflows and the whole suite for the properties that would break if a real
# benchmark ever ran on a runner.
