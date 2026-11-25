"""DualRng: protocol-driven psi experiments on two hardware random number generators.

Acquisition (recorder), protocol (schedule), confirmatory analysis (one
pre-specified statistic per session type) and exploratory analysis (scan) are
kept in separate modules so that nothing downstream can stall the recorder and
nothing exploratory can leak into the ledger.
"""

# Bumped whenever the confirmatory analysis changes in a way that could change
# a session's result. Part of every session's config hash.
ANALYSIS_VERSION = "2.0.0"
