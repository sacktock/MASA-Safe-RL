"""ProbShield diagnostic plot specs.

Importing this package registers ``margin``, ``margin_dist``, and ``betas``
with the global PlotSpec registry. Use ``--specs-from masa.plotting.examples.probshield``
on the CLI to opt in.
"""

from . import margin       # noqa: F401
from . import margin_dist  # noqa: F401
from . import betas        # noqa: F401
