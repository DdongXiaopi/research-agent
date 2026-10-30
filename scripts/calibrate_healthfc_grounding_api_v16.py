"""Compatibility entry point for the generic HealthFC API calibrator.

The v16 filename is retained for existing operators, but provider/model
selection is no longer embedded here. Use ``--config`` (or the generic
runner's default config) to select any HTTPS OpenAI-compatible model. Historical
v16 artifacts retain their own hashes and are not replayable through this
compatibility entry point.
"""

from __future__ import annotations

from scripts.calibrate_healthfc_grounding_api import main


if __name__ == "__main__":
    raise SystemExit(main())
