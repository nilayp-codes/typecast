"""Test setup: keep LiteLLM offline (bundled price map, no fetch from GitHub at import)."""

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
