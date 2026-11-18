
# ─────────────────────────────────────────────────────────────
# 2.  CONFIGURATION
# ─────────────────────────────────────────────────────────────

import os
import warnings
warnings.filterwarnings("ignore")

START_DATE = "2015-01-01"   # ~130 months; post-MSCI upgrades UAE/Qatar, post-Saudi opening
END_DATE   = "2026-05-16"

KUWAIT_MSCI_UPGRADE = "2020-06-01"

# Liquidity filters
MIN_MONTHS_DATA = 24        # 2+ years of history required
MIN_AVG_VOLUME  = 5_000     # Kuwait has lower volumes; conservative threshold
MIN_PRICE       = 0.10      # exclude suspended / penny names

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)
