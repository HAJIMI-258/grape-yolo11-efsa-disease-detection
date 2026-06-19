from __future__ import annotations

import shutil

from ultralytics.utils import LOGGER


class AP50CheckpointMixin:
    """Save an extra checkpoint selected by validation AP50 without changing Ultralytics fitness."""

    def save_model(self):
        saved = super().save_model()
        if not saved:
            return saved

        ap50 = float(self.metrics.get("metrics/mAP50(B)", -1.0))
        best_ap50 = float(getattr(self, "best_ap50", -1.0))
        if ap50 >= best_ap50:
            self.best_ap50 = ap50
            target = self.wdir / "best_map50.pt"
            shutil.copy2(self.last, target)
            LOGGER.info("Saved AP50-best checkpoint: %s (AP50=%.5f)", target, ap50)
        return saved
