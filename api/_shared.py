from functools import lru_cache
from pathlib import Path
import sys
import json
import joblib


# Ensure src/ is importable
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

try:
    from preprocess import preprocess_features, FEATURE_COLS, map_columns  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"Failed to import preprocessing utilities: {e}")


@lru_cache(maxsize=1)
def load_artifacts():
    model_path = ROOT / "model" / "model.joblib"
    scaler_path = ROOT / "model" / "scaler.joblib"
    metrics_path = ROOT / "model" / "metrics.json"

    if not model_path.exists() or not scaler_path.exists():
        return None, None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    metrics = None
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            metrics = None
    return model, scaler, metrics


__all__ = [
    "ROOT",
    "preprocess_features",
    "FEATURE_COLS",
    "map_columns",
    "load_artifacts",
]
