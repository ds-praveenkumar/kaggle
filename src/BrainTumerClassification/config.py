from pathlib import Path

class Config:
    data_dir = Path( Path.cwd().resolve().parents[0] / "datasets" / ""