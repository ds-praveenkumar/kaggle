from pathlib import Path

class Config:
    ROOT = Path(Path().cwd()).resolve()
    DATASET = Path(ROOT) / "datasets" / "products"
    TRAIN_DATASET = Path(DATASET) / "train"
    VALIDATION_DATASET = Path(DATASET) / "test"
    TEST_DATASET = Path(ROOT) / "images"
