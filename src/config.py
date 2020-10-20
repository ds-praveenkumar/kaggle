from pathlib import Path

class Config:
    ROOT = Path(Path().cwd()).resolve()
    DATASET = Path(ROOT) / "datasets" / "products"
    CHAR74K = Path(ROOT) / "datasets" / 
    TRAIN_DATASET = Path(DATASET) / "train"
    VALIDATION_DATASET = Path(DATASET) / "test"
    TEST_DATASET = Path(ROOT) / "images" / "English" / "Fnt"

    MODELS_ROOT =  Path(ROOT) / "models" 
    MODELS = Path(ROOT) / "models" / "toy_clf_densenet161"
    
