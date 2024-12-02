from pathlib import Path
import logging

class Config:
    # Paths
    ROOT_DIR = Path("E:\DeepScan\deepfake_cluster\data-backup")
    DATA_DIR = ROOT_DIR / "faces"
    WEIGHTS_DIR = ROOT_DIR / "pretrained-weights"
    RESULTS_DIR = ROOT_DIR / "results"
    ANNOTATIONS_DIR = ROOT_DIR / "annotations"
    
    # Create necessary directories
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "weights").mkdir(exist_ok=True)
    (RESULTS_DIR / "plots").mkdir(exist_ok=True)
    (RESULTS_DIR / "metrics").mkdir(exist_ok=True)
    ANNOTATIONS_DIR.mkdir(exist_ok=True)
    
    # Training params
    BATCH_SIZE = 24
    MAX_EPOCHS = 8
    IMAGE_SIZE = 224
    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Model configs
    DROPOUT_RATE = 0.2
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5 
    
    # Dataset configs
    DATASET_FRACTION = 0.5  # Use 50% of data
    FF_SUBSETS = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    
    # Logging config
    LOG_DIR = ROOT_DIR / "logs"
    LOG_DIR.mkdir(exist_ok=True)
    
    @staticmethod
    def setup_logging():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Config.LOG_DIR / 'training.log'),
                logging.StreamHandler()
            ]
        )