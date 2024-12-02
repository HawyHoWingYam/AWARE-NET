from pathlib import Path
import logging
import torch

class Config:
    # Paths
    ROOT_DIR = Path("E:/DeepScan/deepfake_cluster/dlib-face")
    DATA_DIR = ROOT_DIR / "faces"
    RESULTS_DIR = ROOT_DIR / "results"
    WEIGHTS_DIR = RESULTS_DIR / "weights"
    ANNOTATIONS_DIR = ROOT_DIR / "annotations"
    LOG_DIR = ROOT_DIR / "logs"
    
    # Dataset structure
    DATASET_STRUCTURE = {
        'ff++': {
            'real_dirs': {
                'actors': 'ff++/real/actors',
                'youtube': 'ff++/real/youtube'
            },
            'fake_dirs': {
                'Deepfakes': 'ff++/fake/Deepfakes',
                'Face2Face': 'ff++/fake/Face2Face',
                'FaceSwap': 'ff++/fake/FaceSwap',
                'NeuralTextures': 'ff++/fake/NeuralTextures',
                'FaceShifter': 'ff++/fake/FaceShifter',
                'DeepFakeDetection': 'ff++/fake/DeepFakeDetection'
            }
        },
        'celebdf': {
            'real_dir': 'celebdf/real',
            'fake_dir': 'celebdf/fake'
        }
    }
    
    # Create all necessary directories recursively
    @classmethod
    def create_directories(cls):
        # Main directories
        cls.ROOT_DIR.mkdir(exist_ok=True)
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.RESULTS_DIR.mkdir(exist_ok=True)
        cls.ANNOTATIONS_DIR.mkdir(exist_ok=True)
        cls.LOG_DIR.mkdir(exist_ok=True)
        
        # Use exact model names from timm
        models = [
            'xception',
            'res2net101_26w_4s',
            'tf_efficientnet_b7_ns',
            'ensemble'
        ]
        
        # Create directories for each model
        for dataset in ['ff++', 'celebdf']:
            for model in models:
                for variant in ['no_aug', 'with_aug']:
                    (cls.RESULTS_DIR / "weights" / dataset / model / variant).mkdir(parents=True, exist_ok=True)
                    (cls.RESULTS_DIR / "plots" / dataset / model / variant).mkdir(parents=True, exist_ok=True)
                    (cls.RESULTS_DIR / "metrics" / dataset / model / variant).mkdir(parents=True, exist_ok=True)
        
        # Setup logging first
        cls.setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("Created directory structure with model names:")
        for model in models:
            logger.info(f"- {model}")
    
    # Training params
    BATCH_SIZE = 32
    MAX_EPOCHS = 5
    PATIENCE = 5
    MIN_EPOCHS = 1
    VALIDATION_FREQ = 1
    
    # Early stopping controls
    EARLY_STOPPING = True
    MIN_DELTA = 0.001
    
    # Validation controls
    VALIDATION_METRIC = 'loss'
    VAL_CHECK_INTERVAL = 100
    
    # Learning rate scheduling
    LR_INITIAL = 1e-4
    LR_MIN = 1e-6
    WARMUP_EPOCHS = 3
    LR_SCHEDULE_TYPE = 'cosine'
    
    # Model configs
    DROPOUT_RATE = 0.2
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Dataset configs
    DATASET_FRACTION = 0.1
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    IMAGE_SIZE = 224 
    
    # GPU optimization
    GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    MIXED_PRECISION = True  # Use automatic mixed precision
    NUM_WORKERS = 8  # Usually set to number of CPU cores
    
    @classmethod
    def optimize_gpu(cls):
        if torch.cuda.is_available():
            # Set GPU memory allocation
            torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
            torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere GPUs
            torch.backends.cudnn.allow_tf32 = True
    
    @staticmethod
    def setup_logging():
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Config.LOG_DIR / 'training.log'),
                logging.StreamHandler()
            ]
        )
        # Add file handler for debug logs
        debug_handler = logging.FileHandler(Config.LOG_DIR / 'debug.log')
        debug_handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(debug_handler)