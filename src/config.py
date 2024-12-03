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
    
    # Model names from timm - use exact names
    MODELS = {
        # 'xception': 'legacy_xception',
        # 'res2net101_26w_4s': 'res2net101_26w_4s',
        # 'tf_efficientnet_b7_ns': 'tf_efficientnet_b7_ns',
        'ensemble': 'ensemble'
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
        
        # Create complete directory structure for each dataset and model
        for dataset in ['ff++', 'celebdf']:
            # Create model-specific directories
            for model_name in cls.MODELS.values():
                for variant in ['no_aug', 'with_aug']:
                    # Weights directory
                    (cls.RESULTS_DIR / 'weights' / dataset / model_name / variant).mkdir(parents=True, exist_ok=True)
                    
                    # Metrics directory
                    (cls.RESULTS_DIR / 'metrics' / dataset / model_name / variant).mkdir(parents=True, exist_ok=True)
                    
                    # Plots directory with augmented samples subdirectory
                    plots_dir = cls.RESULTS_DIR / 'plots' / dataset / model_name / variant
                    plots_dir.mkdir(parents=True, exist_ok=True)
                    if variant == 'with_aug':
                        (plots_dir / 'augmented_samples').mkdir(exist_ok=True)
        
        # Create cross-evaluation directories
        for source_dataset in ['ff++', 'celebdf']:
            for model_name in cls.MODELS.values():
                for target_dataset in ['ff++', 'celebdf']:
                    if source_dataset != target_dataset:
                        (cls.RESULTS_DIR / 'cross_evaluation' / source_dataset / model_name / target_dataset).mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        cls.setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("Created directory structure with models:")
        for model_name in cls.MODELS.values():
            logger.info(f"- {model_name}")
    
    # Training params
    BATCH_SIZE = 32
    MAX_EPOCHS = 30
    PATIENCE = 7
    MIN_EPOCHS = 10
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
    DATASET_FRACTION = 0.5
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    IMAGE_SIZE = 224 
    
    # GPU optimization
    GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    MIXED_PRECISION = True  # Use automatic mixed precision
    NUM_WORKERS = 8  # Usually set to number of CPU cores
    
    # Augmentation configs
    AUGMENTATION_RATIO = 0.3  # 30% increase in dataset size
    AUGMENTATION_PARAMS = {
        # Geometric transformations
        'rotation': {'probability': 0.5, 'max_left': 15, 'max_right': 15},
        'shear': {'probability': 0.3, 'max_shear_left': 10, 'max_shear_right': 10},
        'flip': {'probability': 0.5},
        'skew': {'probability': 0.3, 'magnitude': 0.3},
        
        # Color transformations
        'color_jitter': {
            'probability': 0.3,
            'brightness': 0.2,  # range: [1-x, 1+x]
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1  # range: [-x, x]
        }
    }
    
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