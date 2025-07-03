from pathlib import Path
import logging
import torch


class BaseConfig:
    """
    基础配置类，包含所有实验共享的默认参数。
    """

    # Paths
    ROOT_DIR = Path("/workspace/AWARE-NET")
    DATA_DIR = ROOT_DIR / "faces"
    RESULTS_DIR = ROOT_DIR / "results"
    WEIGHTS_DIR = RESULTS_DIR / "weights"
    ANNOTATIONS_DIR = ROOT_DIR / "annotations"
    LOG_DIR = ROOT_DIR / "logs"

    # Dataset structure
    DATASET_STRUCTURE = {
        "ff++": {
            "real_dirs": {"actors": "ff++/real/actors", "youtube": "ff++/real/youtube"},
            "fake_dirs": {
                "Deepfakes": "ff++/manipulated/Deepfakes",
                "Face2Face": "ff++/manipulated/Face2Face",
                "FaceSwap": "ff++/manipulated/FaceSwap",
                "NeuralTextures": "ff++/manipulated/NeuralTextures",
                "FaceShifter": "ff++/manipulated/FaceShifter",
                "DeepFakeDetection": "ff++/manipulated/DeepFakeDetection",
            },
        },
        "celebdf": {
            "real_dirs": {
                "celeb-real": "celebdf/celeb-real",
                "youtube-real": "celebdf/youtube-real",
            },
            "fake_dirs": {"celeb-synthesis": "celebdf/celeb-synthesis"},
        },
    }

    # Model names mapping
    MODELS = {
        "xception": {
            "timm_name": "legacy_xception",
            "dir_name": "xception",
            "variant_key": "legacy_xception",
            "weights_dir": "xception",
        },
        "res2net101_26w_4s": {
            "timm_name": "res2net101_26w_4s",
            "dir_name": "res2net101",
            "variant_key": "res2net101_26w_4s",
            "weights_dir": "res2net101",
        },
        "tf_efficientnet_b7_ns": {
            "timm_name": "tf_efficientnet_b7_ns",
            "dir_name": "efficientnet_b7",
            "variant_key": "tf_efficientnet_b7_ns",
            "weights_dir": "efficientnet_b7",
        },
        "ensemble": {
            "timm_name": "ensemble",
            "dir_name": "ensemble",
            "variant_key": "ensemble",
            "weights_dir": "ensemble",
        },
    }

    # Default Training params - can be overridden by subclasses
    BATCH_SIZE = 32
    MAX_EPOCHS = 15
    PATIENCE = 5
    MIN_EPOCHS = 10
    VALIDATION_FREQ = 1
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    LR_SCHEDULE_TYPE = "cosine"
    WARMUP_EPOCHS = 2
    LR_MIN = 1e-6
    DROPOUT_RATE = 0.3

    # Dataset configs
    DATASET_FRACTION = 1.0  # Use 100% of the data by default
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    IMAGE_SIZE = 224

    # GPU optimization
    GRADIENT_ACCUMULATION_STEPS = 2
    MIXED_PRECISION = True
    NUM_WORKERS = 8

    # Augmentation configs
    AUGMENTATION_RATIO = 0.3
    AUGMENTATION_PARAMS = {
        "rotation": {"probability": 0.5, "max_left": 15, "max_right": 15},
        "shear": {"probability": 0.3, "max_shear_left": 10, "max_shear_right": 10},
        "flip": {"probability": 0.5},
        "skew": {"probability": 0.3, "magnitude": 0.3},
        "color_jitter": {
            "probability": 0.3,
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1,
        },
            # 添加噪声和模糊等额外增强
    'noise': {'probability': 0.3, 'std': 0.02},
    'blur': {'probability': 0.3, 'kernel_size': 3}
    }

    # Annotation configs
    FORCE_NEW_ANNOTATIONS = False
    ANNOTATION_CACHE = True

    @classmethod
    def get_model_weights_dir(cls, model_key, dataset, variant="no_aug"):
        if model_key not in cls.MODELS:
            raise ValueError(f"Invalid model key: {model_key}")
        return cls.WEIGHTS_DIR / dataset / cls.MODELS[model_key]["weights_dir"] / variant

    @classmethod
    def get_latest_weights_path(cls, model_key, dataset, variant="no_aug"):
        weights_dir = cls.get_model_weights_dir(model_key, dataset, variant)
        if not weights_dir.exists():
            return None
        weight_files = list(weights_dir.glob("*.pth"))
        if not weight_files:
            return None
        return max(weight_files, key=lambda p: p.stat().st_mtime)

    @classmethod
    def create_directories(cls):
        cls.ROOT_DIR.mkdir(exist_ok=True)
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.RESULTS_DIR.mkdir(exist_ok=True)
        cls.ANNOTATIONS_DIR.mkdir(exist_ok=True)
        cls.LOG_DIR.mkdir(exist_ok=True)
        cls.setup_logging()
        logger = logging.getLogger(__name__)
        for dataset in ["ff++", "celebdf"]:
            for model_config in cls.MODELS.values():
                for variant in ["no_aug", "with_aug"]:
                    (cls.WEIGHTS_DIR / dataset / model_config["weights_dir"] / variant).mkdir(parents=True, exist_ok=True)
                    (cls.RESULTS_DIR / "metrics" / dataset / model_config["dir_name"] / variant).mkdir(parents=True, exist_ok=True)
                    plots_dir = (cls.RESULTS_DIR / "plots" / dataset / model_config["dir_name"] / variant)
                    plots_dir.mkdir(parents=True, exist_ok=True)
                    if variant == "with_aug":
                        (plots_dir / "augmented_samples").mkdir(exist_ok=True)
        for source_dataset in ["ff++", "celebdf"]:
            for model_config in cls.MODELS.values():
                for target_dataset in ["ff++", "celebdf"]:
                    if source_dataset != target_dataset:
                        (cls.RESULTS_DIR / "cross_evaluation" / source_dataset / model_config["dir_name"] / target_dataset).mkdir(parents=True, exist_ok=True)
        logger.info("Created directory structure.")

    @staticmethod
    def setup_logging():
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        if not root_logger.handlers:
            console_handler = logging.StreamHandler()
            console_format = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
            console_handler.setFormatter(console_format)
            file_handler = logging.FileHandler(BaseConfig.LOG_DIR / "training.log")
            file_handler.setFormatter(console_format)
            root_logger.addHandler(console_handler)
            root_logger.addHandler(file_handler)


class FFppConfig(BaseConfig):
    """Hyperparameters specific to the FF++ dataset."""
    LEARNING_RATE = 1.5e-4
    BATCH_SIZE = 48
    MAX_EPOCHS = 20
    PATIENCE = 4
    DROPOUT_RATE = 0.4
    AUGMENTATION_PARAMS = {
        **BaseConfig.AUGMENTATION_PARAMS,
        "blur": {"probability": 0.4, "min_factor": 1, "max_factor": 2},
        "jpeg_compression": {"probability": 0.4, "min_quality": 75, "max_quality": 95},
    }


class CelebDFConfig(BaseConfig):
    """Hyperparameters specific to the Celeb-DF dataset."""
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 32
    MAX_EPOCHS = 12
    PATIENCE = 3
    DROPOUT_RATE = 0.5
    LR_SCHEDULE_TYPE = "step"


class EnsembleFinetuneConfig(BaseConfig):
    """Hyperparameters for fine-tuning the ensemble model."""
    LEARNING_RATE = 5e-5
    BATCH_SIZE = 16
    MAX_EPOCHS = 10
    PATIENCE = 2


class FastDebugConfig(BaseConfig):
    """A minimal configuration for fast debugging and testing."""
    DATASET_FRACTION = 0.05
    MAX_EPOCHS = 2
    BATCH_SIZE = 8
    NUM_WORKERS = 2


def get_config(name="default"):
    """Factory function to get the desired configuration instance."""
    configs = {
        "default": BaseConfig,
        "ff++": FFppConfig,
        "celebdf": CelebDFConfig,
        "ensemble": EnsembleFinetuneConfig,
        "debug": FastDebugConfig,
    }
    config_class = configs.get(name.lower())
    if config_class is None:
        raise ValueError(f"Unknown configuration name: {name}")
    return config_class()