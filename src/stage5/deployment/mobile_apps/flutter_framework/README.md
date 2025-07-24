# AWARE-NET Flutter Mobile Application

Cross-platform mobile application for real-time deepfake detection using the AWARE-NET cascade system.

## Features

- **Real-time Detection**: Live camera feed analysis with instant results
- **Batch Processing**: Multiple image/video analysis with progress tracking
- **Performance Monitoring**: Real-time inference speed and accuracy metrics
- **Offline Capability**: On-device processing with ONNX models
- **Cross-Platform**: Single codebase for iOS and Android
- **Material Design**: Beautiful, intuitive user interface

## Architecture

```
lib/
├── main.dart                 # Application entry point
├── models/
│   ├── detection_result.dart # Detection result data model
│   ├── cascade_config.dart   # Cascade configuration model
│   └── performance_metrics.dart # Performance monitoring model
├── services/
│   ├── detection_service.dart # Core detection service
│   ├── model_loader.dart     # ONNX model loading
│   ├── camera_service.dart   # Camera integration
│   └── analytics_service.dart # Usage analytics
├── screens/
│   ├── home_screen.dart      # Main application screen
│   ├── camera_screen.dart    # Real-time detection screen
│   ├── gallery_screen.dart   # Batch processing screen
│   └── settings_screen.dart  # Configuration screen
├── widgets/
│   ├── detection_display.dart # Result visualization
│   ├── performance_chart.dart # Real-time metrics
│   └── progress_indicator.dart # Processing progress
└── utils/
    ├── image_processor.dart  # Image preprocessing
    ├── video_processor.dart  # Video frame extraction
    └── model_utils.dart      # ONNX model utilities
```

## Installation

### Prerequisites
- Flutter SDK (>=3.0.0)
- Dart SDK (>=3.0.0)
- Android Studio / Xcode for platform-specific builds
- ONNX Runtime Flutter plugin

### Setup
```bash
# Install dependencies
flutter pub get

# Generate platform-specific code
flutter packages pub run build_runner build

# Run on device/simulator
flutter run
```

## Usage

### Real-time Detection
```dart
import 'package:aware_net/services/detection_service.dart';

final detectionService = DetectionService();
await detectionService.initialize();

// Process camera frame
final result = await detectionService.detectFromImage(cameraImage);
print('Prediction: ${result.prediction}, Confidence: ${result.confidence}');
```

### Batch Processing
```dart
// Process multiple images
final images = await picker.pickMultiImage();
final results = await detectionService.processBatch(images);

for (final result in results) {
  print('${result.filename}: ${result.prediction} (${result.confidence:.3f})');
}
```

## Configuration

### Model Configuration
```yaml
# pubspec.yaml
flutter:
  assets:
    - assets/models/aware_net_mobile.onnx
    - assets/models/cascade_config.json
    - assets/config/app_config.json
```

### Performance Settings
- **Inference Mode**: CPU/GPU/Neural Engine
- **Batch Size**: 1-8 images per batch
- **Quality Settings**: High/Medium/Low processing quality
- **Cache Settings**: Model caching and result persistence

## Deployment

### Android
```bash
# Build APK
flutter build apk --release

# Build App Bundle
flutter build appbundle --release
```

### iOS
```bash
# Build iOS app
flutter build ios --release

# Archive for App Store
flutter build ipa --release
```

## Performance Targets

- **Startup Time**: <3 seconds from launch to ready
- **Inference Speed**: <100ms per image on mid-range devices
- **Memory Usage**: <200MB total application memory
- **Battery Efficiency**: <5% battery drain per 100 detections
- **Model Size**: <25MB total ONNX models

## Security & Privacy

- **On-Device Processing**: All detection happens locally
- **No Data Upload**: Images/videos never leave the device
- **Secure Storage**: Models and configurations encrypted at rest
- **Permission Management**: Minimal required permissions
- **Privacy Compliance**: GDPR/CCPA compliant by design