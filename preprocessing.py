import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.hub")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from retinaface import RetinaFace
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)

class DatasetPreprocessor:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        
        # Dataset paths
        self.videos_dir = self.base_dir / 'videos'
        self.frames_dir = self.base_dir / 'frames'
        self.faces_dir = self.base_dir / 'faces'
        
        # Parameters
        self.image_size = 256
        self.padding = 0.25
        self.max_faces_real = 32
        self.max_faces_fake = 16
        
        # FF++ subsets
        self.ffpp_manipulated_subsets = [
            'Deepfakes', 'Face2Face', 'FaceSwap',
            'NeuralTextures', 'FaceShifter', 'DeepFakeDetection'
        ]
        self.compression = 'c40'
        
        # Initialize face detector
        if torch.cuda.is_available():
            logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.set_device(0)
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
        else:
            logging.info("Using CPU")
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.face_detector = RetinaFace

    def calculate_sampling_interval(self, total_frames, fps, is_real=False):
        """Calculate smart sampling interval based on video properties"""
        video_duration = total_frames / fps  # duration in seconds
        max_faces = self.max_faces_real if is_real else self.max_faces_fake
        
        if video_duration < 5:  # Very short video
            return 1  # Sample every frame
        
        # Base interval on video duration
        if is_real:
            # For real videos, sample more uniformly
            base_interval = max(1, int(total_frames / (max_faces * 2)))  # More dense sampling
            # Ensure at least 2 frames per second
            return min(base_interval, fps // 2)
        else:
            # For fake videos, focus on middle portions
            base_interval = max(1, int(total_frames / (max_faces * 1.2)))  # Less dense sampling
            # Ensure at least 1 frame per second
            return min(base_interval, fps)

    def process_video(self, video_path, faces_dir, is_real=False):
        """Process single video and extract faces"""
        try:
            logging.info(f"\nProcessing: {video_path.name}")
            
            # Open video
            video = cv2.VideoCapture(str(video_path))
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(video.get(cv2.CAP_PROP_FPS))
            
            # Get face limit and sampling interval
            max_faces = self.max_faces_real if is_real else self.max_faces_fake
            interval = self.calculate_sampling_interval(total_frames, fps, is_real)
            
            # For longer videos, focus on middle portion
            if total_frames > fps * 10:  # videos longer than 10 seconds
                start_frame = int(total_frames * 0.2)  # Skip first 20%
                end_frame = int(total_frames * 0.8)    # Skip last 20%
                video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                total_frames = end_frame - start_frame
            
            logging.info(f"Video sampling - Duration: {total_frames/fps:.1f}s, "
                        f"FPS: {fps}, Interval: {interval}, "
                        f"Target faces: {max_faces}, "
                        f"Estimated samples: {total_frames/interval:.0f}")
            
            # Create output directory
            faces_dir.mkdir(parents=True, exist_ok=True)
            
            frame_count = 0
            saved_count = 0
            faces_detected = 0
            
            # Process frames
            while video.isOpened() and saved_count < max_faces:
                ret, frame = video.read()
                if not ret:
                    break
                    
                if frame_count % interval == 0:
                    # Convert frame to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces
                    faces = self.face_detector.detect_faces(frame_rgb)
                    # Convert face dictionary to list format that the code expects
                    face_list = []
                    for face_key in faces:  # face_key will be 'face_1', 'face_2', etc.
                        face_dict = faces[face_key]
                        # Create a new dict with the expected structure
                        face_info = {
                            'score': face_dict['score'],
                            'bbox': face_dict['facial_area']  # Map facial_area to bbox
                        }
                        face_list.append(face_info)

                    faces_detected += len(face_list)
                    
                    if face_list:  # Now use face_list instead of faces
                        # Get best face
                        best_face = max(face_list, key=lambda x: 
                            (x['score'], (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))
                        )
                        
                        # Extract face with padding
                        x0, y0, x1, y1 = best_face['bbox']
                        w, h = x1 - x0, y1 - y0
                        
                        x0 = max(0, x0 - int(w * self.padding))
                        x1 = min(frame.shape[1], x1 + int(w * self.padding))
                        y0 = max(0, y0 - int(h * self.padding))
                        y1 = min(frame.shape[0], y1 + int(h * self.padding))
                        
                        face_img = frame[int(y0):int(y1), int(x0):int(x1)]
                        
                        # Save face if large enough
                        if min(face_img.shape[:2]) >= 64:
                            face_resized = cv2.resize(face_img, (self.image_size, self.image_size))
                            face_name = f"{video_path.stem}_{saved_count:04d}_face.png"
                            cv2.imwrite(str(faces_dir / face_name), face_resized)
                            saved_count += 1
                            
                            # if saved_count % 5 == 0:
                            #     logging.info(f"Saved {saved_count}/{max_faces} faces")
                
                frame_count += 1
            
            video.release()
            logging.info(f"Completed {video_path.name}: Found {faces_detected} faces, "
                        f"Saved {saved_count}/{max_faces} faces")
            return saved_count
            
        except Exception as e:
            logging.error(f"Error processing {video_path.name}: {str(e)}")
            return 0

    def process_dataset(self, dataset_type):
        """Process dataset sequentially"""
        if dataset_type.lower() == 'ffpp':
            # Process real videos
            for real_subset in ['youtube', 'actors']:
                videos_dir = self.videos_dir / 'FF++' / 'original_sequences' / real_subset / self.compression / 'videos'
                faces_dir = self.faces_dir / 'ff++' / 'real' / real_subset
                logging.info(f"Processing {videos_dir} and {faces_dir}")

                videos = list(videos_dir.glob('*.mp4'))
                logging.info(f"\nProcessing {len(videos)} real videos from {real_subset}")
                
                total_saved = 0
                for video in tqdm(videos, desc=f"Processing {real_subset} videos"):
                    total_saved += self.process_video(video, faces_dir, is_real=True)
                
                logging.info(f"Completed real_subset {real_subset}: Total faces saved: {total_saved}")
            
            # Process fake videos
            for subset in self.ffpp_manipulated_subsets:
                videos_dir = self.videos_dir / 'FF++' / 'manipulated_sequences' / subset / self.compression / 'videos'
                faces_dir = self.faces_dir / 'ff++' / 'manipulated' / subset
                
                videos = list(videos_dir.glob('*.mp4'))
                logging.info(f"\nProcessing {len(videos)} fake videos from {subset}")
                
                total_saved = 0
                for video in tqdm(videos, desc=f"Processing {subset} videos"):
                    total_saved += self.process_video(video, faces_dir, is_real=False)
                
                logging.info(f"Completed fake_subset {subset}: Total faces saved: {total_saved}")
            
        elif dataset_type.lower() == 'celebdf':
            # Define CelebDF subsets and their properties
            celebdf_subsets = {
                'Celeb-real': {
                    'path': self.videos_dir / 'CelebDF-v2' / 'Celeb-real',
                    'is_real': True
                },
                'YouTube-real': {
                    'path': self.videos_dir / 'CelebDF-v2' / 'YouTube-real',
                    'is_real': True
                },
                'Celeb-synthesis': {
                    'path': self.videos_dir / 'CelebDF-v2' / 'Celeb-synthesis',
                    'is_real': False
                }
            }
            
            # Process each subset
            for subset_name, subset_info in celebdf_subsets.items():
                videos_dir = subset_info['path']
                faces_dir = self.faces_dir / 'celebdf' / subset_name.lower()
                is_real = subset_info['is_real']
                
                videos = list(videos_dir.glob('*.mp4'))
                logging.info(f"\nProcessing {len(videos)} videos from CelebDF {subset_name}")
                
                total_saved = 0
                for video in tqdm(videos, desc=f"Processing CelebDF {subset_name}"):
                    total_saved += self.process_video(video, faces_dir, is_real=is_real)
                
                logging.info(f"Completed CelebDF {subset_name}: Total faces saved: {total_saved}")

if __name__ == "__main__":
    preprocessor = DatasetPreprocessor("/root/autodl-tmp")
    
    # Process both datasets
    preprocessor.process_dataset('ffpp')
    preprocessor.process_dataset('celebdf')