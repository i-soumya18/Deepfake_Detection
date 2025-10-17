# Deepfake Detection - Robust Preprocessing Pipeline
# Following SOTA Model Architecture Guidelines

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from facenet_pytorch import MTCNN
import json
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class FaceDetector:
    """
    Advanced face detection and extraction using MTCNN
    Following the architecture guidelines for face-centric analysis
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.mtcnn = MTCNN(
            image_size=256,
            margin=30,  # 30% margin around bounding box as per guidelines
            min_face_size=60,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=False,
            device=device,
            keep_all=True
        )
        
    def detect_and_extract_face(self, image, return_prob=False):
        """
        Detect and extract faces from image
        Args:
            image: Input image (numpy array or PIL Image)
            return_prob: Whether to return detection probabilities
        Returns:
            faces: List of detected face tensors
            boxes: Bounding boxes
            probs: Detection probabilities (if return_prob=True)
        """
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        boxes, probs = self.mtcnn.detect(image)
        
        if boxes is None:
            return [], [], []
        
        # Extract faces
        faces = []
        for box in boxes:
            if box is not None:
                # Add margin and extract face
                x1, y1, x2, y2 = box.astype(int)
                
                # Ensure coordinates are within image bounds
                h, w = image.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                face = image[y1:y2, x1:x2]
                if face.size > 0:
                    faces.append(face)
        
        if return_prob:
            return faces, boxes, probs
        return faces

class UnsharpMaskingProcessor:
    """
    Unsharp masking preprocessing as mentioned in architecture
    Helps highlight manipulation artifacts
    """
    
    def __init__(self, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.amount = amount
        self.threshold = threshold
    
    def apply(self, image):
        """Apply unsharp masking to enhance edges and artifacts"""
        if len(image.shape) == 3:
            # Apply to each channel separately
            enhanced = np.zeros_like(image)
            for i in range(image.shape[2]):
                enhanced[:, :, i] = self._apply_channel(image[:, :, i])
            return enhanced
        else:
            return self._apply_channel(image)
    
    def _apply_channel(self, channel):
        """Apply unsharp masking to single channel"""
        # Gaussian blur
        blurred = cv2.GaussianBlur(channel, self.kernel_size, self.sigma)
        
        # Create mask
        mask = channel - blurred
        
        # Apply threshold
        mask = np.where(np.abs(mask) < self.threshold, 0, mask)
        
        # Apply unsharp masking
        enhanced = channel + self.amount * mask
        
        # Clip values to valid range
        enhanced = np.clip(enhanced, 0, 255)
        
        return enhanced.astype(np.uint8)

class FrequencyDomainProcessor:
    """
    Frequency domain preprocessing for FFT and DCT analysis
    Part of the Frequency Domain Branch
    """
    
    def __init__(self):
        pass
    
    def extract_fft_features(self, image):
        """Extract FFT magnitude and phase features"""
        if len(image.shape) == 3:
            # Convert to grayscale for frequency analysis
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply FFT
        fft = np.fft.fft2(gray)
        fft_shifted = np.fft.fftshift(fft)
        
        # Extract magnitude and phase
        magnitude = np.abs(fft_shifted)
        phase = np.angle(fft_shifted)
        
        # Log transform for better visualization
        magnitude_log = np.log(magnitude + 1)
        
        return {
            'magnitude': magnitude_log,
            'phase': phase,
            'fft_raw': fft_shifted
        }
    
    def extract_dct_features(self, image, block_size=8):
        """Extract DCT features using block-wise processing"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Ensure image dimensions are divisible by block_size
        h, w = gray.shape
        h_pad = (block_size - h % block_size) % block_size
        w_pad = (block_size - w % block_size) % block_size
        
        if h_pad > 0 or w_pad > 0:
            gray = np.pad(gray, ((0, h_pad), (0, w_pad)), mode='reflect')
        
        h, w = gray.shape
        dct_blocks = []
        
        # Process in blocks
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = gray[i:i+block_size, j:j+block_size].astype(np.float32)
                dct_block = cv2.dct(block)
                dct_blocks.append(dct_block)
        
        return np.array(dct_blocks)

class TextureAnalyzer:
    """
    Texture analysis for the Texture & Detail Branch
    Includes LBP and high-frequency noise analysis
    """
    
    def __init__(self):
        pass
    
    def extract_lbp_features(self, image, radius=3, n_points=24):
        """Extract Local Binary Pattern features"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Simple LBP implementation
        h, w = gray.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = gray[i, j]
                code = 0
                
                # Sample points in circle
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    
                    if 0 <= x < h and 0 <= y < w:
                        if gray[x, y] >= center:
                            code |= (1 << k)
                
                lbp[i, j] = code
        
        return lbp
    
    def extract_noise_residuals(self, image):
        """Extract high-frequency noise residuals using SRM-like filters"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = image.astype(np.float32)
        
        # High-pass filters for noise detection
        filters = [
            np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]),  # Laplacian
            np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),   # Cross
            np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) # Edge detection
        ]
        
        residuals = []
        for filt in filters:
            residual = cv2.filter2D(gray, -1, filt)
            residuals.append(residual)
        
        return np.stack(residuals, axis=2)

class DataAugmentation:
    """
    Advanced data augmentation pipeline
    Following the architecture guidelines for robust training
    """
    
    def __init__(self, image_size=224, mode='train'):
        self.image_size = image_size
        self.mode = mode
        
        if mode == 'train':
            self.transform = A.Compose([
                # Spatial augmentations
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.3),
                
                # Color augmentations
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
                
                # Noise and quality augmentations
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                
                # Normalization
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            # Ensure RGB format
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Already in correct format
                pass
            else:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        return self.transform(image=image)['image']

class MixUpAugmentation:
    """
    MixUp augmentation for real-fake pairs
    Mentioned in the architecture guidelines
    """
    
    def __init__(self, alpha=0.5):
        self.alpha = alpha
    
    def __call__(self, real_image, fake_image, real_label, fake_label):
        """Apply MixUp between real and fake images"""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        # Mix images
        mixed_image = lam * real_image + (1 - lam) * fake_image
        
        # Mix labels
        mixed_label = lam * real_label + (1 - lam) * fake_label
        
        return mixed_image, mixed_label

class VideoProcessor:
    """
    Video processing for temporal consistency analysis
    Extracts frames and maintains temporal information
    """
    
    def __init__(self, face_detector, max_frames=32, frame_interval=1):
        self.face_detector = face_detector
        self.max_frames = max_frames
        self.frame_interval = frame_interval
    
    def extract_frames_and_faces(self, video_path):
        """
        Extract frames and faces from video
        Args:
            video_path: Path to video file
        Returns:
            frames: List of extracted face frames
            metadata: Video metadata
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return [], {}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        frames = []
        frame_idx = 0
        extracted_count = 0
        
        while extracted_count < self.max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames at intervals
            if frame_idx % self.frame_interval == 0:
                # Detect and extract face
                faces = self.face_detector.detect_and_extract_face(frame)
                
                if faces and len(faces) > 0:
                    # Take the largest face
                    largest_face = max(faces, key=lambda x: x.shape[0] * x.shape[1] if hasattr(x, 'shape') else 0)
                    if hasattr(largest_face, 'shape') and largest_face.size > 0:
                        frames.append({
                            'frame': largest_face,
                            'frame_idx': frame_idx,
                            'timestamp': frame_idx / fps if fps > 0 else 0
                        })
                        extracted_count += 1
            
            frame_idx += 1
        
        cap.release()
        
        metadata = {
            'fps': fps,
            'duration': duration,
            'total_frames': frame_count,
            'extracted_frames': len(frames)
        }
        
        return frames, metadata

class DeepfakePreprocessor:
    """
    Complete preprocessing pipeline for deepfake detection
    Integrates all components following the architecture guidelines
    """
    
    def __init__(self, image_size=224, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.image_size = image_size
        self.device = device
        
        # Initialize components
        self.face_detector = FaceDetector(device)
        self.unsharp_masking = UnsharpMaskingProcessor()
        self.frequency_processor = FrequencyDomainProcessor()
        self.texture_analyzer = TextureAnalyzer()
        self.augmentation_train = DataAugmentation(image_size, 'train')
        self.augmentation_val = DataAugmentation(image_size, 'val')
        self.mixup = MixUpAugmentation()
        self.video_processor = VideoProcessor(self.face_detector)
    
    def process_image(self, image_path, apply_augmentation=True, mode='train'):
        """
        Complete image preprocessing pipeline
        Args:
            image_path: Path to image file
            apply_augmentation: Whether to apply augmentation
            mode: 'train' or 'val'
        Returns:
            processed_data: Dictionary containing all processed features
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect and extract face
        faces = self.face_detector.detect_and_extract_face(image)
        
        if not faces:
            # If no face detected, use center crop
            h, w = image.shape[:2]
            size = min(h, w)
            start_h = (h - size) // 2
            start_w = (w - size) // 2
            face = image[start_h:start_h + size, start_w:start_w + size]
        else:
            # Use the largest detected face
            face = max(faces, key=lambda x: x.shape[0] * x.shape[1])
        
        # Resize to standard size for processing
        face = cv2.resize(face, (256, 256))
        
        # Apply unsharp masking
        enhanced_face = self.unsharp_masking.apply(face)
        
        # Extract frequency domain features
        fft_features = self.frequency_processor.extract_fft_features(enhanced_face)
        dct_features = self.frequency_processor.extract_dct_features(enhanced_face)
        
        # Extract texture features
        lbp_features = self.texture_analyzer.extract_lbp_features(enhanced_face)
        noise_residuals = self.texture_analyzer.extract_noise_residuals(enhanced_face)
        
        # Apply data augmentation
        if apply_augmentation:
            if mode == 'train':
                spatial_tensor = self.augmentation_train(enhanced_face)
            else:
                spatial_tensor = self.augmentation_val(enhanced_face)
        else:
            spatial_tensor = self.augmentation_val(enhanced_face)
        
        return {
            'spatial': spatial_tensor,
            'fft_magnitude': torch.from_numpy(fft_features['magnitude']).float(),
            'fft_phase': torch.from_numpy(fft_features['phase']).float(),
            'dct_blocks': torch.from_numpy(dct_features).float(),
            'lbp': torch.from_numpy(lbp_features).float(),
            'noise_residuals': torch.from_numpy(noise_residuals).float(),
            'original_face': enhanced_face
        }
    
    def process_video(self, video_path, max_frames=16):
        """
        Complete video preprocessing pipeline
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to extract
        Returns:
            processed_data: Dictionary containing processed video features
        """
        # Extract frames and faces
        frames, metadata = self.video_processor.extract_frames_and_faces(video_path)
        
        if not frames:
            return None
        
        # Limit frames
        frames = frames[:max_frames]
        
        processed_frames = []
        for frame_data in frames:
            face = frame_data['frame']
            
            # Apply unsharp masking
            enhanced_face = self.unsharp_masking.apply(face)
            
            # Apply augmentation (validation mode for consistency)
            spatial_tensor = self.augmentation_val(enhanced_face)
            
            processed_frames.append({
                'spatial': spatial_tensor,
                'timestamp': frame_data['timestamp'],
                'frame_idx': frame_data['frame_idx']
            })
        
        return {
            'frames': processed_frames,
            'metadata': metadata,
            'temporal_length': len(processed_frames)
        }
    
    def create_dataset_batch(self, data_list, labels, apply_mixup=False):
        """
        Create batch with optional MixUp augmentation
        Args:
            data_list: List of processed data dictionaries
            labels: Corresponding labels
            apply_mixup: Whether to apply MixUp
        Returns:
            batch: Batched tensors
        """
        batch_size = len(data_list)
        
        # Stack spatial features
        spatial_batch = torch.stack([data['spatial'] for data in data_list])
        
        # Stack frequency features (resize if needed)
        fft_magnitude_batch = []
        for data in data_list:
            fft_mag = data['fft_magnitude']
            if len(fft_mag.shape) == 2:
                # Resize to standard size
                fft_mag_resized = F.interpolate(
                    fft_mag.unsqueeze(0).unsqueeze(0), 
                    size=(64, 64), 
                    mode='bilinear'
                ).squeeze()
                fft_magnitude_batch.append(fft_mag_resized)
        
        fft_magnitude_batch = torch.stack(fft_magnitude_batch)
        
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        # Apply MixUp if requested and we have both real and fake samples
        if apply_mixup and len(set(labels)) > 1:
            # Find real and fake indices
            real_indices = [i for i, label in enumerate(labels) if label == 0]
            fake_indices = [i for i, label in enumerate(labels) if label == 1]
            
            if real_indices and fake_indices:
                # Randomly pair real and fake samples
                mixed_data = []
                mixed_labels = []
                
                for i in range(min(len(real_indices), len(fake_indices))):
                    real_idx = real_indices[i % len(real_indices)]
                    fake_idx = fake_indices[i % len(fake_indices)]
                    
                    mixed_spatial, mixed_label = self.mixup(
                        spatial_batch[real_idx], 
                        spatial_batch[fake_idx],
                        labels_tensor[real_idx], 
                        labels_tensor[fake_idx]
                    )
                    
                    mixed_data.append(mixed_spatial)
                    mixed_labels.append(mixed_label)
                
                if mixed_data:
                    # Add mixed samples to batch
                    mixed_spatial_batch = torch.stack(mixed_data)
                    mixed_labels_batch = torch.stack(mixed_labels)
                    
                    spatial_batch = torch.cat([spatial_batch, mixed_spatial_batch])
                    labels_tensor = torch.cat([labels_tensor, mixed_labels_batch])
        
        return {
            'spatial': spatial_batch,
            'fft_magnitude': fft_magnitude_batch,
            'labels': labels_tensor
        }

def test_preprocessing_pipeline():
    """Test the preprocessing pipeline with sample data"""
    print("🧪 Testing Preprocessing Pipeline")
    print("=" * 40)
    
    # Initialize preprocessor
    preprocessor = DeepfakePreprocessor(image_size=224)
    
    # Test paths
    data_root = Path("/home/soumya/PycharmProjects/Deepfake_Detection")
    
    # Test image processing
    print("\n🖼️ Testing Image Processing...")
    faceforensics_path = data_root / "FaceForensics/cropped_images"
    
    if faceforensics_path.exists():
        # Find a sample image
        sample_folders = list(faceforensics_path.iterdir())[:2]
        for folder in sample_folders:
            if folder.is_dir():
                images = list(folder.glob("*.png"))[:1]
                for img_path in images:
                    print(f"Processing: {img_path.name}")
                    
                    result = preprocessor.process_image(img_path, mode='train')
                    if result:
                        print(f"✅ Spatial tensor shape: {result['spatial'].shape}")
                        print(f"✅ FFT magnitude shape: {result['fft_magnitude'].shape}")
                        print(f"✅ LBP features shape: {result['lbp'].shape}")
                        print(f"✅ Noise residuals shape: {result['noise_residuals'].shape}")
                    else:
                        print("❌ Failed to process image")
                    break
            if folder.is_dir():
                break
    
    # Test video processing
    print("\n🎬 Testing Video Processing...")
    celeb_df_path = data_root / "Celeb-df-v2/Celeb-real"
    
    if celeb_df_path.exists():
        videos = list(celeb_df_path.glob("*.mp4"))[:1]
        for video_path in videos:
            print(f"Processing: {video_path.name}")
            
            result = preprocessor.process_video(video_path, max_frames=8)
            if result:
                print(f"✅ Extracted {result['temporal_length']} frames")
                print(f"✅ Video duration: {result['metadata']['duration']:.2f}s")
                print(f"✅ Video FPS: {result['metadata']['fps']:.2f}")
                if result['frames']:
                    print(f"✅ Frame tensor shape: {result['frames'][0]['spatial'].shape}")
            else:
                print("❌ Failed to process video")
            break
    
    print("\n✅ Preprocessing Pipeline Test Complete!")

if __name__ == "__main__":
    test_preprocessing_pipeline()