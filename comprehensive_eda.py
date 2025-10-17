# Deepfake Detection - Comprehensive EDA Analysis
# Following SOTA Model Architecture Guidelines

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DeepfakeEDA:
    """
    Comprehensive Exploratory Data Analysis for Deepfake Detection
    Analyzes both Celeb-DF-v2 and FaceForensics++ datasets
    """
    
    def __init__(self, data_root="/home/soumya/PycharmProjects/Deepfake_Detection"):
        self.data_root = Path(data_root)
        self.celeb_df_path = self.data_root / "Celeb-df-v2"
        self.faceforensics_path = self.data_root / "FaceForensics"
        
        # Initialize data containers
        self.celeb_data = {
            'real': [],
            'fake': [],
            'youtube_real': []
        }
        self.faceforensics_data = []
        self.analysis_results = {}
        
    def scan_datasets(self, limit=100):
        """
        Scan available datasets and collect metadata
        Args:
            limit: Maximum number of files to scan per category (for quick analysis)
        """
        print("🔍 Scanning Available Datasets...")
        
        # Scan Celeb-DF-v2
        self._scan_celeb_df(limit)
        
        # Scan FaceForensics++
        self._scan_faceforensics(limit)
        
        print(f"✅ Dataset scanning completed!")
        print(f"📊 Celeb-DF Real videos: {len(self.celeb_data['real'])}")
        print(f"📊 Celeb-DF Fake videos: {len(self.celeb_data['fake'])}")
        print(f"📊 YouTube Real videos: {len(self.celeb_data['youtube_real'])}")
        print(f"📊 FaceForensics images: {len(self.faceforensics_data)}")
        
    def _scan_celeb_df(self, limit):
        """Scan Celeb-DF-v2 dataset"""
        # Real videos
        real_path = self.celeb_df_path / "Celeb-real"
        if real_path.exists():
            videos = list(real_path.glob("*.mp4"))[:limit]
            for video in videos:
                self.celeb_data['real'].append({
                    'path': str(video),
                    'filename': video.name,
                    'type': 'real',
                    'dataset': 'celeb-df',
                    'size': video.stat().st_size if video.exists() else 0
                })
        
        # Fake videos
        fake_path = self.celeb_df_path / "Celeb-synthesis"
        if fake_path.exists():
            videos = list(fake_path.glob("*.mp4"))[:limit]
            for video in videos:
                # Extract source and target IDs from filename
                parts = video.stem.split('_')
                source_id = parts[0] if len(parts) > 0 else "unknown"
                target_id = parts[1] if len(parts) > 1 else "unknown"
                
                self.celeb_data['fake'].append({
                    'path': str(video),
                    'filename': video.name,
                    'type': 'fake',
                    'dataset': 'celeb-df',
                    'source_id': source_id,
                    'target_id': target_id,
                    'size': video.stat().st_size if video.exists() else 0
                })
        
        # YouTube real videos
        youtube_path = self.celeb_df_path / "YouTube-real"
        if youtube_path.exists():
            videos = list(youtube_path.glob("*.mp4"))[:limit]
            for video in videos:
                self.celeb_data['youtube_real'].append({
                    'path': str(video),
                    'filename': video.name,
                    'type': 'real',
                    'dataset': 'youtube-real',
                    'size': video.stat().st_size if video.exists() else 0
                })
    
    def _scan_faceforensics(self, limit):
        """Scan FaceForensics++ dataset"""
        cropped_path = self.faceforensics_path / "cropped_images"
        if cropped_path.exists():
            folders = list(cropped_path.iterdir())[:limit//10]  # Sample folders
            for folder in folders:
                if folder.is_dir():
                    images = list(folder.glob("*.png"))[:10]  # Sample images per folder
                    for img in images:
                        # Determine if real or fake based on folder naming convention
                        # Usually 000-399 are real, 400+ are manipulated
                        folder_num = int(folder.name.split('_')[0])
                        is_real = folder_num < 400
                        
                        self.faceforensics_data.append({
                            'path': str(img),
                            'filename': img.name,
                            'folder': folder.name,
                            'type': 'real' if is_real else 'fake',
                            'dataset': 'faceforensics',
                            'size': img.stat().st_size if img.exists() else 0
                        })
    
    def analyze_dataset_distribution(self):
        """Analyze class distribution across datasets"""
        print("\n📈 Dataset Distribution Analysis")
        print("=" * 50)
        
        # Create summary dataframe
        all_data = []
        
        # Add Celeb-DF data
        for category, items in self.celeb_data.items():
            for item in items:
                all_data.append(item)
        
        # Add FaceForensics data
        all_data.extend(self.faceforensics_data)
        
        df = pd.DataFrame(all_data)
        
        if df.empty:
            print("❌ No data found in datasets!")
            return None
        
        # Overall distribution
        print(f"Total files analyzed: {len(df)}")
        print("\n🎯 Class Distribution:")
        class_dist = df['type'].value_counts()
        print(class_dist)
        
        print(f"\n📊 Real vs Fake Ratio: {class_dist.get('real', 0):.0f}:{class_dist.get('fake', 0):.0f}")
        
        # Dataset-wise distribution
        print(f"\n🗂️ Dataset Distribution:")
        dataset_dist = df['dataset'].value_counts()
        print(dataset_dist)
        
        # Create visualizations
        self._plot_distributions(df)
        
        self.analysis_results['distribution'] = {
            'total_files': len(df),
            'class_distribution': class_dist.to_dict(),
            'dataset_distribution': dataset_dist.to_dict()
        }
        
        return df
    
    def _plot_distributions(self, df):
        """Create distribution plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Class distribution
        df['type'].value_counts().plot(kind='bar', ax=axes[0,0], color=['lightgreen', 'lightcoral'])
        axes[0,0].set_title('Real vs Fake Distribution', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Class')
        axes[0,0].set_ylabel('Count')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Dataset distribution
        df['dataset'].value_counts().plot(kind='bar', ax=axes[0,1], color='skyblue')
        axes[0,1].set_title('Dataset Distribution', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Dataset')
        axes[0,1].set_ylabel('Count')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Cross-tabulation
        cross_tab = pd.crosstab(df['dataset'], df['type'])
        cross_tab.plot(kind='bar', stacked=True, ax=axes[1,0], color=['lightgreen', 'lightcoral'])
        axes[1,0].set_title('Dataset vs Class Distribution', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Dataset')
        axes[1,0].set_ylabel('Count')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].legend(title='Type')
        
        # File size distribution
        if 'size' in df.columns and df['size'].sum() > 0:
            df['size_mb'] = df['size'] / (1024 * 1024)
            df.boxplot(column='size_mb', by='type', ax=axes[1,1])
            axes[1,1].set_title('File Size Distribution by Type', fontsize=14, fontweight='bold')
            axes[1,1].set_xlabel('Type')
            axes[1,1].set_ylabel('Size (MB)')
        else:
            axes[1,1].text(0.5, 0.5, 'File size data\nnot available', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('File Size Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_video_properties(self, sample_size=10):
        """Analyze video properties like duration, fps, resolution"""
        print("\n🎬 Video Properties Analysis")
        print("=" * 50)
        
        video_data = []
        
        # Sample videos from each category
        video_paths = []
        
        # Add Celeb-DF videos
        for category, items in self.celeb_data.items():
            video_paths.extend([item['path'] for item in items[:sample_size//3]])
        
        if not video_paths:
            print("❌ No video files found for analysis!")
            return None
        
        print(f"Analyzing {len(video_paths)} video files...")
        
        for video_path in video_paths:
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    # Get video properties
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration = frame_count / fps if fps > 0 else 0
                    
                    video_data.append({
                        'path': video_path,
                        'fps': fps,
                        'duration': duration,
                        'frame_count': frame_count,
                        'width': width,
                        'height': height,
                        'resolution': f"{width}x{height}"
                    })
                    
                cap.release()
            except Exception as e:
                print(f"⚠️ Error processing {video_path}: {e}")
        
        if not video_data:
            print("❌ Could not analyze any video files!")
            return None
        
        df_video = pd.DataFrame(video_data)
        
        # Display statistics
        print(f"\n📊 Video Statistics (n={len(df_video)}):")
        print(f"Average Duration: {df_video['duration'].mean():.2f} seconds")
        print(f"Average FPS: {df_video['fps'].mean():.2f}")
        print(f"Most common resolution: {df_video['resolution'].mode().iloc[0] if not df_video['resolution'].empty else 'N/A'}")
        
        # Plot video properties
        self._plot_video_properties(df_video)
        
        self.analysis_results['video_properties'] = {
            'avg_duration': df_video['duration'].mean(),
            'avg_fps': df_video['fps'].mean(),
            'resolutions': df_video['resolution'].value_counts().to_dict()
        }
        
        return df_video
    
    def _plot_video_properties(self, df_video):
        """Plot video property distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Duration distribution
        df_video['duration'].hist(bins=20, ax=axes[0,0], color='lightblue', alpha=0.7)
        axes[0,0].set_title('Video Duration Distribution', fontsize=12, fontweight='bold')
        axes[0,0].set_xlabel('Duration (seconds)')
        axes[0,0].set_ylabel('Frequency')
        
        # FPS distribution
        df_video['fps'].hist(bins=20, ax=axes[0,1], color='lightgreen', alpha=0.7)
        axes[0,1].set_title('FPS Distribution', fontsize=12, fontweight='bold')
        axes[0,1].set_xlabel('FPS')
        axes[0,1].set_ylabel('Frequency')
        
        # Resolution distribution
        resolution_counts = df_video['resolution'].value_counts().head(10)
        resolution_counts.plot(kind='bar', ax=axes[1,0], color='coral')
        axes[1,0].set_title('Resolution Distribution', fontsize=12, fontweight='bold')
        axes[1,0].set_xlabel('Resolution')
        axes[1,0].set_ylabel('Count')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Frame count vs duration scatter
        axes[1,1].scatter(df_video['duration'], df_video['frame_count'], alpha=0.6, color='purple')
        axes[1,1].set_title('Frame Count vs Duration', fontsize=12, fontweight='bold')
        axes[1,1].set_xlabel('Duration (seconds)')
        axes[1,1].set_ylabel('Frame Count')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_image_quality(self, sample_size=20):
        """Analyze image quality metrics"""
        print("\n🖼️ Image Quality Analysis")
        print("=" * 50)
        
        image_data = []
        
        # Sample images from FaceForensics
        image_paths = [item['path'] for item in self.faceforensics_data[:sample_size]]
        
        if not image_paths:
            print("❌ No image files found for analysis!")
            return None
        
        print(f"Analyzing {len(image_paths)} image files...")
        
        for img_path in image_paths:
            try:
                # Load image
                img = cv2.imread(img_path)
                if img is not None:
                    # Calculate quality metrics
                    height, width = img.shape[:2]
                    
                    # Convert to grayscale for analysis
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Calculate metrics
                    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()  # Blur detection
                    brightness = np.mean(gray)
                    contrast = np.std(gray)
                    
                    # Color channel analysis
                    b, g, r = cv2.split(img)
                    color_balance = {
                        'red_mean': np.mean(r),
                        'green_mean': np.mean(g),
                        'blue_mean': np.mean(b)
                    }
                    
                    image_data.append({
                        'path': img_path,
                        'width': width,
                        'height': height,
                        'blur_score': blur_score,
                        'brightness': brightness,
                        'contrast': contrast,
                        'red_mean': color_balance['red_mean'],
                        'green_mean': color_balance['green_mean'],
                        'blue_mean': color_balance['blue_mean']
                    })
                    
            except Exception as e:
                print(f"⚠️ Error processing {img_path}: {e}")
        
        if not image_data:
            print("❌ Could not analyze any image files!")
            return None
        
        df_images = pd.DataFrame(image_data)
        
        # Display statistics
        print(f"\n📊 Image Quality Statistics (n={len(df_images)}):")
        print(f"Average Resolution: {df_images['width'].mean():.0f}x{df_images['height'].mean():.0f}")
        print(f"Average Blur Score: {df_images['blur_score'].mean():.2f}")
        print(f"Average Brightness: {df_images['brightness'].mean():.2f}")
        print(f"Average Contrast: {df_images['contrast'].mean():.2f}")
        
        # Plot image quality metrics
        self._plot_image_quality(df_images)
        
        self.analysis_results['image_quality'] = {
            'avg_width': df_images['width'].mean(),
            'avg_height': df_images['height'].mean(),
            'avg_blur_score': df_images['blur_score'].mean(),
            'avg_brightness': df_images['brightness'].mean(),
            'avg_contrast': df_images['contrast'].mean()
        }
        
        return df_images
    
    def _plot_image_quality(self, df_images):
        """Plot image quality metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Blur score distribution
        df_images['blur_score'].hist(bins=15, ax=axes[0,0], color='skyblue', alpha=0.7)
        axes[0,0].set_title('Blur Score Distribution', fontsize=12, fontweight='bold')
        axes[0,0].set_xlabel('Blur Score (higher = sharper)')
        axes[0,0].set_ylabel('Frequency')
        
        # Brightness distribution
        df_images['brightness'].hist(bins=15, ax=axes[0,1], color='yellow', alpha=0.7)
        axes[0,1].set_title('Brightness Distribution', fontsize=12, fontweight='bold')
        axes[0,1].set_xlabel('Brightness')
        axes[0,1].set_ylabel('Frequency')
        
        # Contrast distribution
        df_images['contrast'].hist(bins=15, ax=axes[0,2], color='orange', alpha=0.7)
        axes[0,2].set_title('Contrast Distribution', fontsize=12, fontweight='bold')
        axes[0,2].set_xlabel('Contrast')
        axes[0,2].set_ylabel('Frequency')
        
        # Resolution scatter plot
        axes[1,0].scatter(df_images['width'], df_images['height'], alpha=0.6, color='green')
        axes[1,0].set_title('Image Resolution Distribution', fontsize=12, fontweight='bold')
        axes[1,0].set_xlabel('Width')
        axes[1,0].set_ylabel('Height')
        
        # Color balance
        color_means = df_images[['red_mean', 'green_mean', 'blue_mean']].mean()
        colors = ['red', 'green', 'blue']
        axes[1,1].bar(colors, color_means, color=colors, alpha=0.7)
        axes[1,1].set_title('Average Color Channel Values', fontsize=12, fontweight='bold')
        axes[1,1].set_ylabel('Average Intensity')
        
        # Brightness vs Contrast correlation
        axes[1,2].scatter(df_images['brightness'], df_images['contrast'], alpha=0.6, color='purple')
        axes[1,2].set_title('Brightness vs Contrast', fontsize=12, fontweight='bold')
        axes[1,2].set_xlabel('Brightness')
        axes[1,2].set_ylabel('Contrast')
        
        plt.tight_layout()
        plt.show()
    
    def display_sample_images(self, samples_per_class=3):
        """Display sample images from each class"""
        print("\n🖼️ Sample Images Visualization")
        print("=" * 50)
        
        # Collect samples
        real_samples = [item for item in self.faceforensics_data if item['type'] == 'real'][:samples_per_class]
        fake_samples = [item for item in self.faceforensics_data if item['type'] == 'fake'][:samples_per_class]
        
        if not real_samples or not fake_samples:
            print("❌ Insufficient samples for visualization!")
            return
        
        # Create subplot
        fig, axes = plt.subplots(2, max(len(real_samples), len(fake_samples)), 
                                figsize=(15, 6))
        
        # Display real samples
        for i, sample in enumerate(real_samples):
            img = cv2.imread(sample['path'])
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[0, i].imshow(img_rgb)
                axes[0, i].set_title(f"Real\n{sample['filename']}", fontsize=10)
                axes[0, i].axis('off')
        
        # Display fake samples
        for i, sample in enumerate(fake_samples):
            img = cv2.imread(sample['path'])
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[1, i].imshow(img_rgb)
                axes[1, i].set_title(f"Fake\n{sample['filename']}", fontsize=10)
                axes[1, i].axis('off')
        
        # Hide empty subplots
        for i in range(max(len(real_samples), len(fake_samples))):
            if i >= len(real_samples):
                axes[0, i].axis('off')
            if i >= len(fake_samples):
                axes[1, i].axis('off')
        
        plt.suptitle('Sample Images: Real vs Fake', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n📋 COMPREHENSIVE EDA SUMMARY REPORT")
        print("=" * 60)
        
        # Dataset Overview
        print("\n🗂️ DATASET OVERVIEW:")
        print("-" * 30)
        total_files = sum(len(items) for items in self.celeb_data.values()) + len(self.faceforensics_data)
        print(f"Total files analyzed: {total_files}")
        
        if 'distribution' in self.analysis_results:
            dist = self.analysis_results['distribution']
            print(f"Real samples: {dist['class_distribution'].get('real', 0)}")
            print(f"Fake samples: {dist['class_distribution'].get('fake', 0)}")
            
            real_count = dist['class_distribution'].get('real', 0)
            fake_count = dist['class_distribution'].get('fake', 0)
            if real_count + fake_count > 0:
                balance_ratio = min(real_count, fake_count) / max(real_count, fake_count)
                print(f"Class balance ratio: {balance_ratio:.2f} (1.0 = perfect balance)")
        
        # Video Properties
        if 'video_properties' in self.analysis_results:
            print(f"\n🎬 VIDEO PROPERTIES:")
            print("-" * 30)
            vid_props = self.analysis_results['video_properties']
            print(f"Average duration: {vid_props['avg_duration']:.2f} seconds")
            print(f"Average FPS: {vid_props['avg_fps']:.2f}")
            print(f"Common resolutions: {list(vid_props['resolutions'].keys())[:3]}")
        
        # Image Quality
        if 'image_quality' in self.analysis_results:
            print(f"\n🖼️ IMAGE QUALITY:")
            print("-" * 30)
            img_qual = self.analysis_results['image_quality']
            print(f"Average resolution: {img_qual['avg_width']:.0f}x{img_qual['avg_height']:.0f}")
            print(f"Average blur score: {img_qual['avg_blur_score']:.2f}")
            print(f"Average brightness: {img_qual['avg_brightness']:.2f}")
            print(f"Average contrast: {img_qual['avg_contrast']:.2f}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS FOR MODEL TRAINING:")
        print("-" * 45)
        print("✓ Use stratified sampling to maintain class balance")
        print("✓ Apply data augmentation to increase dataset diversity")
        print("✓ Consider temporal consistency for video-based training")
        print("✓ Implement quality-based filtering for poor samples")
        print("✓ Use cross-dataset validation for generalization")
        print("✓ Apply unsharp masking preprocessing as per architecture")
        print("✓ Resize images to 224x224 or 256x256 for model input")
        
        # Data Readiness Assessment
        print(f"\n✅ DATA READINESS ASSESSMENT:")
        print("-" * 35)
        readiness_score = 0
        
        if total_files > 50:
            print("✓ Sufficient data volume for training")
            readiness_score += 25
        else:
            print("⚠️ Limited data volume - consider data augmentation")
        
        if 'distribution' in self.analysis_results:
            real_count = self.analysis_results['distribution']['class_distribution'].get('real', 0)
            fake_count = self.analysis_results['distribution']['class_distribution'].get('fake', 0)
            if real_count > 0 and fake_count > 0:
                print("✓ Both classes present")
                readiness_score += 25
                
                balance_ratio = min(real_count, fake_count) / max(real_count, fake_count)
                if balance_ratio > 0.3:
                    print("✓ Reasonable class balance")
                    readiness_score += 25
                else:
                    print("⚠️ Class imbalance detected")
        
        if 'image_quality' in self.analysis_results:
            blur_score = self.analysis_results['image_quality']['avg_blur_score']
            if blur_score > 100:  # Threshold for acceptable sharpness
                print("✓ Good image quality detected")
                readiness_score += 25
            else:
                print("⚠️ Some images may be blurry")
        
        print(f"\n🎯 Overall Data Readiness: {readiness_score}%")
        
        if readiness_score >= 75:
            print("🚀 Dataset is ready for model training!")
        elif readiness_score >= 50:
            print("⚠️ Dataset needs some preprocessing before training")
        else:
            print("❌ Dataset requires significant preparation")
        
        return self.analysis_results

def main():
    """Main execution function"""
    print("🚀 Starting Comprehensive Deepfake Detection EDA")
    print("Following SOTA Model Architecture Guidelines")
    print("=" * 60)
    
    # Initialize EDA analyzer
    eda = DeepfakeEDA()
    
    # Perform comprehensive analysis
    try:
        # Scan datasets
        eda.scan_datasets(limit=100)  # Limited for quick analysis
        
        # Analyze distributions
        df = eda.analyze_dataset_distribution()
        
        # Analyze video properties
        if eda.celeb_data['real'] or eda.celeb_data['fake']:
            eda.analyze_video_properties(sample_size=10)
        
        # Analyze image quality
        if eda.faceforensics_data:
            eda.analyze_image_quality(sample_size=20)
            eda.display_sample_images(samples_per_class=3)
        
        # Generate summary report
        results = eda.generate_summary_report()
        
        # Save results
        import json
        with open('/home/soumya/PycharmProjects/Deepfake_Detection/eda_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n✅ EDA Analysis Complete!")
        print("📄 Results saved to 'eda_results.json'")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()