#!/usr/bin/env python3
# Copyright 2024-2025 Image Quality Evaluation Tool
"""
Image Quality Evaluation Tool

Compares generated images/videos with reference images/videos and computes:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)

Usage:
    # Evaluate single image
    python evaluate_quality.py --generated image.png --reference baseline.png --output metrics.json
    
    # Evaluate video (compares frame by frame)
    python evaluate_quality.py --generated video.mp4 --reference baseline.mp4 --output metrics.json
    
    # From your generation code
    from evaluate_quality import evaluate_image_quality
    metrics = evaluate_image_quality(generated_path, reference_path, output_path)
"""

import argparse
import json
import os
import sys
import logging
from typing import Dict, Optional, Tuple, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Try to import LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

class QualityEvaluator:
    """Image/Video Quality Evaluator"""
    
    def __init__(self, device: str = "auto"):
        """
        Initialize evaluator
        
        Args:
            device: Device for LPIPS computation ('cuda', 'cpu', or 'auto')
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Initialize LPIPS model if available
        self.lpips_model = None
        self.lpips_available = LPIPS_AVAILABLE  # Use instance variable
        
        if self.lpips_available:
            try:
                self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
                self.lpips_model.eval()
                logging.info(f"LPIPS model initialized on {self.device}")
            except Exception as e:
                logging.warning(f"Failed to initialize LPIPS model: {e}")
                self.lpips_available = False
        else:
            logging.warning("LPIPS not available. Install with: pip install lpips")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess image
        
        Args:
            image_path: Path to image file
            
        Returns:
            numpy array of shape (H, W, C) with values in [0, 1]
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load image
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img) / 255.0
        else:
            raise ValueError(f"Unsupported image format: {image_path}")
        
        return img_array
    
    def preprocess_video_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess video frame
        
        Args:
            frame: Raw frame from cv2 (BGR format)
            
        Returns:
            numpy array of shape (H, W, C) with values in [0, 1] (RGB format)
        """
        # Convert BGR to RGB and normalize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb / 255.0
    
    def compute_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute PSNR between two images
        
        Args:
            img1, img2: Images as numpy arrays with values in [0, 1]
            
        Returns:
            PSNR value
        """
        return float(psnr(img1, img2, data_range=1.0))
    
    def compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute SSIM between two images
        
        Args:
            img1, img2: Images as numpy arrays with values in [0, 1]
            
        Returns:
            SSIM value
        """
        # Convert to grayscale if needed for SSIM computation
        if len(img1.shape) == 3:
            # Use multichannel SSIM
            return float(ssim(img1, img2, data_range=1.0, channel_axis=2))
        else:
            return float(ssim(img1, img2, data_range=1.0))
    
    def compute_lpips(self, img1: np.ndarray, img2: np.ndarray) -> Optional[float]:
        """
        Compute LPIPS between two images
        
        Args:
            img1, img2: Images as numpy arrays with values in [0, 1]
            
        Returns:
            LPIPS value or None if LPIPS not available
        """
        if not self.lpips_available or self.lpips_model is None:
            return None
        
        try:
            # Convert to tensors and normalize to [-1, 1]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            # Convert numpy arrays to PIL Images then to tensors
            pil1 = Image.fromarray((img1 * 255).astype(np.uint8))
            pil2 = Image.fromarray((img2 * 255).astype(np.uint8))
            
            tensor1 = transform(pil1).unsqueeze(0).to(self.device)
            tensor2 = transform(pil2).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                lpips_value = self.lpips_model(tensor1, tensor2)
            
            return float(lpips_value.item())
            
        except Exception as e:
            logging.warning(f"LPIPS computation failed: {e}")
            return None
    
    def resize_to_match(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resize images to match dimensions (resize to smaller one)
        
        Args:
            img1, img2: Images as numpy arrays
            
        Returns:
            Resized images
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Resize to the smaller dimensions
        target_h = min(h1, h2)
        target_w = min(w1, w2)
        
        if (h1, w1) != (target_h, target_w):
            img1 = cv2.resize(img1, (target_w, target_h))
        
        if (h2, w2) != (target_h, target_w):
            img2 = cv2.resize(img2, (target_w, target_h))
        
        return img1, img2
    
    def evaluate_images(self, generated_path: str, reference_path: str) -> Dict[str, float]:
        """
        Evaluate quality metrics between two images
        
        Args:
            generated_path: Path to generated image
            reference_path: Path to reference image
            
        Returns:
            Dictionary containing metrics
        """
        # Check if reference file exists
        if not os.path.exists(reference_path):
            raise FileNotFoundError(f"Reference file not found: {reference_path}")
        
        if not os.path.exists(generated_path):
            raise FileNotFoundError(f"Generated file not found: {generated_path}")
        
        # Load and preprocess images
        img_gen = self.preprocess_image(generated_path)
        img_ref = self.preprocess_image(reference_path)
        
        # Resize to match if necessary
        img_gen, img_ref = self.resize_to_match(img_gen, img_ref)
        
        # Compute metrics
        metrics = {}
        
        try:
            metrics['psnr'] = self.compute_psnr(img_ref, img_gen)
        except Exception as e:
            logging.warning(f"PSNR computation failed: {e}")
            metrics['psnr'] = None
        
        try:
            metrics['ssim'] = self.compute_ssim(img_ref, img_gen)
        except Exception as e:
            logging.warning(f"SSIM computation failed: {e}")
            metrics['ssim'] = None
        
        try:
            metrics['lpips'] = self.compute_lpips(img_ref, img_gen)
        except Exception as e:
            logging.warning(f"LPIPS computation failed: {e}")
            metrics['lpips'] = None
        
        return metrics
    
    def evaluate_videos(self, generated_path: str, reference_path: str, 
                       max_frames: Optional[int] = None) -> Dict[str, Union[float, Dict]]:
        """
        Evaluate quality metrics between two videos (frame by frame)
        
        Args:
            generated_path: Path to generated video
            reference_path: Path to reference video
            max_frames: Maximum number of frames to compare (None for all)
            
        Returns:
            Dictionary containing average metrics and frame-by-frame metrics
        """
        # Check if reference file exists
        if not os.path.exists(reference_path):
            raise FileNotFoundError(f"Reference video not found: {reference_path}")
        
        if not os.path.exists(generated_path):
            raise FileNotFoundError(f"Generated video not found: {generated_path}")
        
        # Open video files
        cap_gen = cv2.VideoCapture(generated_path)
        cap_ref = cv2.VideoCapture(reference_path)
        
        if not cap_gen.isOpened() or not cap_ref.isOpened():
            raise ValueError("Failed to open video files")
        
        frame_metrics = []
        frame_idx = 0
        
        try:
            while True:
                if max_frames is not None and frame_idx >= max_frames:
                    break
                
                ret_gen, frame_gen = cap_gen.read()
                ret_ref, frame_ref = cap_ref.read()
                
                if not ret_gen or not ret_ref:
                    break
                
                # Preprocess frames
                img_gen = self.preprocess_video_frame(frame_gen)
                img_ref = self.preprocess_video_frame(frame_ref)
                
                # Resize to match if necessary
                img_gen, img_ref = self.resize_to_match(img_gen, img_ref)
                
                # Compute metrics for this frame
                frame_metric = {}
                
                try:
                    frame_metric['psnr'] = self.compute_psnr(img_ref, img_gen)
                except:
                    frame_metric['psnr'] = None
                
                try:
                    frame_metric['ssim'] = self.compute_ssim(img_ref, img_gen)
                except:
                    frame_metric['ssim'] = None
                
                try:
                    frame_metric['lpips'] = self.compute_lpips(img_ref, img_gen)
                except:
                    frame_metric['lpips'] = None
                
                frame_metrics.append(frame_metric)
                frame_idx += 1
                
                if frame_idx % 30 == 0:  # Progress logging
                    logging.info(f"Processed {frame_idx} frames...")
        
        finally:
            cap_gen.release()
            cap_ref.release()
        
        if not frame_metrics:
            raise ValueError("No frames could be processed")
        
        # Compute average metrics
        avg_metrics = {}
        
        # Calculate averages, ignoring None values
        for metric in ['psnr', 'ssim', 'lpips']:
            values = [fm[metric] for fm in frame_metrics if fm[metric] is not None]
            if values:
                avg_metrics[f'avg_{metric}'] = np.mean(values)
                avg_metrics[f'std_{metric}'] = np.std(values)
            else:
                avg_metrics[f'avg_{metric}'] = None
                avg_metrics[f'std_{metric}'] = None
        
        return {
            'average_metrics': avg_metrics,
            'frame_count': len(frame_metrics),
            'frame_metrics': frame_metrics
        }


def evaluate_image_quality(generated_path: str, reference_path: str, 
                         output_path: Optional[str] = None,
                         device: str = "auto") -> Dict[str, Union[float, Dict]]:
    """
    Main function to evaluate image or video quality
    
    Args:
        generated_path: Path to generated image/video
        reference_path: Path to reference image/video
        output_path: Path to save metrics JSON (optional)
        device: Device for LPIPS computation
        
    Returns:
        Dictionary containing quality metrics
    """
    evaluator = QualityEvaluator(device=device)
    
    # Determine if we're dealing with images or videos
    is_video_gen = generated_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    is_video_ref = reference_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    if is_video_gen != is_video_ref:
        raise ValueError("Both files must be either images or videos")
    
    # Compute metrics
    if is_video_gen:
        logging.info("Evaluating video quality...")
        metrics = evaluator.evaluate_videos(generated_path, reference_path)
    else:
        logging.info("Evaluating image quality...")
        metrics = evaluator.evaluate_images(generated_path, reference_path)
    
    # Add metadata
    result = {
        'generated_path': generated_path,
        'reference_path': reference_path,
        'file_type': 'video' if is_video_gen else 'image',
        'metrics': metrics,
        'timestamp': str(datetime.now())
    }
    
    # Save to file if specified
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        logging.info(f"Metrics saved to: {output_path}")
    
    return result


def evaluate_quality_with_origin(image_path: str, tag: str):
    """评测生成图像质量"""
    
    # 构建基准文件路径
    baseline_path = image_path.replace(f"{tag}_", "ori_")  # 假设基准文件是ori_前缀
    
    if not os.path.exists(baseline_path):
        logging.error(f"Baseline file not found: {baseline_path}")
        logging.error("Please ensure baseline image exists for quality evaluation.")
        return None
    
    # 构建质量评测结果保存路径
    metrics_path = os.path.join(os.path.dirname(image_path), f"{tag}_quality_metrics.json")
    
    try:
        # 执行质量评测
        logging.info(f"Evaluating quality against baseline: {baseline_path}")
        result = evaluate_image_quality(image_path, baseline_path, metrics_path)
        
        # 打印结果
        metrics = result['metrics']
        logging.info(f"Quality Metrics:")
        if metrics.get('psnr') is not None:
            logging.info(f"  PSNR: {metrics['psnr']:.4f} dB")
        if metrics.get('ssim') is not None:
            logging.info(f"  SSIM: {metrics['ssim']:.4f}")
        if metrics.get('lpips') is not None:
            logging.info(f"  LPIPS: {metrics['lpips']:.4f}")
        
        return result
        
    except Exception as e:
        logging.error(f"Quality evaluation failed: {e}")
        return None


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Image/Video Quality Evaluation Tool')
    
    parser.add_argument('--generated', '-g', type=str, required=True,
                        help='Path to generated image/video')
    parser.add_argument('--reference', '-r', type=str, required=True,
                        help='Path to reference image/video')
    parser.add_argument('--output', '-o', type=str, 
                        help='Path to save metrics JSON file')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Device for LPIPS computation')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum frames to process for video evaluation')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Evaluate quality
        result = evaluate_image_quality(
            args.generated, 
            args.reference, 
            args.output, 
            args.device
        )
        
        # Print results
        print("\n" + "="*60)
        print("QUALITY EVALUATION RESULTS")
        print("="*60)
        print(f"Generated: {args.generated}")
        print(f"Reference: {args.reference}")
        print(f"File Type: {result['file_type']}")
        print()
        
        if result['file_type'] == 'image':
            metrics = result['metrics']
            print("Metrics:")
            if metrics.get('psnr') is not None:
                print(f"  PSNR: {metrics['psnr']:.4f} dB")
            if metrics.get('ssim') is not None:
                print(f"  SSIM: {metrics['ssim']:.4f}")
            if metrics.get('lpips') is not None:
                print(f"  LPIPS: {metrics['lpips']:.4f}")
            else:
                print("  LPIPS: Not available (install with: pip install lpips)")
        else:  # video
            avg_metrics = result['metrics']['average_metrics']
            print(f"Average Metrics ({result['metrics']['frame_count']} frames):")
            if avg_metrics.get('avg_psnr') is not None:
                print(f"  PSNR: {avg_metrics['avg_psnr']:.4f} ± {avg_metrics['std_psnr']:.4f} dB")
            if avg_metrics.get('avg_ssim') is not None:
                print(f"  SSIM: {avg_metrics['avg_ssim']:.4f} ± {avg_metrics['std_ssim']:.4f}")
            if avg_metrics.get('avg_lpips') is not None:
                print(f"  LPIPS: {avg_metrics['avg_lpips']:.4f} ± {avg_metrics['std_lpips']:.4f}")
            else:
                print("  LPIPS: Not available (install with: pip install lpips)")
        
        print("="*60)
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()