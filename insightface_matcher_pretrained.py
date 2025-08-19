import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
import logging

logger = logging.getLogger(__name__)

class InsightFaceMatcher:
    def __init__(self, det_size=(640, 640), threshold=0.4):
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=det_size)
        self.threshold = threshold
        logger.info("InsightFace matcher initialized successfully")

    def preprocess_image(self, img):
        """Preprocess image for better face detection"""
        # Convert grayscale to 3-channel RGB
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            logger.info("Converted grayscale image to RGB.")
        elif len(img.shape) == 3 and img.shape[2] == 1:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            logger.info("Converted single-channel image to RGB.")
        elif len(img.shape) == 3 and img.shape[2] == 3:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img

        # Histogram equalization for each channel (improves contrast)
        img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        logger.info("Applied histogram equalization.")

        # Upscale small images to at least 512x512 (increased from 256x256)
        h, w = img_eq.shape[:2]
        if h < 512 or w < 512:
            scale = max(512 / h, 512 / w)
            new_size = (int(w * scale), int(h * scale))
            # Use INTER_LANCZOS4 for better quality upscaling of small images
            img_eq = cv2.resize(img_eq, new_size, interpolation=cv2.INTER_LANCZOS4)
            logger.info(f"Upscaled image from {w}x{h} to {img_eq.shape[1]}x{img_eq.shape[0]} (scale: {scale:.2f}x).")

        # Ensure image is in uint8 format
        if img_eq.dtype != np.uint8:
            img_eq = (img_eq * 255).astype(np.uint8)

        return img_eq

    def get_embedding(self, img):
        """Get face embedding with multiple detection attempts"""
        # Preprocess image
        img_processed = self.preprocess_image(img)
        
        # Try different detection sizes for better face detection
        detection_sizes = [(640, 640), (320, 320), (1280, 1280), (1600, 1600)]
        
        # For very small images, try even larger detection sizes
        h, w = img_processed.shape[:2]
        if h < 200 or w < 200:
            detection_sizes.extend([(2000, 2000), (2400, 2400)])
            logger.info(f"Image is very small ({w}x{h}), adding larger detection sizes.")
        
        for det_size in detection_sizes:
            try:
                # Temporarily change detection size
                self.app.det_model.det_size = det_size
                
                # Get faces
                faces = self.app.get(img_processed)
                logger.info(f"Detection attempt with size {det_size}: Found {len(faces)} faces")
                
                if faces:
                    # Return the first face embedding
                    embedding = faces[0].embedding
                    logger.info(f"Successfully extracted face embedding with size {det_size}")
                    return embedding
                    
            except Exception as e:
                logger.warning(f"Detection failed with size {det_size}: {e}")
                continue
        
        # If no faces found with any size, try with original image
        try:
            faces = self.app.get(img_processed)
            logger.info(f"Final attempt: Found {len(faces)} faces")
            if faces:
                return faces[0].embedding
        except Exception as e:
            logger.error(f"Final detection attempt failed: {e}")
        
        logger.error("No faces detected in image after all attempts")
        return None

    def compare(self, img1, img2):
        """Compare two images and return similarity result"""
        logger.info("Starting face comparison...")
        
        # Get embeddings for both images
        emb1 = self.get_embedding(img1)
        emb2 = self.get_embedding(img2)
        
        if emb1 is None:
            logger.error("No face detected in first image")
            return "No face detected in image 1", 0.0
        elif emb2 is None:
            logger.error("No face detected in second image")
            return "No face detected in image 2", 0.0
        
        # Calculate similarity
        try:
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            logger.info(f"Similarity score: {sim:.4f}")
            
            result = "Same Person" if sim > self.threshold else "Different Person"
            return result, float(sim)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return "Error calculating similarity", 0.0

# Required dependencies:
# pip install insightface onnxruntime opencv-python