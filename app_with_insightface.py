from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import base64
from PIL import Image
import io
import numpy as np
import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize InsightFace matcher
face_matcher = None

try:
    # Load InsightFace matcher
    from insightface_matcher_pretrained import InsightFaceMatcher
    face_matcher = InsightFaceMatcher(threshold=0.4)
    logger.info("✓ InsightFace matcher loaded successfully")
except Exception as e:
    logger.error(f"Could not load InsightFace: {e}")

# Ensure upload directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index_enhanced.html')

@app.route('/compare', methods=['POST'])
def compare_faces():
    """API endpoint to compare two face images using InsightFace"""
    try:
        data = request.get_json()
        if not data or 'image1' not in data or 'image2' not in data:
            return jsonify({'error': 'Two images are required'}), 400
        
        # Decode base64 images
        image1_data = data['image1'].split(',')[1] if ',' in data['image1'] else data['image1']
        image2_data = data['image2'].split(',')[1] if ',' in data['image2'] else data['image2']
        
        # Decode and save images for debugging
        import time
        timestamp = int(time.time())
        
        try:
            image1 = Image.open(io.BytesIO(base64.b64decode(image1_data)))
            image2 = Image.open(io.BytesIO(base64.b64decode(image2_data)))
            
            # Save images for debugging
            image1_path = os.path.join(UPLOAD_FOLDER, f'debug_img1_{timestamp}.png')
            image2_path = os.path.join(UPLOAD_FOLDER, f'debug_img2_{timestamp}.png')
            image1.save(image1_path)
            image2.save(image2_path)
            logger.info(f"Saved debug images: {image1_path}, {image2_path}")
            
        except Exception as e:
            logger.error(f"Error decoding/saving images: {e}")
            return jsonify({'error': f'Error processing images: {str(e)}'}), 500
        
        # Convert to OpenCV format
        image1_cv = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
        image2_cv = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)
        
        logger.info(f"Image 1 shape: {image1_cv.shape}, dtype: {image1_cv.dtype}")
        logger.info(f"Image 2 shape: {image2_cv.shape}, dtype: {image2_cv.dtype}")
        
        # Use InsightFace matcher
        if face_matcher is None:
            return jsonify({'error': 'InsightFace model not loaded'}), 500
        
        try:
            result, similarity = face_matcher.compare(image1_cv, image2_cv)
            confidence_level = "High" if similarity > 0.7 else "Medium" if similarity > 0.5 else "Low"
            details = {"model": "InsightFace", "threshold": face_matcher.threshold}
            
            response = {
                'result': result,
                'similarity_score': float(similarity),
                'confidence': confidence_level,
                'details': details,
                'debug_info': {
                    'image1_saved': image1_path,
                    'image2_saved': image2_path,
                    'image1_shape': image1_cv.shape,
                    'image2_shape': image2_cv.shape
                }
            }
            
        except Exception as e:
            logger.error(f"Error in face comparison: {e}")
            response = {
                'error': str(e),
                'result': 'Error',
                'similarity_score': 0.0,
                'confidence': 'Error',
                'debug_info': {
                    'image1_saved': image1_path,
                    'image2_saved': image2_path,
                    'image1_shape': image1_cv.shape,
                    'image2_shape': image2_cv.shape
                }
            }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in face comparison: {str(e)}")
        return jsonify({'error': f'Error processing images: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'message': 'Face verification service is running',
        'model_loaded': face_matcher is not None
    })

@app.route('/model-info')
def model_info():
    """Get information about the loaded model"""
    try:
        if face_matcher is None:
            return jsonify({'error': 'No model loaded'}), 500
        
        model_info = {
            'insightface': {
                'type': 'InsightFace (Pre-trained)',
                'threshold': face_matcher.threshold,
                'description': 'High-accuracy face recognition using InsightFace'
            }
        }
        
        return jsonify({
            'models': model_info,
            'total_models': 1
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 