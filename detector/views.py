import os
from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.conf import settings
import torch
import numpy as np
from PIL import Image
import cv2
import tempfile
from django.utils import timezone
from datetime import datetime
import random

from .models import DetectionHistory
from .utils import process_image, process_video, create_model

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model(device)

# Constants
CONFIDENCE_THRESHOLD = 0.45  # Lowered from 0.5 to make detection more sensitive
MIN_CONFIDENCE_THRESHOLD = 0.25  # Lowered from 0.3 to allow more detections
HIGH_CONFIDENCE_THRESHOLD = 0.65  # Lowered from 0.7 to make high confidence more achievable


def calculate_confidence_score(raw_confidence, model_confidence, is_fake):
    """Calculate the final confidence score."""
    # Ensure raw_confidence is between 0 and 1
    raw_confidence = max(0, min(1, raw_confidence))
    
    # Calculate base confidence with more balanced scaling
    if is_fake:
        base_confidence = raw_confidence
    else:
        base_confidence = 1 - raw_confidence
    
    # Apply balanced scaling based on model confidence
    # This makes the confidence less extreme
    adjusted_confidence = base_confidence * (0.3 + model_confidence * 0.7)
    
    # Add some randomness to prevent 100% confidence
    adjusted_confidence = adjusted_confidence * (0.95 + random.random() * 0.1)
    
    # Convert to percentage and ensure it's between 0 and 100
    final_confidence = min(100, max(0, adjusted_confidence * 100))
    
    # Round to 2 decimal places to ensure consistency
    return round(final_confidence, 2)


def get_certainty_level(model_confidence, prediction_margin):
    """Determine the certainty level of the prediction."""
    if model_confidence > 0.8 and prediction_margin > 0.2:
        return "High"
    elif model_confidence > 0.6 and prediction_margin > 0.1:
        return "Medium"
    else:
        return "Low"


def index(request):
    """Home page view showing upload form and detection history."""
    history = DetectionHistory.objects.all().order_by('-timestamp')[:10]
    return render(request, 'detector/index.html', {'history': history})


def datasets(request):
    """View for displaying available deepfake datasets."""
    datasets = [
        {
            'name': 'FaceForensics++',
            'description': 'Large-scale facial manipulation detection dataset',
            'url': 'https://github.com/ondyari/FaceForensics',
            'type': 'Video'
        },
        {
            'name': 'DeepFake Detection Challenge Dataset',
            'description': 'Dataset from Facebook\'s deepfake detection challenge',
            'url': 'https://www.kaggle.com/c/deepfake-detection-challenge/data',
            'type': 'Video'
        },
        {
            'name': 'Celeb-DF',
            'description': 'Large-scale deepfake video dataset with high quality',
            'url': 'http://www.cs.albany.edu/~lsw/celeb-deepfakeforensics.html',
            'type': 'Video'
        },
        {
            'name': 'DFDC Preview Dataset',
            'description': 'Preview dataset from the Deepfake Detection Challenge',
            'url': 'https://ai.facebook.com/datasets/dfdc/',
            'type': 'Video'
        },
        {
            'name': 'ThisPersonDoesNotExist',
            'description': 'AI-generated face images',
            'url': 'https://thispersondoesnotexist.com/',
            'type': 'Image'
        }
    ]
    return render(request, 'detector/datasets.html', {'datasets': datasets})


def detect(request):
    """Handle media upload and perform deepfake detection."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)

    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file provided'}, status=400)

    file = request.FILES['file']
    if not file:
        return JsonResponse({'error': 'Empty file'}, status=400)

    # Check file type
    ext = os.path.splitext(file.name)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png', '.gif']:
        input_type = 'image'
    elif ext in ['.mp4', '.avi', '.mov', '.wmv']:
        input_type = 'video'
    else:
        return JsonResponse({'error': 'Invalid file type'}, status=400)

    try:
        # Save file temporarily
        file_path = default_storage.save(f'uploads/{file.name}', file)
        file_path = os.path.join(settings.MEDIA_ROOT, file_path)

        try:
            # Process file based on type
            if input_type == 'image':
                processed_data = process_image(file_path)
            else:
                processed_data = process_video(file_path)

            if processed_data is None:
                return JsonResponse({'error': 'Failed to process media file'}, status=400)

            # Move processed data to device
            processed_data = processed_data.to(device)

            # Get prediction
            with torch.no_grad():
                if input_type == 'video':
                    # For videos, use ensemble prediction with temporal consistency
                    batch_predictions = []
                    batch_confidences = []
                    num_frames = processed_data.size(0)

                    # Process video in chunks
                    for i in range(0, num_frames, 12):
                        chunk = processed_data[i:min(i + 12, num_frames)]
                        pred, conf = model(chunk)
                        if pred is None or conf is None:
                            return JsonResponse({'error': 'Model prediction failed'}, status=500)
                        batch_predictions.extend(pred.cpu().numpy())
                        batch_confidences.extend(conf.cpu().numpy())

                    predictions = np.array(batch_predictions)
                    confidences = np.array(batch_confidences)

                    # Weight predictions by their confidence
                    weighted_preds = predictions * confidences
                    raw_confidence = np.average(weighted_preds, weights=confidences)
                    model_confidence = np.mean(confidences)

                else:
                    # For images, use single prediction
                    prediction, model_confidence = model(processed_data)
                    if prediction is None or model_confidence is None:
                        return JsonResponse({'error': 'Model prediction failed'}, status=500)
                    raw_confidence = float(prediction[0])
                    model_confidence = float(model_confidence[0])

                # Print raw values for debugging
                print(f"Raw confidence: {raw_confidence}")
                print(f"Model confidence: {model_confidence}")

                # Calculate prediction margin
                prediction_margin = abs(raw_confidence - CONFIDENCE_THRESHOLD)

                # More sensitive fake detection
                # Consider it fake if either:
                # 1. Raw confidence is above threshold
                # 2. Raw confidence is close to threshold and model confidence is high
                # 3. Raw confidence is significantly above 0.5
                is_fake = (
                    raw_confidence > CONFIDENCE_THRESHOLD or
                    (raw_confidence > 0.4 and model_confidence > 0.3) or
                    raw_confidence > 0.6
                )

                # Calculate final confidence score
                display_confidence = calculate_confidence_score(
                    raw_confidence, model_confidence, is_fake)

                # Get certainty level
                certainty_level = get_certainty_level(
                    model_confidence, prediction_margin)

                # Print final decision for debugging
                print(f"Final decision - Is fake: {is_fake}, Confidence: {display_confidence}%")

            # Get current time in UTC
            current_time = timezone.now()

            # Save detection history
            history = DetectionHistory.objects.create(
                file=file,
                filename=file.name,
                input_type=input_type,
                is_fake=is_fake,
                confidence=display_confidence,
                timestamp=current_time
            )

            result = {
                'is_fake': is_fake,
                'confidence': display_confidence,
                'filename': file.name,
                'input_type': input_type,
                'certainty': certainty_level,
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'prediction_details': {
                    'raw_confidence': round(float(raw_confidence), 3),
                    'model_confidence': round(float(model_confidence), 3),
                    'prediction_margin': round(float(prediction_margin), 3)
                }
            }

            return JsonResponse(result)

        except ValueError as e:
            return JsonResponse({'error': f'Processing error: {str(e)}'}, status=400)
        except Exception as e:
            return JsonResponse({'error': f'Analysis error: {str(e)}'}, status=500)
        finally:
            # Clean up temporary file
            if os.path.exists(file_path):
                os.remove(file_path)

    except Exception as e:
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)
