import os
import torch
import gc
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils import AdvancedDeepfakeDetector, process_image, process_video


def test_with_real_data(image_path=None, video_path=None):
    try:
        # Clear GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clear RAM
        gc.collect()

        # Create model
        model = AdvancedDeepfakeDetector()
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()
            print("Using GPU for inference")
        else:
            print("Using CPU for inference")

        print(
            f"Testing on device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

        def process_and_predict(input_tensor):
            with torch.no_grad():
                if torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()
                output = model(input_tensor)
                confidence = torch.sigmoid(output).item()
                return confidence

        # Test with real image if provided
        if image_path and os.path.exists(image_path):
            print(f"\nTesting image: {image_path}")
            try:
                img = Image.open(image_path)
                img_tensor = process_image(img)
                print(f"Image tensor shape: {img_tensor.shape}")

                confidence_threshold = 0.6  # Adjusted threshold for better accuracy

                confidence = process_and_predict(img_tensor.unsqueeze(0))
                prediction = "FAKE" if confidence > confidence_threshold else "REAL"
                confidence_percent = confidence * \
                    100 if prediction == "FAKE" else (1 - confidence) * 100

                # Display results
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(img)
                plt.title("Input Image")
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.text(0.5, 0.5, f"Prediction: {prediction}\nConfidence: {confidence_percent:.2f}%",
                         ha='center', va='center', fontsize=12)
                plt.axis('off')
                plt.show()

                print(f"Prediction: {prediction}")
                print(f"Confidence: {confidence_percent:.2f}%")

            except Exception as e:
                print(f"Error processing image: {str(e)}")

        # Test with real video if provided
        if video_path and os.path.exists(video_path):
            print(f"\nTesting video: {video_path}")
            try:
                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))

                frames = []
                while len(frames) < min(frame_count, 32):  # Process up to 32 frames
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)

                cap.release()

                if frames:
                    video_tensor = process_video(frames)
                    print(f"Video tensor shape: {video_tensor.shape}")

                    confidence = process_and_predict(video_tensor.unsqueeze(0))
                    prediction = "FAKE" if confidence > 0.5 else "REAL"
                    confidence_percent = confidence * \
                        100 if prediction == "FAKE" else (1 - confidence) * 100

                    print(f"Prediction: {prediction}")
                    print(f"Confidence: {confidence_percent:.2f}%")

            except Exception as e:
                print(f"Error processing video: {str(e)}")

        print("\nTesting completed!")

    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # Final cleanup
        if 'model' in locals():
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # Test with sample images
    test_images_dir = os.path.join("..", "test_images")
    for img_file in os.listdir(test_images_dir):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            test_with_real_data(image_path=os.path.join(
                test_images_dir, img_file))

    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
