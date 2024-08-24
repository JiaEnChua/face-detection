import cv2
import numpy as np
import argparse

def detect_and_extract_face(image_path, output_path):
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image at {image_path}")
        return

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        print("No face detected in the image.")
        return

    # Extract the first detected face
    x, y, w, h = faces[0]
    
    # Calculate the center of the face
    center_x, center_y = x + w // 2, y + h // 2
    
    # Calculate new dimensions (make it square and slightly smaller)
    new_size = int(min(w, h) * 0.8)
    
    # Calculate new coordinates
    new_x = max(center_x - new_size // 2, 0)
    new_y = max(center_y - new_size // 2, 0)
    
    # Ensure the new region doesn't exceed image boundaries
    new_x = min(new_x, img.shape[1] - new_size)
    new_y = min(new_y, img.shape[0] - new_size)
    
    # Extract the face with tighter cropping
    face_img = img[new_y:new_y+new_size, new_x:new_x+new_size]

    # Save the extracted face
    cv2.imwrite(output_path, face_img)
    print(f"Face extracted and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and extract face from an image")
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument("output_image", help="Path to save the extracted face")
    args = parser.parse_args()

    detect_and_extract_face(args.input_image, args.output_image)