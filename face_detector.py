import cv2
import numpy as np
import argparse
import mediapipe as mp

def detect_and_extract_face(image_path, output_path):
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image at {image_path}")
        return

    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        print("No face detected in the image.")
        return

    # Get the landmarks
    face_landmarks = results.multi_face_landmarks[0]

    # Get the bounding box of the face
    h, w, _ = img.shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    for landmark in face_landmarks.landmark:
        x, y = int(landmark.x * w), int(landmark.y * h)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)

    # Expand the bounding box to include hair
    expansion_factor = 0.3  # Adjust this value to include more or less hair
    x_min = max(0, int(x_min - (x_max - x_min) * expansion_factor))
    y_min = max(0, int(y_min - (y_max - y_min) * expansion_factor))
    x_max = min(w, int(x_max + (x_max - x_min) * expansion_factor))
    y_max = min(h, int(y_max + (y_max - y_min) * expansion_factor))

    # Create a mask using the landmarks
    mask = np.zeros((h, w), dtype=np.uint8)
    points = [(int(landmark.x * w), int(landmark.y * h)) for landmark in face_landmarks.landmark]
    hull = cv2.convexHull(np.array(points))
    cv2.fillConvexPoly(mask, hull, 255)

    # Expand the mask
    kernel = np.ones((20, 20), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Apply the mask to the original image
    face_img = cv2.bitwise_and(img, img, mask=mask)

    # Crop the image
    face_img = face_img[y_min:y_max, x_min:x_max]
    mask = mask[y_min:y_max, x_min:x_max]

    # Create a white background
    white_bg = np.full(face_img.shape, 255, dtype=np.uint8)

    # Blend the face with the white background
    face_img = np.where(mask[:,:,None].astype(bool), face_img, white_bg)

    # Save the extracted face
    cv2.imwrite(output_path, face_img)
    print(f"Face and hair extracted and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and extract face from an image")
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument("output_image", help="Path to save the extracted face")
    args = parser.parse_args()

    detect_and_extract_face(args.input_image, args.output_image)