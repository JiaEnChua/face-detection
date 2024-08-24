import cv2
import numpy as np
import argparse
import dlib

def detect_and_extract_face(image_path, output_path):
    # Load the pre-trained face detection model
    face_detector = dlib.get_frontal_face_detector()
    
    # Load the pre-trained facial landmark detector
    predictor_path = "shape_predictor_68_face_landmarks.dat"  # You need to download this file
    landmark_predictor = dlib.shape_predictor(predictor_path)

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image at {image_path}")
        return

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector(gray)

    if len(faces) == 0:
        print("No face detected in the image.")
        return

    # Get the first detected face
    face = faces[0]

    # Detect landmarks
    landmarks = landmark_predictor(gray, face)

    # Create a mask using landmarks
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    points = np.array([(p.x, p.y) for p in landmarks.parts()])
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)

    # Apply the mask to the original image
    face_img = cv2.bitwise_and(img, img, mask=mask)

    # Get bounding box of the face
    x, y, w, h = face.left(), face.top(), face.width(), face.height()

    # Crop the image to the bounding rectangle of the face
    face_img = face_img[y:y+h, x:x+w]
    mask = mask[y:y+h, x:x+w]

    # Create a white background
    white_bg = np.full(face_img.shape, 255, dtype=np.uint8)

    # Blend the face with the white background
    face_img = np.where(mask[:,:,None].astype(bool), face_img, white_bg)

    # Save the extracted face
    cv2.imwrite(output_path, face_img)
    print(f"Face extracted and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and extract face from an image")
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument("output_image", help="Path to save the extracted face")
    args = parser.parse_args()

    detect_and_extract_face(args.input_image, args.output_image)