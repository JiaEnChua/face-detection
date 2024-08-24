import cv2
import numpy as np
import argparse
import mediapipe as mp

def detect_and_extract_face(image_path):
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image at {image_path}")
        return None, None

    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        print("No face detected in the image.")
        return None, None

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

    return face_img, mask

def replace_green_circle(input_image_path, face_image, face_mask, output_path):
    # Read the input image
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Error: Unable to read image at {input_image_path}")
        return

    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range for green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Create a mask for green color
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours in the green mask
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No green area found in the image.")
        return

    # Find the largest contour (assuming it's the green area)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the green area
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Calculate the diagonal of the green area
    diagonal = int(np.sqrt(w**2 + h**2))

    # Resize face_image to be larger than the diagonal
    scale_factor = 1.2  # Increase this value to make the face larger
    new_size = int(diagonal * scale_factor)
    face_resized = cv2.resize(face_image, (new_size, new_size))
    face_mask_resized = cv2.resize(face_mask, (new_size, new_size))

    # Calculate offsets to center the face in the green area
    x_offset = (new_size - w) // 2
    y_offset = (new_size - h) // 2

    # Crop the resized face to fit the green area
    face_cropped = face_resized[y_offset:y_offset+h, x_offset:x_offset+w]
    face_mask_cropped = face_mask_resized[y_offset:y_offset+h, x_offset:x_offset+w]

    # Create a mask for the green area
    green_area_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(green_area_mask, [largest_contour], 0, (255), -1)

    # Apply the green area mask to the face
    face_masked = cv2.bitwise_and(face_cropped, face_cropped, mask=green_area_mask[y:y+h, x:x+w])

    # Place the face inside the green area
    img[y:y+h, x:x+w] = cv2.bitwise_and(img[y:y+h, x:x+w], img[y:y+h, x:x+w], mask=cv2.bitwise_not(green_area_mask[y:y+h, x:x+w]))
    img[y:y+h, x:x+w] = cv2.add(img[y:y+h, x:x+w], face_masked)

    # Save the result
    cv2.imwrite(output_path, img)
    print(f"Image with replaced face saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace green circle in an image with extracted face")
    parser.add_argument("face_image", help="Path to the image for face extraction")
    parser.add_argument("input_image", help="Path to the image with green circle")
    parser.add_argument("output_image", help="Path to save the final image")
    args = parser.parse_args()

    # Extract face
    face_img, face_mask = detect_and_extract_face(args.face_image)
    if face_img is not None and face_mask is not None:
        # Replace green circle with face
        replace_green_circle(args.input_image, face_img, face_mask, args.output_image)
    else:
        print("Face extraction failed. Cannot proceed with replacement.")