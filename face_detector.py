import cv2
import numpy as np
import argparse
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

def detect_and_extract_face(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image at {image_path}")
        return None, None

    # Convert the image to RGB (MediaPipe uses RGB)
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create ImageSegmenterOptions
    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Update the path to the model file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models', 'selfie_multiclass_256x256.tflite')

    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        output_category_mask=True)

    # Create the segmenter
    with ImageSegmenter.create_from_options(options) as segmenter:
        # Perform segmentation
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        segmentation_result = segmenter.segment(mp_image)
        category_mask = segmentation_result.category_mask

    # Convert the MediaPipe Image to a numpy array
    category_mask_array = np.array(category_mask.numpy_view())

    # Create a binary mask for hair and face skin
    hair_face_mask = np.zeros(category_mask_array.shape, dtype=np.uint8)
    hair_face_mask[(category_mask_array == 1) | (category_mask_array == 3)] = 255

    # Apply the mask to the original image
    head_img = cv2.bitwise_and(img, img, mask=hair_face_mask)

    # Create a transparent background
    b, g, r = cv2.split(head_img)
    alpha = hair_face_mask
    head_img_rgba = cv2.merge((b, g, r, alpha))

    # Find the bounding box of the non-zero regions
    non_zero = cv2.findNonZero(hair_face_mask)
    if non_zero is not None:
        x, y, w, h = cv2.boundingRect(non_zero)
        # Crop the image
        head_img_rgba = head_img_rgba[y:y+h, x:x+w]
        hair_face_mask = hair_face_mask[y:y+h, x:x+w]
    else:
        print("No face or hair detected in the image.")
        return None, None

    # Save the extracted head with transparent background
    output_path = "output_head.png"
    cv2.imwrite(output_path, head_img_rgba)
    print(f"<<<<<<<<<<<Head (hair and face skin) extracted and saved to {output_path}")

    return head_img_rgba, hair_face_mask

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

    # Calculate offsets to center the face in the green area
    x_offset = (new_size - w) // 2
    y_offset = (new_size - h) // 2

    # Crop the resized face to fit the green area
    face_cropped = face_resized[y_offset:y_offset+h, x_offset:x_offset+w]

    # Create a mask for the green area
    green_area_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(green_area_mask, [largest_contour], 0, (255), -1)

    # Split the face_cropped into color and alpha channels
    b, g, r, a = cv2.split(face_cropped)
    face_rgb = cv2.merge((b, g, r))

    # Apply the alpha channel as a mask
    face_masked = cv2.bitwise_and(face_rgb, face_rgb, mask=a)

    # Place the face inside the green area
    img[y:y+h, x:x+w] = cv2.bitwise_and(img[y:y+h, x:x+w], img[y:y+h, x:x+w], mask=cv2.bitwise_not(green_area_mask[y:y+h, x:x+w]))
    img[y:y+h, x:x+w] = cv2.add(img[y:y+h, x:x+w], face_masked)

    # Save the result
    cv2.imwrite(output_path, img)
    print(f"Image with replaced face saved to {output_path} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

if __name__ == "__main__":
    detect_and_extract_face("input_image.jpg")