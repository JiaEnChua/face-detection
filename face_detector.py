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

def replace_green_circle(input_image_path, head_img_rgba, face_mask, output_path):
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

    # Resize head_img_rgba to fit the green area
    head_resized = cv2.resize(head_img_rgba, (w, h))

    # Split the head_resized into color and alpha channels
    b, g, r, a = cv2.split(head_resized)
    head_rgb = cv2.merge((b, g, r))

    # Create a mask from the alpha channel
    alpha_mask = a.astype(np.float32) / 255.0

    # Create a full-size alpha mask
    full_alpha_mask = np.zeros(img.shape[:2], dtype=np.float32)
    full_alpha_mask[y:y+h, x:x+w] = alpha_mask

    # Create an inverse mask for the green area
    inverse_green_mask = cv2.bitwise_not(green_mask)

    # Blend the head image with the original image only in the green area
    for c in range(3):  # for each color channel
        img[:, :, c] = img[:, :, c] * (inverse_green_mask / 255.0) + \
                       (img[:, :, c] * (1 - full_alpha_mask) + \
                        np.pad(head_rgb[:, :, c], ((y, img.shape[0]-y-h), (x, img.shape[1]-x-w)), mode='constant') * full_alpha_mask)

    # Convert the result back to uint8
    img = img.astype(np.uint8)

    # Save the result
    cv2.imwrite(output_path, img)
    print(f"Image with replaced face saved to {output_path} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

def replace_head_no_mask(input_image_path, head_img_rgba, face_mask, output_path):
    # Read the input image
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Error: Unable to read image at {input_image_path}")
        return

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
    head_mask = np.zeros(category_mask_array.shape, dtype=np.uint8)
    head_mask[(category_mask_array == 1) | (category_mask_array == 3)] = 255

    # Find the bounding box of the head
    non_zero = cv2.findNonZero(head_mask)
    x, y, w, h = cv2.boundingRect(non_zero)

    # Resize head_img_rgba to fit the detected head area
    head_resized = cv2.resize(head_img_rgba, (w, h))

    # Split the head_resized into color and alpha channels
    b, g, r, a = cv2.split(head_resized)
    head_rgb = cv2.merge((b, g, r))

    # Create a mask from the alpha channel
    alpha_mask = a.astype(np.float32) / 255.0

    # Blend the head image with the original image in the detected head area
    for c in range(3):  # for each color channel
        img[y:y+h, x:x+w, c] = img[y:y+h, x:x+w, c] * (1 - alpha_mask) + head_rgb[:, :, c] * alpha_mask

    # Convert the result back to uint8
    img = img.astype(np.uint8)

    # Save the result
    cv2.imwrite(output_path, img)
    print(f"Image with replaced head saved to {output_path} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

if __name__ == "__main__":
    detect_and_extract_face("input_image.jpg")