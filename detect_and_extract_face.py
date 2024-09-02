import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def detect_and_extract_face(img):
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
    return head_img_rgba, hair_face_mask

