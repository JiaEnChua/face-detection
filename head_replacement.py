import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def replace_head_no_mask(input_image_path, head_img_rgba, face_mask):
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
    return img