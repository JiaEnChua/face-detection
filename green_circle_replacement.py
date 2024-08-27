import cv2
import numpy as np
import mediapipe as mp
from face_detector import detect_and_extract_face
import os

def replace_green_circle(original_img, green_img, head_img_rgba, face_mask):
    # Check if inputs are valid NumPy arrays
    if not isinstance(original_img, np.ndarray) or not isinstance(green_img, np.ndarray):
        print("Error: Input images must be NumPy arrays")
        return None

    # Convert green_img to HSV color space
    hsv = cv2.cvtColor(green_img, cv2.COLOR_BGR2HSV)

    # Define range for green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Create a mask for green color
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours in the green mask
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No green area found in the image.")
        return green_img

    # Find the largest contour (assuming it's the green area)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the green area
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Use MediaPipe to perform image segmentation on the original image
    rgb_image = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models', 'selfie_multiclass_256x256.tflite')

    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        output_category_mask=True)

    with ImageSegmenter.create_from_options(options) as segmenter:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        segmentation_result = segmenter.segment(mp_image)
        category_mask = segmentation_result.category_mask.numpy_view()

    # Combine hair (category 1) and face skin (category 3) for the head mask
    head_mask = ((category_mask == 1) | (category_mask == 3)).astype(np.uint8)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    head_mask = cv2.morphologyEx(head_mask, cv2.MORPH_CLOSE, kernel)
    head_mask = cv2.morphologyEx(head_mask, cv2.MORPH_OPEN, kernel)

    head_contours, _ = cv2.findContours(head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if head_contours:
        print('<<<<<< there are', len(head_contours), 'contours >>>>>>')

        # Calculate the center of the green area
        green_center_x = x + w // 2
        green_center_y = y + h // 2

        # Get image dimensions
        img_height, img_width = original_img.shape[:2]

        # Find the head contour with the most overlap with the green area
        max_overlap = 0
        closest_head = None
        face_index = 0
        for i, contour in enumerate(head_contours):
            mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            
            # Ensure we're within image boundaries
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_width, x + w)
            y2 = min(img_height, y + h)
            
            overlap = cv2.countNonZero(cv2.bitwise_and(mask[y1:y2, x1:x2], green_mask[y1:y2, x1:x2]))
            
            if overlap > max_overlap:
                max_overlap = overlap
                closest_head = contour
                face_index = i

            print(f"Contour {i}: Overlap = {overlap}")

        if closest_head is None:
            print("No suitable head contour found.")
            return original_img
        else:
            print('<<<<<< closest head found >>>>>>', face_index)

        fx, fy, fw, fh = cv2.boundingRect(closest_head)

        # Ensure the head bounding rectangle is within image boundaries
        fx = max(0, fx)
        fy = max(0, fy)
        fw = min(fw, img_width - fx)
        fh = min(fh, img_height - fy)

        # Resize head_img_rgba to fit the detected head area
        head_resized = cv2.resize(head_img_rgba, (fw, fh))

        # Split the head_resized into color and alpha channels
        b, g, r, a = cv2.split(head_resized)
        head_rgb = cv2.merge((b, g, r))

        # Create a mask from the alpha channel
        alpha_mask = a.astype(np.float32) / 255.0

        # Blend the head image with the original image
        for c in range(3):  # for each color channel
            original_img[fy:fy+fh, fx:fx+fw, c] = (
                original_img[fy:fy+fh, fx:fx+fw, c] * (1 - alpha_mask) + 
                head_rgb[:, :, c] * alpha_mask
            )

    # Convert the result back to uint8
    result_img = original_img.astype(np.uint8)
    print('<<<<<<<< Image saved >>>>>>>>>')
    cv2.imwrite('./uploads/output.jpg', result_img)

    return result_img

if __name__ == "__main__":
    face_img, face_mask = detect_and_extract_face('/Users/jiaenchua/Desktop/face-detection/uploads/face_image.jpg')
    original_img = cv2.imread('/Users/jiaenchua/Desktop/face-detection/uploads/original_input_image.jpg')
    green_img = cv2.imread('/Users/jiaenchua/Desktop/face-detection/uploads/input_image.png')
    
    if original_img is None or green_img is None:
        print("Error: Unable to read one or more input images")
    else:
        result_img = replace_green_circle(original_img, green_img, face_img, face_mask)