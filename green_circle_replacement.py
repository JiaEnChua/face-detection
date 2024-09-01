import cv2
import numpy as np
import mediapipe as mp
import os

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def find_green_area(green_img, green_color_code):
    # Convert green_img to HSV color space
    hsv = cv2.cvtColor(green_img, cv2.COLOR_BGR2HSV)

    # Convert the hex color code to RGB
    rgb_color = hex_to_rgb(green_color_code)
    # Convert RGB to HSV
    hsv_color = cv2.cvtColor(np.uint8([[rgb_color]]), cv2.COLOR_RGB2HSV)[0][0]

    # Define range for the specified color in HSV
    lower_color = np.array([max(0, hsv_color[0] - 10), 50, 50])
    upper_color = np.array([min(179, hsv_color[0] + 10), 255, 255])

    # Create a mask for the specified color
    color_mask = cv2.inRange(hsv, lower_color, upper_color)

    # Find contours in the color mask
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No area with the specified color found in the image.")
        return None

    # Find the largest contour (assuming it's the target area)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the target area
    x, y, w, h = cv2.boundingRect(largest_contour)

    return x, y, w, h


def find_head_mask(original_img, x, y, w, h):
    # Crop the original image to the area of interest
    roi = original_img[y:y+h, x:x+w]
    
    # Use MediaPipe to perform image segmentation on the cropped image
    rgb_image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

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

    # Create a full-size mask with zeros
    full_mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
    
    # Place the head mask in the correct position within the full-size mask
    full_mask[y:y+h, x:x+w] = head_mask

    return full_mask

def apply_head_to_image(original_img, head_img_rgba, face_mask, x, y, w, h):
    # Resize head_img_rgba to fit the detected head area
    head_resized = cv2.resize(head_img_rgba, (w, h))

    # Split the head_resized into color and alpha channels
    b, g, r, a = cv2.split(head_resized)
    head_rgb = cv2.merge((b, g, r))

    # Create a mask from the alpha channel and the face mask
    alpha_mask = a.astype(np.float32) / 255.0
    combined_mask = alpha_mask * (face_mask.astype(np.float32) / 255.0)

    # Blend the head image with the original image
    for c in range(3):  # for each color channel
        original_img[y:y+h, x:x+w, c] = (
            original_img[y:y+h, x:x+w, c] * (1 - combined_mask) + 
            head_rgb[:, :, c] * combined_mask
        )

    return original_img

def replace_green_circle(original_img, green_img, head_img_rgba, face_mask, green_color_code):
    # Check if inputs are valid NumPy arrays
    if not isinstance(original_img, np.ndarray) or not isinstance(green_img, np.ndarray):
        print("Error: Input images must be NumPy arrays")
        return None

    # Step 1: Find the area with the specified color
    color_area = find_green_area(green_img, green_color_code)
    if color_area is None:
        return green_img

    x, y, w, h = color_area
    print(f"Color area: x={x}, y={y}, w={w}, h={h}")

    # Step 2: Find the best possible head near the color area
    head_mask = find_head_mask(original_img, x, y, w, h)
    
    # Find contours in the head mask
    contours, _ = cv2.findContours(head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No head found in the image.")
        return original_img

    # Find the closest head to the color area
    closest_head = min(contours, key=lambda c: 
        abs(cv2.boundingRect(c)[0] - x) + abs(cv2.boundingRect(c)[1] - y))
    
    hx, hy, hw, hh = cv2.boundingRect(closest_head)
    print(f"Closest head: x={hx}, y={hy}, w={hw}, h={hh}")

    # Calculate the distance between the color area and the closest head
    distance = ((hx - x)**2 + (hy - y)**2)**0.5
    print(f"Distance between color area and closest head: {distance:.2f} pixels")

    # Step 3: Resize face_mask to match the head dimensions
    face_mask_resized = cv2.resize(face_mask, (hw, hh))

    # Step 4: Resize head_img_rgba to match the head dimensions
    head_img_resized = cv2.resize(head_img_rgba, (hw, hh))

    # Step 5: Apply the resized head to the original image
    result_img = apply_head_to_image(original_img, head_img_resized, face_mask_resized, hx, hy, hw, hh)

    # Optionally, draw rectangles on the result image to visualize the areas
    cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Color area
    cv2.rectangle(result_img, (hx, hy), (hx+hw, hy+hh), (0, 0, 255), 2)  # Closest head
    cv2.imwrite('./uploads/output_temp.png', result_img)

    return result_img.astype(np.uint8)

if __name__ == "__main__":
    face_img = cv2.imread('/Users/jiaenchua/Desktop/face-detection/uploads/face_image.png', cv2.IMREAD_UNCHANGED)
    original_img = cv2.imread('/Users/jiaenchua/Desktop/face-detection/uploads/original_input_image.jpg')
    green_img = cv2.imread('/Users/jiaenchua/Desktop/face-detection/uploads/input_image.png')
    green_color_code = '#00FF00'  # Example color code
    
    if original_img is None or green_img is None or face_img is None:
        print("Error: Unable to read one or more input images")
    else:
        result_img = replace_green_circle(original_img, green_img, face_img, face_img, green_color_code)
        cv2.imwrite('./uploads/output.png', result_img)
        print("Image processing complete. Output saved as 'output.png'")