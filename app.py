from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import base64
from detect_and_extract_face import detect_and_extract_face
from green_circle_replacement import replace_green_circle
from replace_head_no_mask import replace_head_no_mask
from generate_face_image_with_prompt import generate_face_image_with_prompt
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def base64_to_cv2(base64_string):
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def pil_to_cv2(pil_image):
    # Convert PIL Image to cv2 image
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

@app.route('/face_swap', methods=['POST'])
def face_swap():
    print("Received request>>>>>>>>>")

    input_image_base64 = request.form['input_image']
    original_input_image_base64 = request.form['original_input_image']
    green_color_code = request.form.get('greenColorCode', '#00FF00')  # Default to pure green if not provided
    imagePrompt = request.form['imagePrompt']

    face_image_pil = generate_face_image_with_prompt(imagePrompt)
    if face_image_pil:
        if isinstance(face_image_pil, str):
            return face_image_pil, 400  # Return error message if generation failed
        
        face_image_cv2 = pil_to_cv2(face_image_pil)
        input_image = base64_to_cv2(input_image_base64)
        original_input_image = base64_to_cv2(original_input_image_base64)
        
        # Check input image is valid
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'input_image.png'), input_image)
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'original_input_image.png'), original_input_image)

        face_img, face_mask = detect_and_extract_face(face_image_cv2)
        if face_img is not None and face_mask is not None:
            result_img, message = replace_green_circle(original_input_image, input_image, face_img, face_mask, green_color_code, app)
            if message is None:
                # Convert the result image to bytes
                _, img_encoded = cv2.imencode('.png', result_img)
                img_bytes = img_encoded.tobytes()
                
                # Return the image bytes directly
                return img_bytes, 200, {'Content-Type': 'image/png'}
            else:
                print(message)
                return message, 400
        else:
            return 'Face extraction failed', 400

    return 'Invalid file type', 400

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)