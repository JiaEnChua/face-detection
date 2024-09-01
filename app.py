from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
import os
import cv2
from face_detector import detect_and_extract_face
from green_circle_replacement import replace_green_circle
from head_replacement import replace_head_no_mask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/face_swap', methods=['POST'])
def face_swap():
    print("Received request>>>>>>>>>")
    if 'face_image' not in request.files or 'input_image' not in request.files or 'original_input_image' not in request.files:
        return 'Missing required files', 400

    face_image = request.files['face_image']
    input_image = request.files['input_image']
    original_input_image = request.files['original_input_image']
    green_color_code = request.form.get('greenColorCode', '#00FF00')  # Default to pure green if not provided

    if face_image.filename == '' or input_image.filename == '' or original_input_image.filename == '':
        return 'No selected file', 400

    if (face_image and allowed_file(face_image.filename) and 
        input_image and allowed_file(input_image.filename) and 
        original_input_image and allowed_file(original_input_image.filename)):
        
        face_filename = secure_filename(face_image.filename)
        input_filename = secure_filename(input_image.filename)
        original_input_filename = secure_filename(original_input_image.filename)
        
        face_path = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        original_input_path = os.path.join(app.config['UPLOAD_FOLDER'], original_input_filename)

        face_image.save(face_path)
        input_image.save(input_path)
        original_input_image.save(original_input_path)
        face_img, face_mask = detect_and_extract_face(face_path)
        if face_img is not None and face_mask is not None:
            result_img, message = replace_green_circle(cv2.imread(original_input_path), cv2.imread(input_path), face_img, face_mask, green_color_code)
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