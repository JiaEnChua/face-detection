from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
import os
from face_detector import detect_and_extract_face, replace_green_circle, replace_head_no_mask
import cv2
import numpy as np
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
    if 'face_image' not in request.files or 'input_image' not in request.files:
        return 'Missing required files', 400

    face_image = request.files['face_image']
    input_image = request.files['input_image']
    no_green_mask = request.form.get('noGreenMask', 'false').lower() == 'true'

    if face_image.filename == '' or input_image.filename == '':
        return 'No selected file', 400

    if face_image and allowed_file(face_image.filename) and input_image and allowed_file(input_image.filename):
        face_filename = secure_filename(face_image.filename)
        input_filename = secure_filename(input_image.filename)
        
        face_path = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')

        face_image.save(face_path)
        input_image.save(input_path)

        face_img, face_mask = detect_and_extract_face(face_path)
        if face_img is not None and face_mask is not None:
            if no_green_mask:
                replace_head_no_mask(input_path, face_img, face_mask, output_path)
            else:
                replace_green_circle(input_path, face_img, face_mask, output_path)
            return send_file(output_path, mimetype='image/jpeg')
        else:
            return 'Face extraction failed', 400

    return 'Invalid file type', 400

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)