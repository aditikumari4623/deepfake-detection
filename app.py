from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import os

from pretrained_xception import build_pretrained_model

# Initialize Flask app
app = Flask(__name__)

# Upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained model
model = build_pretrained_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Use pre-trained model to detect deepfake
        result = detect_deepfake(filepath, model)

        # Set a threshold for deepfake classification (you can adjust this)
        is_deepfake = result > 0.5

        return render_template('result.html', is_deepfake=is_deepfake)

# Run Flask app
if __name__ == '_main_':
    app.run(debug=True)