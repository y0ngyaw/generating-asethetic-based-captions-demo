from flask import Flask, render_template, request, jsonify, send_file, current_app
import os
from PIL import Image
from model.inference import evaluate

app = Flask(__name__, static_url_path='/static')
APP_ROOT = os.path.dirname(os.path.abspath(__file__))   # refers to application_top
APP_STATIC = os.path.join(APP_ROOT, 'static')
FILES_STATIC = os.path.join(APP_STATIC, 'files')


# Pages
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image', methods=['POST'])
def upload_image():
	filename = request.files['file'].name
	img = Image.open(request.files['file'].stream)
	captions = evaluate(img)
	return jsonify({'captions': captions})
