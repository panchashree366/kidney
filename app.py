
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model("kidney_model.keras")

classes = ['Cyst', 'Normal', 'Stone', 'Tumor']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files:
        return render_template('index.html')

    file = request.files['image']

    if file.filename == '':
        return render_template('index.html')

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = cv2.imread(filepath)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    index = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)

    result = classes[index]

    return render_template(
        'index.html',
        prediction=result,
        confidence=round(confidence, 2),
        image_path=filepath
    )


if __name__ == '__main__':
    app.run(debug=True)

