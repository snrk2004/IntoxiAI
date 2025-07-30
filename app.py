from flask import Flask, render_template, request, redirect, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2
import os
import base64

app = Flask(__name__)

# Load your model
model = load_model(r'D:\IntoxiAI_app\IntoxiAI_app\drunk_sober_mobilenet_categorical.h5')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

def preprocess_image_array(img_array):
    img_array = cv2.resize(img_array, (224, 224))
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype('float32') / 255.0  # Normalize to [0, 1]
    return img_array

def predict_image(model, img_array):
    predictions = model.predict(img_array)
    return predictions

# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            filepath = os.path.join("static", file.filename)
            file.save(filepath)
            img_array = preprocess_image(filepath)
        elif "captured_image" in request.form and request.form["captured_image"] != "":
            img_data = request.form["captured_image"]
            try:
                img_data = img_data.split(",")[1]
                img_data = base64.b64decode(img_data)
                img_array = np.frombuffer(img_data, np.uint8)
                img_array = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img_array = preprocess_image_array(img_array)
            except IndexError:
                return redirect(request.url)
        else:
            return redirect(request.url)

        predictions = predict_image(model, img_array)
        class_labels = {0: 'DRUNK', 1: 'SOBER'}
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_labels[predicted_class]
        confidence = predictions[0][predicted_class] * 100

        return render_template("index.html", label=predicted_label, confidence=confidence)
    return render_template("index.html", label=None, confidence=None)

@app.route("/detect_face", methods=["POST"])
def detect_face():
    if "image" not in request.json:
        return jsonify({"error": "No image provided"}), 400

    img_data = request.json["image"]
    
    # Add padding to the base64 string if necessary
    img_data += "=" * ((4 - len(img_data) % 4) % 4)

    try:
        img_data = base64.b64decode(img_data)
    except Exception as e:
        return jsonify({"error": "Invalid base64 image data"}), 400

    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        return jsonify({"face_detected": True})
    else:
        return jsonify({"face_detected": False})

if __name__ == "__main__":
    app.run(debug=True)
