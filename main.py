import json
from flask import Flask, request, jsonify
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = tf.keras.models.load_model('model/PlantPal_v2.h5')

# Load class labels from classes.txt
def load_classes():
    classes = {}
    with open('classes.txt', 'r') as f:
        for line in f:
            class_name, class_index = line.strip().split()
            classes[int(class_index)] = class_name  # Use int() to ensure the index is an integer
    return classes

# Set image dimensions
img_height, img_width = 224, 224

# Class indices mapping
classes = load_classes()

# Function to preprocess and predict the image
def predict_image(image_path):
    # Load image and preprocess it for model prediction
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Get prediction from the model
    prediction = model.predict(img_array)

    print(f"Raw Prediction: {prediction}")  # Print raw prediction

    # Check if the prediction is valid (not -1)
    predicted_class_index = np.argmax(prediction, axis=1)[0]

    print(f"Predicted Class Index: {predicted_class_index}")  # Debug line
    print(f"Classes Dictionary: {classes}")  # Debug line

    try:
        predicted_class_label = classes[predicted_class_index]
        confidence = prediction[0][predicted_class_index] * 100
    except KeyError as e:
        print(f"KeyError: The predicted class index {predicted_class_index} is not in the classes dictionary")
        raise e

    print(f"Predicted class label: {predicted_class_label}")
    print(f"Confidence: {confidence:.2f}%")
    os.remove(image_path)

    # Load details.json and fetch details for the predicted class
    details_path = 'details.json'
    if not os.path.exists(details_path):
        raise FileNotFoundError(f"Details file '{details_path}' not found")
    with open(details_path, 'r') as file:
            details = json.load(file)
    # Retrieve class-specific details
    if predicted_class_label not in details:
        raise KeyError(f"The predicted class label '{predicted_class_label}' is not in the details JSON")
    
    class_details = details[predicted_class_label]

    print(f"Class details: {class_details}")

    return predicted_class_label, confidence, class_details

# API route to upload the image
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file found"}), 400

    image = request.files['image']
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    # Predict the uploaded image
    predicted_class_label, confidence, class_details = predict_image(image_path)

    # Return the prediction result
    return jsonify({
        "message": "Image uploaded and processed successfully",
        "predicted_class": predicted_class_label,
        "confidence": f"{confidence:.2f}%",
        "class_details": class_details,
        "image_path": image_path
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)