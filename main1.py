from flask import Flask, request
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_image():
    print("Request received");
    if 'image' not in request.files:
        return {"error": "No image file found"}, 400

    image = request.files['image']
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)
    print("Image uploaded successfully")
    return {"message": "Image uploaded successfully", "path": image_path}
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)