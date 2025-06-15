from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import base64

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Load model
model = load_model(r'Model\best_model.h5')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def predict_pneumonia(file_stream):
    # Load image directly from file stream
    img = Image.open(io.BytesIO(file_stream))
    img = img.convert('RGB')  # Ensure 3 channels
    img = img.resize((256, 256))
    
    # Convert to numpy array and preprocess
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    # Make prediction
    prediction = model.predict(img_array)
    return prediction[0][0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('home'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('home'))
    
    if file and allowed_file(file.filename):
        try:
            # Read file into memory without saving
            file_stream = file.read()
            
            # Make prediction
            prediction = predict_pneumonia(file_stream)
            result = "Pneumonia Detected" if prediction > 0.5 else "Normal"
            confidence = round(prediction * 100, 2) if result == "Pneumonia Detected" else round((1 - prediction) * 100, 2)
            
            # Convert back to bytes for temporary display
            img_bytes = io.BytesIO()
            Image.open(io.BytesIO(file_stream)).save(img_bytes, format='JPEG')
            img_bytes = img_bytes.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            return render_template('result.html', 
                      result=result,
                      confidence=confidence,
                      img_data=img_base64)
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('home'))
    else:
        flash('Allowed file types are: png, jpg, jpeg')
        return redirect(url_for('home'))

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)