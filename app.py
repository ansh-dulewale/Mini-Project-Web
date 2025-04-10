import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.layers import Layer
import tensorflow as tf
from PIL import Image

# Define a custom Cast layer
class Cast(Layer):
    def __init__(self, dtype='float32', **kwargs):
        super(Cast, self).__init__(**kwargs)
        self._dtype = dtype

    def call(self, inputs):
        return tf.cast(inputs, dtype=self._dtype)

    def get_config(self):
        config = super(Cast, self).get_config()
        config.update({'dtype': self._dtype})
        return config

    @property
    def dtype(self):
        return self._dtype

# Initialize Flask app
app = Flask(__name__)

# Define paths relative to the app root
UPLOAD_FOLDER = os.path.join(app.root_path, 'static/uploads/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the fine-tuned models with custom_objects
custom_objects = {'Cast': Cast}
try:
    vgg16_model = load_model(os.path.join(app.root_path, 'models/vgg16_waste_classification_tf.h5'), custom_objects=custom_objects)
    resnet50_model = load_model(os.path.join(app.root_path, 'models/recyclable_classifier_model.h5'), custom_objects=custom_objects)
    inceptionv3_model = load_model(os.path.join(app.root_path, 'models/inceptionv3_waste_classification_tf.h5'), custom_objects=custom_objects)
except Exception as e:
    print(f"Error loading models: {e}")
    raise

# Dictionary to map model names to their instances
models_dict = {
    'VGG16': vgg16_model,
    'ResNet50': resnet50_model,
    'InceptionV3': inceptionv3_model
}

# Function to preprocess the uploaded image
def preprocess_image(image_path, model_name):
    try:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        if model_name == 'ResNet50':
            return preprocess_input(img_array)
        elif model_name == 'VGG16':
            return vgg_preprocess(img_array)
        elif model_name == 'InceptionV3':
            return inception_preprocess(img_array)
        else:
            raise ValueError("Unsupported model for preprocessing")
    except Exception as e:
        raise Exception(f"Preprocessing error: {e}")

# Function to predict using a specific model
def predict_with_model(model, image_path, model_name):
    try:
        img_array = preprocess_image(image_path, model_name)
        pred = model.predict(img_array)[0][0]
        label = 1 if pred >= 0.5 else 0
        class_name = "Recyclable Spotted" if label == 1 else "Organic Spotted"
        confidence = pred if label == 1 else 1 - pred
        return class_name, confidence
    except Exception as e:
        raise Exception(f"Prediction error: {e}")

# Function to predict using ensemble (majority voting)
def ensemble_predict(image_path):
    try:
        vgg16_pred, _ = predict_with_model(vgg16_model, image_path, 'VGG16')
        resnet50_pred, _ = predict_with_model(resnet50_model, image_path, 'ResNet50')
        inceptionv3_pred, _ = predict_with_model(inceptionv3_model, image_path, 'InceptionV3')

        vgg16_label = 1 if vgg16_pred == "Recyclable Spotted" else 0
        resnet50_label = 1 if resnet50_pred == "Recyclable Spotted" else 0
        inceptionv3_label = 1 if inceptionv3_pred == "Recyclable Spotted" else 0

        votes = [vgg16_label, resnet50_label, inceptionv3_label]
        final_label = 1 if sum(votes) >= 2 else 0
        final_class = "Recyclable Spotted" if final_label == 1 else "Organic Spotted"

        avg_prob = (float(str(vgg16_pred).split(': ')[1].rstrip('%')) + 
                   float(str(resnet50_pred).split(': ')[1].rstrip('%')) + 
                   float(str(inceptionv3_pred).split(': ')[1].rstrip('%'))) / 3 / 100
        confidence = avg_prob if final_label == 1 else 1 - avg_prob

        return final_class, f"{confidence:.2%}"
    except Exception as e:
        raise Exception(f"Ensemble prediction error: {e}")

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected")

        selected_model = request.form.get('model')
        if not selected_model:
            return render_template('index.html', error="Please select a model")

        if file and file.filename:
            filename = file.filename if file.filename != 'captured_image.jpg' else 'captured_image.jpg'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                if selected_model == 'Ensemble':
                    predicted_class, confidence = ensemble_predict(filepath)
                else:
                    model = models_dict.get(selected_model)
                    if not model:
                        return render_template('index.html', error="Invalid model selected")
                    predicted_class, confidence = predict_with_model(model, filepath, selected_model)
                
                return render_template('index.html', 
                                     prediction=predicted_class, 
                                     confidence=confidence,
                                     image_path=f"uploads/{filename}",
                                     selected_model=selected_model)
            except Exception as e:
                return render_template('index.html', error=f"Error processing image: {str(e)}")
    
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ.get('PORT', 5000))  # Use Render's PORT or default to 5000
    app.run(host='0.0.0.0', port=port, debug=True)