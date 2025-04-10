import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
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

# Define paths
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the fine-tuned models with custom_objects
custom_objects = {'Cast': Cast}
vgg16_model = load_model('models/vgg16_waste_classification_tf.h5', custom_objects=custom_objects)
resnet50_model = load_model('models/recyclable_classifier_model.h5', custom_objects=custom_objects)
inceptionv3_model = load_model('models/inceptionv3_waste_classification_tf.h5', custom_objects=custom_objects)

# Dictionary to map model names to their instances
models_dict = {
    'VGG16': vgg16_model,
    'ResNet50': resnet50_model,
    'InceptionV3': inceptionv3_model
}

# Function to preprocess the uploaded image
def preprocess_image(image_path, model_name):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Apply model-specific preprocessing
    if model_name == 'ResNet50':
        img_array = preprocess_input(img_array)  # ResNet50-specific preprocessing
    else:
        img_array = img_array / 255.0  # Normalize for VGG16 and InceptionV3
    return img_array

# Function to predict using a specific model
def predict_with_model(model, image_path, model_name):
    img_array = preprocess_image(image_path, model_name)
    pred = model.predict(img_array)[0][0]
    label = 1 if pred >= 0.5 else 0
    class_name = "Recyclable Spotted" if label == 1 else "Organic Spotted"
    confidence = pred if label == 1 else 1 - pred
    return class_name, confidence

# Function to predict using ensemble (majority voting)
def ensemble_predict(image_path):
    # Preprocess for each model
    img_array_resnet = preprocess_image(image_path, 'ResNet50')
    img_array_others = preprocess_image(image_path, 'VGG16')  # Same preprocessing for VGG16 and InceptionV3
    
    # Get predictions from each model
    vgg16_pred = vgg16_model.predict(img_array_others)[0][0]
    resnet50_pred = resnet50_model.predict(img_array_resnet)[0][0]
    inceptionv3_pred = inceptionv3_model.predict(img_array_others)[0][0]
    
    # Convert probabilities to binary predictions (threshold = 0.5)
    vgg16_label = 1 if vgg16_pred >= 0.5 else 0
    resnet50_label = 1 if resnet50_pred >= 0.5 else 0
    inceptionv3_label = 1 if inceptionv3_pred >= 0.5 else 0
    
    # Majority voting
    votes = [vgg16_label, resnet50_label, inceptionv3_label]
    final_label = 1 if sum(votes) >= 2 else 0
    final_class = "Recyclable Spotted" if final_label == 1 else "Organic Spotted"
    
    # Average probability for display
    avg_prob = (vgg16_pred + resnet50_pred + inceptionv3_pred) / 3
    confidence = avg_prob if final_label == 1 else 1 - avg_prob
    
    return final_class, confidence

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected")
        
        # Get the selected model
        selected_model = request.form.get('model')
        if not selected_model:
            return render_template('index.html', error="Please select a model")

        if file:
            # Save the uploaded file
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction based on selected model
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
                                     confidence=f"{confidence:.2%}", 
                                     image_path=f"uploads/{filename}",
                                     selected_model=selected_model)
            except Exception as e:
                return render_template('index.html', error=f"Error processing image: {str(e)}")
    
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', debug=True)  # Changed from app.run(debug=True)