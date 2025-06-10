import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the new, multi-class model
model = tf.keras.models.load_model('tomato_classifier_model_vgg16_4_classes.h5')

img_width, img_height = 224, 224

class_labels = {0: 'Early Vegetative Stage', 1: 'Flowering Initiation Stage', 2: 'Fruiting and Ripening', 3: 'Germination Stage'}

# Function to preprocess a new image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

# Make a prediction
def predict_stage(img_path):
    processed_image = preprocess_image(img_path)
    prediction = model.predict(processed_image)

    predicted_class_index = np.argmax(prediction[0])

    # Print the result
    print(f"Prediction probabilities: {prediction[0]}")
    print(f"The model predicts this plant is in the: {class_labels[predicted_class_index]}")

new_image_path = './data/Stage2_Flowering_Initiation/8cb7612861fca3b4d336b5f4d65593ee.jpg'

predict_stage(new_image_path)

