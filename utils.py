import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import  Model
from scipy.spatial import distance
from PIL import Image



# Model Defining
def get_extract_model():
    vgg16_model = VGG16(weights="imagenet")
    extract_model = Model(inputs=vgg16_model.inputs, outputs = vgg16_model.get_layer("fc1").output)
    return extract_model

# Image Preprocessing, image to tensor
def image_preprocess(img):
    img = img.resize((224,224)) # VGG16 constraint
    img = img.convert("RGB")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def extract_vector(model, image_path, verbose=0):
    if verbose != 0:
        print("Extracting: ", image_path)
    img = Image.open(image_path)
    img_tensor = image_preprocess(img)

    # Features extraction
    vector = model.predict(img_tensor)[0]
    # Vector normalization
    vector = vector / np.linalg.norm(vector)
    return vector