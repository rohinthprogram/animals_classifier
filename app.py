from flask import Flask, request

import json
import numpy as np
from keras.models import load_model
import os

import base64
from PIL import Image
from io import BytesIO
from skimage.transform import resize

app = Flask(__name__)

classes = list(dict(json.load(open('translation.json'))).values())

def convert_base64_to_image(image_str, return_type='numpy'):
    '''
    Converts a base64 encoded image to Pillow Image or Numpy Array
    
    Args:
        image_str (str): The pure base64 encoded string of the image
        return_type (str): The type of image you want to convert it to. 
                           Choices are [ numpy | pillow ]. Default is numpy.
    Returns:
        PIL.Image or numpy.array: The converted image
    '''
    image = Image.open(BytesIO(base64.b64decode(image_str)))
    if return_type == 'numpy':
        return np.array(image)
    else:
        return image

@app.route('/', methods=['GET', 'POST'])
def home():
    return 'Hello'


@app.route('/classify', methods=['POST'])
def predict():
    #print(request)
    
    imgstr = request.json['img']
    img = convert_base64_to_image(imgstr)
    img = resize(img, (100, 100))
    
    img = np.reshape(img, (1, 100, 100, 3)) 

    model = load_model('./save/animals_classifier.h5')
    
    score = list(sorted(list(model.predict(img).tolist())[0]))

    return str(score)
    

if __name__ == '__main__':
    app.run(debug=True)
