from flask import Flask, render_template, request

import requests, json
import numpy as np
from keras.models import load_model
import os

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

classes = list(dict(json.load(open('translation.json'))).values())

def api_call_cellstarthub(img):
    API_KEY = os.environ.get("API_KEY")
    USERNAME = os.environ.get("USERNAME")
    API_NAME = os.environ.get("API_NAME")

    #endpoint = f"https://api.cellstrathub.com/{ USERNAME }/{ API_NAME }"
    #endpoint = f'localhost:5000/predict'
    headers = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
    }

    payload = {'img':img.tolist()}

    # make a get request to load the model (needed if calling api after long time)
    # print(requests.get(endpoint, headers=headers).json())

    # Send POST request to get the output
    response = requests.post(endpoint, headers=headers, json=payload).json()

    print(response)
    return response['output']


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        img = request.files['img']
        imgstr = img.read()
        img = np.fromstring(imgstr, np.uint8)
        print(img)
        output = api_call_cellstarthub(img)
        return render_template('home.html', output=output)

    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    print(request)
    img = request
    
    model = load_model('./save/animals_classifier.h5')
    score = list(model.predict(img).tolist())[0]
    label = classes[score.index(max(score))]

    return label

if __name__ == '__main__':
    app.run(debug=True)