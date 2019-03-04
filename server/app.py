from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from fastai.vision import *
import base64
from PIL import Image
from io import BytesIO
import time

import json
app = Flask(__name__)
CORS(app)

defaults.device = torch.device('cpu')
learn = load_learner('./')

@app.route('/api/v1/classify', methods=['POST'])
def classifyEndpoint():
  if request.method == 'POST':
      """ Receive base 64 encoded image """
      print('Request received')
      request_data = json.loads(request.get_data())
      data = request_data['data'][22:]

      probability = classify(data)
      print(probability)
      response = {
        'nsfw': probability,
        'sfw': 1 - probability
      }
      return jsonify(response)

# Takes an image in bytes and returns the probability that it is NSFW
def classify(data):
  im = Image.open(BytesIO(base64.b64decode(data)))
  im.save('temp.png', 'PNG')

  img = open_image('temp.png')
  pred_class,pred_idx,outputs = learn.predict(img)

  return outputs[0].item()
