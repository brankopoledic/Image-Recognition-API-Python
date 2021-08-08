from flask import Flask, jsonify, request, make_response
from flask_restful import reqparse, abort, Api, Resource
import requests
from PIL import Image
from io import BytesIO
from haishoku.haishoku import Haishokuu
from colormap import rgb2hex
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from model import OpenNsfw
import numpy as np
from imageai.Detection import ObjectDetection
from imageai.Classification import ImageClassification
import os
import urllib.request

app = Flask(__name__)
api = Api(app)


@app.route("/api", methods=["POST"])
def post():
      json_data = request.get_json(force=True)
      img = json_data['image']

      directory = 'files'
      number_of_files = len([item for item in os.listdir(directory) if os.path.isfile(os.path.join(directory, item))])

      EXTN = ".jpg"
      filename = "files/" + str(number_of_files).zfill(7) + EXTN

      urllib.request.urlretrieve(img, filename)

      im = Image.open(filename)

      width, height = im.size

      palette = Haishokuu.getPalette(filename)

      def objectDetection(filename): 
            execution_path = os.getcwd()      
            detector = ObjectDetection()
            detector.setModelTypeAsYOLOv3()
            detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
            detector.loadModel()
            detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , filename), output_image_path=os.path.join(execution_path , "new.jpg"), minimum_percentage_probability=30)

            objectDetection = []
            for eachObject in detections:
                  objectDetection.append({"label": eachObject["name"], "score": eachObject["percentage_probability"]})

            return objectDetection 

      def classification(filename):
            execution_path = os.getcwd()
            prediction = ImageClassification()
            prediction.setModelTypeAsResNet50()
            prediction.setModelPath(os.path.join(execution_path, "resnet50_imagenet_tf.2.0.h5"))
            prediction.loadModel()
            predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, filename), result_count=5)

            classification = []
            for eachPrediction, eachProbability in zip(predictions, probabilities):
                  classification.append({"label": eachPrediction, "score": eachProbability})

            return classification     

      def security(filename):
            model = OpenNsfw()
            imgs = image.load_img(filename, target_size=(224, 224))
            x = image.img_to_array(imgs)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            preds = model.predict(x)
            sfw = str(preds[0][0])
            nsfw = str(preds[0][1])
            return {'sfw': sfw, 'nsfw': nsfw }

      colors = []
      for color in palette:
            colors.append({"color": rgb2hex(color[1][0], color[1][1], color[1][2]), "score": color[0]})


      return { 
            'width': width, 
            'height': height, 
            'color_palette': colors, 
            'security': security(filename),
            'classification': classification(filename),
            'objectDetection': objectDetection(filename)
      }


if __name__ == "__main__":
    app.run()