from io import BytesIO
import tarfile
# import tempfile
from six.moves import urllib
import io
 
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import dtlpy as dl
import os
import shutil
import cv2
from flask import Flask, flash, redirect, render_template, request, send_file, send_from_directory
import base64
 
# %tensorflow_version 1.x
import tensorflow as tf

from flask import Flask, request, jsonify
from werkzeug import run_simple

from prefix_manager import manage_prefixes

import numpy as np
import json

# print(tf.__version__)

# if dl.token_expired():
#     dl.login()
data= {
    "2": {
        "gray_scale_value": 2,
        "label_name": "Machine - Tractor",
        "label_colour": "#1e51fa"
    },
    "1": {
        "gray_scale_value": 1,
        "label_name": "Priority I - Tree",
        "label_colour": "#727501"
    },
    "0": {
        "gray_scale_value": 0,
        "label_name": "Background",
        "label_colour": "#3bd993"
    },
    "5": {
        "gray_scale_value": 5,
        "label_name": "Field - Untilled-unplanted/harvested",
        "label_colour": "#3db126"
    },
    "3": {
        "gray_scale_value": 3,
        "label_name": "Bale",
        "label_colour": "#6e4801"
    },
    "4": {
        "gray_scale_value": 4,
        "label_name": "Windrow",
        "label_colour": "#b3a844"
    }
}
ALLOWED_EXTENSIONS = {'jpeg', 'jpg', 'png'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  # Input name of the exported model
  #The actual input tensor name to use during inference is 'ImageTensor:0'.
  INPUT_TENSOR_NAME = 'ImageTensor:0'
  #We are going to use this Output_tensor_name to get required output which is a list
  #with two elements 1st one is class probabilties 'SemanticProbabilities' per pixel and 2nd one is class per pixel'SemanticPredictions'.

  OUTPUT_TENSOR_NAME = ['SemanticProbabilities:0','SemanticPredictions:0']
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph_1000.pb'
  
  def __init__(self, frozen_graph_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
   
    with tf.io.gfile.GFile(frozen_graph_path, "rb") as f:
      graph_def = tf.compat.v1.GraphDef.FromString(f.read())

    # if graph_def is None:
    #   raise RuntimeError('Cannot find inference graph in tar archive.')
    if graph_def is None:
     raise RuntimeError('Cannot find inference graph ')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')
    tf.compat.v1.disable_eager_execution()
    self.sess = tf.compat.v1.Session(graph=self.graph)

  def run(self, image):
    
    tf.compat.v1.disable_eager_execution()
    with tf.compat.v1.Session(graph=self.graph) as sess:
      width, height = image.size
      resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
      self.resize_ratio=resize_ratio
      target_size = (int(resize_ratio * width), int(resize_ratio * height))
      resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
      batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
      
      prob_seg_map = batch_seg_map[0][0]
      pred_seg_map = batch_seg_map[1][0]    
      # print(pred_seg_map)
      with open('class_probabilties.txt','w') as f:
        for x in prob_seg_map:
          np.savetxt(f, x)

      return resized_image, pred_seg_map
     
def create_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = []
 
  
#   with open(metadata_file_path,'r') as read_json:
#   data = json.load(meta_json)
  for i in range(len(data)):
   label_data = data[str(i)]
   hex = label_data["label_colour"]
   hex =  hex.lstrip('#')
          #map = []
          #map.append(list(int(hex[i:i+2], 16) for i in (0, 2, 4)))
   map = list(int(hex[i:i+2], 16) for i in (0, 2, 4))
   colormap.append(map)
  colormap = np.array(colormap)
          
  return colormap


# def create_pascal_label_colormap():
#   """Creates a label colormap used in PASCAL VOC segmentation benchmark.

#   Returns:
#     A Colormap for visualizing segmentation results.
#   """
#   colormap = np.zeros((256, 3), dtype=int)
#   ind = np.arange(256, dtype=int)

#   for shift in reversed(range(8)):
#     for channel in range(3):
#       colormap[:, channel] |= ((ind >> channel) & 1) << shift
#     ind >>= 3
#   print("shift",shift)
#   print("channel",channel)
#   #print("colormap",colormap)
#   return colormap


def label_to_color_image(label):
 
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def vis_segmentation(MODEL,image,seg_map,prediction_path=None):
  """Visualizes input image, segmentation map and overlay view."""
  
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  width,height=image.size
  seg_image=Image.fromarray(np.uint8(seg_image)).convert('RGB')
  resize_ratio = 1.0 / MODEL.resize_ratio
  target_size = (int(resize_ratio * width), int(resize_ratio * height))
  # print(final,target_size)
  resized_image = seg_image.resize(target_size, Image.ANTIALIAS)
  # print(type(resized_image))
  opencvImage = cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2BGR)
  #cv2.imwrite(prediction_path,opencvImage)

  return opencvImage


LABEL_NAMES = np.asarray([
 'Background','Priority I - Tree', 'Field - Untilled-unplanted/harvested', 'Machine - Tractor', 'Bale' ,'Windrow'

])
 
FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
#print(FULL_LABEL_MAP)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
print(type(FULL_COLOR_MAP))

#Weight file path
MODEL = None

# print('model loaded successfully!')

valid_images_ext=[".jpg",".png",".jpeg"]
list_of_dirs=[]


def run_visualization(original_im,save_path):
  """Inferences DeepLab model and visualizes result."""
  #try:
  #  original_im = Image.open(path)
  #except IOError:
  #  print('Cannot retrieve image. Please check path: ' + path)
  #  return

  print('running deeplab on image ' )
  resized_im, seg_map = MODEL.run(original_im)

  result = vis_segmentation(MODEL,resized_im, seg_map,save_path)
  return result

#output path
out_dir=".output/"

# Location of our model
model_path = 'frozen_inference_graph_1000.pb'

# create and configure the app
app = Flask(__name__, instance_relative_config=True)

app.config.from_mapping(
        SECRET_KEY='secret_for_client_session'
    )

# We create a route in our app to create a response for all requests on '/'
@app.route('/', methods=(['GET', 'POST']))

def predict():
  print("request")
  if request.method == 'POST':
    print("post")
    uploaded_files = request.files.getlist("image")

    global MODEL
    if not MODEL:
      print("loaded model")
      MODEL = DeepLabModel(model_path)

    images = []

    for upload in uploaded_files :
      # Make sure the file is one of the allowed filetypes
      if not allowed_file(upload.filename) :
          flash('One or multiple files was in a unsupported format.')
          return render_template('index.html')

      img = Image.open(upload.stream)
      img.load()

      # Use our model to predict the class of the file sent over a form.
      img = run_visualization(img, 'response.jpeg')
      img_str = cv2.imencode('.jpeg', img)[1].tostring()
      img_base64 = base64.b64encode(img_str).decode('ascii')

      images.append(img_base64)

    return render_template('index.html', images=images)
  else :
    return render_template('index.html')

app = manage_prefixes(app)

if __name__ == '__main__':
    run_simple("0", 8000, use_reloader=True, application=app)
    
