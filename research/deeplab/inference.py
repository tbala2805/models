from io import BytesIO
import tarfile
# import tempfile
from six.moves import urllib
 
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import dtlpy as dl
import os
import shutil
import cv2

 
# %tensorflow_version 1.x
import tensorflow as tf

# print(tf.__version__)

# if dl.token_expired():
#     dl.login()

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  # Input name of the exported model
  #The actual input tensor name to use during inference is 'ImageTensor:0'.
  INPUT_TENSOR_NAME = 'ImageTensor:0'
  #We are going to use this Output_tensor_name to get required output which is a list
  #with two elements 1st one is class probabilties 'SemanticProbabilities' per pixel and 2nd one is class per pixel'SemanticPredictions'.

  OUTPUT_TENSOR_NAME = ['SemanticProbabilities:0','SemanticPredictions:0']
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph.pb'
  
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


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3
  print("shift",shift)
  print("channel",channel)
  #print("colormap",colormap)
  return colormap


def label_to_color_image(label):
 
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

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
  cv2.imwrite(prediction_path,opencvImage) 


LABEL_NAMES = np.asarray([
   'Background','Priority I - Tree', 'Field - Untilled-unplanted/harvested', 'Machine - Tractor', 'Bale' ,'Windrow'

])
 
FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
#print(FULL_LABEL_MAP)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
print(type(FULL_COLOR_MAP))

#Weight file path

model_path='models/research/deeplab/datasets/ADE20K/exp/train_on_train_set/exp/frozen_inference_graph.pb'

MODEL = DeepLabModel('frozen_inference_graph.pb')

# print('model loaded successfully!')

valid_images_ext=[".jpg",".png",".jpeg"]
list_of_dirs=[]

#Input image path
list_of_dirs.append("datasets/images/")

Image_path=[]
for check_dir in list_of_dirs:
  for file in os.listdir(check_dir):
    print(file)
    for ext in valid_images_ext:
      if file.endswith(ext):
        Image_path.append(os.path.join(check_dir,file))
print(Image_path)


def run_visualization(path,save_path):
  """Inferences DeepLab model and visualizes result."""
  try:
    original_im = Image.open(path)
  except IOError:
    print('Cannot retrieve image. Please check path: ' + path)
    return

  print('running deeplab on image ' )
  resized_im, seg_map = MODEL.run(original_im)

  vis_segmentation(MODEL,resized_im, seg_map,save_path)

#output path
out_dir=".output/"


if os.path.exists(out_dir):
  shutil.rmtree(out_dir)
  os.makedirs(out_dir)
else:
  os.makedirs(out_dir)
for path in Image_path[:]:
  filename=os.path.basename(path)
  run_visualization(path,out_dir + filename)
