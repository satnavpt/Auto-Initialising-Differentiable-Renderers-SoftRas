import os
import re
from PIL import Image
import numpy as np
import segment as seg

segment = True
meshPred = False
cameraPred = False

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

# take a folder of images as input

input_folder = 'data/aloi/png2/1'
images = []
for image in sorted_alphanumeric(os.listdir(input_folder)):
    im_file = os.path.join(input_folder, image)
    images.append(np.array(Image.open(im_file).resize((256,256)).convert('RGB')))
images = np.array(images)

# create a set of camera paremeters

cameras = []
for i in range(72):
    cameras.append([2.732, 0., i*5.])
cameras = np.array(cameras).astype('float32')

# estimate viewpoints

if cameraPred:
    pass

# segment images to produce silhouettes

if segment:
    s = seg.Segment()
    images = s.segmentMany(images)

# generate mesh prediction for silhouettes

if meshPred:
    pass

# pass images, mesh init and viewpoints to softras to generate mesh

