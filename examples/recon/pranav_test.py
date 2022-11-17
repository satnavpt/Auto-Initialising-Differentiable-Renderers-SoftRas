import argparse
import torch
import torch.nn.parallel
import torchvision.transforms as T
import datasets
from utils import img_cvt
import soft_renderer as sr
import soft_renderer.functional as srf
import models
import models_large
import time
import os
import imageio
import numpy as np
from PIL import Image

IMAGE_SIZE = 64
MODEL_DIRECTORY = 'data/models/pretrained_model.tar'
INPUT_IMAGE = 'data/custom_dataset/Test1.png'
INPUT_DIRECTORY = None
SIGMA_VAL = 0.01
TIMESTAMP = str(int(time.time()*1e7))
OUTPUT_DIRECTORY = './data/results/pranav_test'

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('-eid', '--experiment-id', type=str)
parser.add_argument('-d', '--model-directory', type=str, default=MODEL_DIRECTORY)
parser.add_argument('-ii', '--input-image', type=str, default=INPUT_IMAGE)
parser.add_argument('-id', '--input-directory', type=str, default=INPUT_DIRECTORY)
parser.add_argument('-od', '--output_directory', type=str, default=OUTPUT_DIRECTORY)

parser.add_argument('-is', '--image-size', type=int, default=IMAGE_SIZE)
parser.add_argument('-sv', '--sigma-val', type=float, default=SIGMA_VAL)

parser.add_argument('--shading-model', action='store_true', help='test shading model')
args = parser.parse_args()

output_dir = os.path.join(args.output_directory, TIMESTAMP)
os.makedirs(output_dir, exist_ok=True)

# setup model & optimizer
if args.shading_model:
    model = models_large.Model('data/obj/sphere/sphere_642.obj', args=args)
else:
    model = models.Model('data/obj/sphere/sphere_642.obj', args=args)
model = model.cuda()
state_dicts = torch.load(args.model_directory)
model.load_state_dict(state_dicts['model'], strict=True)
model.eval()

transform = T.Compose([T.Resize((64,64)), T.ToTensor()])

def test_single(image_path):
    im = Image.open(image_path).convert('RGBA')
    im = transform(im)

    images = torch.unsqueeze(im, 0)
    images = torch.autograd.Variable(images).cuda()

    vertices, faces = model.reconstruct(images)
    vertices, faces = vertices[0], faces[0]
    image = images[0]

    mesh_path = os.path.join(output_dir, 'out.obj')
    input_path = os.path.join(output_dir, 'out.png')
    srf.save_obj(mesh_path, vertices, faces)
    imageio.imsave(input_path, img_cvt(image))

def test_multiple(input_path):
    images = []
    for image_path in os.listdir(input_path):
        image_path = os.path.join(input_path, image_path)
        im = Image.open(image_path).convert('RGBA')
        im = transform(im)
        images.append(im)

    images = torch.stack(images)
    images = torch.autograd.Variable(images).cuda()

    vertices, faces = model.reconstruct(images)

    for k in range(vertices.size(0)):
        mesh_path = os.path.join(output_dir, (str(k) + '.obj'))
        input_path = os.path.join(output_dir, (str(k) + '.png'))
        srf.save_obj(mesh_path, vertices[k], faces[k])
        imageio.imsave(input_path, img_cvt(images[k]))

if args.input_directory is not None:
    test_multiple(args.input_directory)
else:
    test_single(args.input_image)