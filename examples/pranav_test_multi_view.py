import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import tqdm
import numpy as np
import imageio
import argparse

# deal with images
import PIL
from PIL import Image
import soft_renderer as sr
import re

# import segment as seg

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '../data')

class Model(nn.Module):
    def __init__(self, template_path):
        super(Model, self).__init__()

        # set template mesh
        self.template_mesh = sr.Mesh.from_obj(template_path)
        self.register_buffer('vertices', self.template_mesh.vertices * 0.5)
        self.register_buffer('faces', self.template_mesh.faces)
        self.register_buffer('textures', self.template_mesh.textures)

        # optimize for displacement map and center
        self.register_parameter('displace', nn.Parameter(torch.zeros_like(self.template_mesh.vertices)))
        self.register_parameter('center', nn.Parameter(torch.zeros(1, 1, 3)))

        # define Laplacian and flatten geometry constraints
        self.laplacian_loss = sr.LaplacianLoss(self.vertices[0].cpu(), self.faces[0].cpu())
        self.flatten_loss = sr.FlattenLoss(self.faces[0].cpu())

    def forward(self, batch_size):
        base = torch.log(self.vertices.abs() / (1 - self.vertices.abs()))
        centroid = torch.tanh(self.center)
        vertices = torch.sigmoid(base + self.displace) * torch.sign(self.vertices)
        vertices = F.relu(vertices) * (1 - centroid) - F.relu(-vertices) * (centroid + 1)
        vertices = vertices + centroid

        # apply Laplacian and flatten geometry constraints
        laplacian_loss = self.laplacian_loss(vertices).mean()
        flatten_loss = self.flatten_loss(vertices).mean()

        return sr.Mesh(vertices.repeat(batch_size, 1, 1),
                       self.faces.repeat(batch_size, 1, 1)), laplacian_loss, flatten_loss


def neg_iou_loss(predict, target):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6
    return 1. - (intersect / union).sum() / intersect.nelement()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename-input', type=str,
                        default=os.path.join(data_dir, 'aloi/sil_source_256.npy'))
    parser.add_argument('-c', '--camera-input', type=str,
                        default=os.path.join(data_dir, 'camera.npy'))
    parser.add_argument('-t', '--template-mesh', type=str,
                        default=os.path.join(data_dir, 'obj/sphere/sphere_1352.obj'))
    parser.add_argument('-o', '--output-dir', type=str,
                        default=os.path.join(data_dir, 'results/deform_aloi'))
    parser.add_argument('-b', '--batch-size', type=int,
                        default=72)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = Model(args.template_mesh).cuda()
    transform = sr.LookAt(viewing_angle=15)
    lighting = sr.Lighting()
    rasterizer = sr.SoftRasterizer(image_size=256, sigma_val=1e-4, aggr_func_rgb='hard')

    object_id = 13
    images = np.load(args.filename_input)[(72*object_id):(72*(object_id+1))].astype('float32') / 255.
    out_dir = os.path.join(args.output_dir, str(object_id+1))
    os.makedirs(out_dir, exist_ok=True)
    
    cameras = []
    for i in range(72):
        cameras.append([2.732, 0., i*5.])
    cameras = np.array(cameras).astype('float32')

    optimizer = torch.optim.Adam(model.parameters(), 0.01, betas=(0.5, 0.99))

    camera_distances = torch.from_numpy(cameras[:, 0])
    elevations = torch.from_numpy(cameras[:, 1])
    viewpoints = torch.from_numpy(cameras[:, 2])
    transform.set_eyes_from_angles(camera_distances, elevations, viewpoints)

    loop = tqdm.tqdm(list(range(0, 250)))
    writer = imageio.get_writer(os.path.join(args.output_dir, 'deform.gif'), mode='I')
    for i in loop:
        images_gt = torch.from_numpy(images).cuda()

        mesh, laplacian_loss, flatten_loss = model(args.batch_size)

        # render
        mesh = lighting(mesh)
        mesh = transform(mesh)
        images_pred = rasterizer(mesh)

        # optimize mesh with silhouette reprojection error and
        # geometry constraints
        loss = neg_iou_loss(images_pred[:, 3], images_gt[:, 3]) + \
            0.03 * laplacian_loss + \
            0.0003 * flatten_loss

        loop.set_description('Loss: %.4f' % (loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            image = images_pred.detach().cpu().numpy()[0].transpose((1, 2, 0))
            writer.append_data((255*image).astype(np.uint8))
            imageio.imsave(os.path.join(out_dir, 'sil_%05d.png' % i), (255*image[..., -1]).astype(np.uint8))

            # save optimized mesh
            model(1)[0].save_obj(os.path.join(out_dir, ((str(object_id+1)) + ('mesh_%05d.obj' % i))), save_texture=False)


if __name__ == '__main__':
    main()
