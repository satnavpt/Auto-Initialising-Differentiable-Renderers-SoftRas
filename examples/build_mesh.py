import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import tqdm
import numpy as np
import imageio
import soft_renderer as sr
import re

# import segment as seg

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

class Model(nn.Module):
    def __init__(self, mesh_init):
        super(Model, self).__init__()

        # set template mesh
        self.template_mesh = mesh_init
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

class Builder:
    def __init__(self, exp_name, images, cameras, mesh_init, batch_size, output_dir, device, iters):
        self.exp_name = exp_name
        self.images = images
        self.cameras = cameras
        self.output_dir = os.path.join(output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.mesh_init = mesh_init
        self.batch_size = batch_size
        self.device = device
        self.iters = iters

    def build_mesh(self):
        model = Model(self.mesh_init).to(self.device)
        transform = sr.LookAt(viewing_angle=15)
        lighting = sr.Lighting()
        rasterizer = sr.SoftRasterizer(image_size=256, sigma_val=1e-4, aggr_func_rgb='hard')
        optimizer = torch.optim.Adam(model.parameters(), 0.01, betas=(0.5, 0.99))

        try:
            images = self.images.astype('float32') / 255.
        except:

            images = self.images.cpu().detach().numpy().astype('float32') / 255.
        cameras = self.cameras

        camera_distances = torch.from_numpy(cameras[:, 0])
        elevations = torch.from_numpy(cameras[:, 1])
        viewpoints = torch.from_numpy(cameras[:, 2])
        transform.set_eyes_from_angles(camera_distances, elevations, viewpoints)

        loop = tqdm.tqdm(list(range(0, self.iters)))
        writer = imageio.get_writer(os.path.join(self.output_dir, 'deform.gif'), mode='I')
        for i in loop:
            images_gt = torch.from_numpy(images).to(self.device)
            mesh, laplacian_loss, flatten_loss = model(self.batch_size)

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

            image = images_pred.detach().cpu().numpy()[0].transpose((1, 2, 0))
            writer.append_data((255*image).astype(np.uint8))

            if (i % 100 == 0) or (i == self.iters-1):
                imageio.imsave(os.path.join(self.output_dir, 'sil_%04d_loss_%04d.png' % (i, loss.item()*10000)), (255*image[..., -1]).astype(np.uint8))
                model(1)[0].save_obj(os.path.join(self.output_dir, ('mesh_%04d_loss_%04d.obj' % (i, loss.item()*10000))), save_texture=False)