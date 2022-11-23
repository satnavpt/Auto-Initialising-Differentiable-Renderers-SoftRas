import os

import soft_renderer.functional as srf
import torch
import numpy as np
import tqdm
import re
import matplotlib.pyplot as plt
import imageio
from PIL import Image

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


class_ids_map = {
    '02691156': 'Airplane',
    '02828884': 'Bench',
    '02933112': 'Cabinet',
    '02958343': 'Car',
    '03001627': 'Chair',
    '03211117': 'Display',
    '03636649': 'Lamp',
    '03691459': 'Loudspeaker',
    '04090263': 'Rifle',
    '04256520': 'Sofa',
    '04379243': 'Table',
    '04401088': 'Telephone',
    '04530566': 'Watercraft',
}


class ShapeNet(object):
    def __init__(self, directory=None, class_ids=None, set_name=None):
        self.class_ids = class_ids
        self.set_name = set_name
        self.elevation = 30.
        self.distance = 2.732

        self.class_ids_map = class_ids_map

        images = []
        voxels = []
        self.num_data = {}
        self.pos = {}
        count = 0
        loop = tqdm.tqdm(self.class_ids)
        loop.set_description('Loading dataset')
        for class_id in loop:
            images.append(list(np.load(
                os.path.join(directory, '%s_%s_images.npz' % (class_id, set_name))).items())[0][1])
            voxels.append(list(np.load(
                os.path.join(directory, '%s_%s_voxels.npz' % (class_id, set_name))).items())[0][1])
            self.num_data[class_id] = images[-1].shape[0]
            self.pos[class_id] = count
            count += self.num_data[class_id]

        images = np.concatenate(images, axis=0)#.reshape((-1, 4, 64, 64))
        print(images.shape)
        images = images.reshape((-1, 4, 64, 64))
        print(images.shape)
        images = np.ascontiguousarray(images)
        print(images[0][:3,:,:].transpose(1,2,0).shape)
        imageio.imsave('testshapenet.png', images[0][:3,:,:].transpose(1,2,0))
        self.images = images
        self.voxels = np.ascontiguousarray(np.concatenate(voxels, axis=0))
        del images
        del voxels

    @property
    def class_ids_pair(self):
        class_names = [self.class_ids_map[i] for i in self.class_ids]
        return zip(self.class_ids, class_names)

    def get_random_batch(self, batch_size):
        data_ids_a = np.zeros(batch_size, 'int32')
        data_ids_b = np.zeros(batch_size, 'int32')
        viewpoint_ids_a = torch.zeros(batch_size)
        viewpoint_ids_b = torch.zeros(batch_size)
        for i in range(batch_size):
            class_id = np.random.choice(self.class_ids)
            object_id = np.random.randint(0, self.num_data[class_id])

            viewpoint_id_a = np.random.randint(0, 24)
            viewpoint_id_b = np.random.randint(0, 24)
            data_id_a = (object_id + self.pos[class_id]) * 24 + viewpoint_id_a
            data_id_b = (object_id + self.pos[class_id]) * 24 + viewpoint_id_b
            data_ids_a[i] = data_id_a
            data_ids_b[i] = data_id_b
            viewpoint_ids_a[i] = viewpoint_id_a
            viewpoint_ids_b[i] = viewpoint_id_b

        # print(data_ids_a)
        # print(viewpoint_ids_a)
        # print(self.images.shape)

        # print(self.images[data_ids_a][0])

        images_a = torch.from_numpy(self.images[data_ids_a].astype('float32') / 255.)
        images_b = torch.from_numpy(self.images[data_ids_b].astype('float32') / 255.)

        distances = torch.ones(batch_size).float() * self.distance
        elevations_a = torch.ones(batch_size).float() * self.elevation
        elevations_b = torch.ones(batch_size).float() * self.elevation
        viewpoints_a = srf.get_points_from_angles(distances, elevations_a, -viewpoint_ids_a * 15)
        viewpoints_b = srf.get_points_from_angles(distances, elevations_b, -viewpoint_ids_b * 15)

        return images_a, images_b, viewpoints_a, viewpoints_b

    def get_all_batches_for_evaluation(self, batch_size, class_id):
        data_ids = np.arange(self.num_data[class_id]) + self.pos[class_id]
        viewpoint_ids = np.tile(np.arange(24), data_ids.size)
        data_ids = np.repeat(data_ids, 24) * 24 + viewpoint_ids

        distances = torch.ones(data_ids.size).float() * self.distance
        elevations = torch.ones(data_ids.size).float() * self.elevation
        viewpoints_all = srf.get_points_from_angles(distances, elevations,
                                                    -torch.from_numpy(viewpoint_ids).float() * 15)

        for i in range((data_ids.size - 1) // batch_size + 1):
            images = torch.from_numpy(
                self.images[data_ids[i * batch_size:(i + 1) * batch_size]].astype('float32') / 255.)
            voxels = torch.from_numpy(
                self.voxels[data_ids[i * batch_size:(i + 1) * batch_size] // 24].astype('float32'))
            yield images, voxels

class ALOI(object):
    def __init__(self, directory='/mnt/c/Users/prana/OneDrive/Documents/University/II/Machine Visual Perception/Machine-Visual-Perception/Project/SoftRas/data/aloi', object_ids=range(1,1001), set_name=None):
        self.object_ids = object_ids
        self.set_name = set_name
        self.elevation = 0.
        self.distance = 2.732

        images = []
        self.pos = {}
        loop = tqdm.tqdm(self.object_ids)
        loop.set_description('Loading dataset')
        for object_id in loop:
            sil_dir = os.path.join(directory, ('sil2/' + str(object_id)))
            sil_files = sorted_alphanumeric(os.listdir(sil_dir))
            for sil_file in sil_files:
                sil = imageio.imread(os.path.join(sil_dir, sil_file))
                images.append(sil.transpose(2,0,1))

        images = np.asarray(images)
        images = np.ascontiguousarray(images)
        self.images = images
        del images

    @property
    def class_ids_pair(self):
        class_names = [self.class_ids_map[i] for i in self.class_ids]
        return zip(self.class_ids, class_names)

    def get_random_batch(self, batch_size):
        object_ids_a = np.zeros(batch_size, 'int32')
        object_ids_b = np.zeros(batch_size, 'int32')
        viewpoint_ids_a = torch.zeros(batch_size)
        viewpoint_ids_b = torch.zeros(batch_size)

        for i in range(batch_size):
            object_id = np.random.choice(self.object_ids)

            object_ids_a[i] = object_id
            object_ids_b[i] = object_id

            viewpoint_id_a = np.random.randint(0,72)
            viewpoint_id_b = np.random.randint(0,72)

            viewpoint_ids_a[i] = viewpoint_id_a
            viewpoint_ids_b[i] = viewpoint_id_b

        object_view_ids_a = (((object_ids_a-1)*72) + to_np(viewpoint_ids_a)).astype(int)
        object_view_ids_b = (((object_ids_b-1)*72) + to_np(viewpoint_ids_b)).astype(int)

        images_a = torch.from_numpy(self.images[object_view_ids_a].astype('float32') / 255.)
        images_b = torch.from_numpy(self.images[object_view_ids_b].astype('float32') / 255.)

        distances    = torch.ones(batch_size).float() * self.distance
        elevations_a = torch.ones(batch_size).float() * self.elevation
        elevations_b = torch.ones(batch_size).float() * self.elevation
        viewpoints_a = srf.get_points_from_angles(distances, elevations_a, -viewpoint_ids_a*5)
        viewpoints_b = srf.get_points_from_angles(distances, elevations_b, -viewpoint_ids_b*5)

        return images_a, images_b, viewpoints_a, viewpoints_b

    def get_all_batches_for_evaluation(self, batch_size, class_id):
        pass

def to_np(tensor):
    return tensor.detach().cpu().numpy()