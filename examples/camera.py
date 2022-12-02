import numpy as np
import os
import os.path as osp
import pycolmap
import imageio
import re
from collections import OrderedDict

class SfMCamera:
	def __init__(self, output_path, input_path):
        # self.run_reconstruction = run_reconstruction
		self.output_path = output_path #output path is the path to the reconstruction folder by default
		self.img_path = input_path
		img = imageio.imread(osp.join(self.img_path, os.listdir(self.img_path)[0])) #read the first image
		self.h, self.w, _ = img.shape
		self.image_center = np.array([0.5*self.h, 0.5*self.w, 1])  #image in homogeneous/projected coordinates (u, v, 1)

	def reconstruct(self):
		"""Run colmap reconstruction pipeline first, requires CUDA"""
		mvs_path = osp.join(self.output_path, "mvs")
		database_path = osp.join(self.output_path, "database.db")

		pycolmap.extract_features(database_path, self.img_path)
		maps = pycolmap.incremental_mapping(database_path, self.img_path, self.output_path)
		maps[0].write(self.output_path)
		pycolmap.undistort_images(mvs_path, self.output_path, self.img_path)
		pycolmap.patch_match_stereo(mvs_path)
		pycolmap.stereo_fusion(osp.join(mvs_path, "dense.ply"), mvs_path)

	def get_camera_extrinsics(self, is_dense=True):
		"""Reads from reconstruction results, returns absolute camera heading for each view/image as a list"""
		if is_dense: #dense reconstruction
			reconstruction_path = osp.join(self.output_path, "dense/0/sparse")
		else:  #sparse reconstruction
			reconstruction_path = osp.join(self.output_path, "sparse/0")
		reconstruction = pycolmap.Reconstruction(reconstruction_path)
		print(reconstruction.summary())

		camera_loc_dct = {}

		def unit_vector(v):
			return v / np.linalg.norm(v)

		def vector_ang(v1, v2):
			v1_u = unit_vector(v1.reshape(3,))
			v2_u = unit_vector(v2.reshape(3,))
			return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

		for image_id, image in reconstruction.images.items():
			# print(f"Image No.{image_id}")
			R = self.quaternion_to_rotation(image.qvec)
			T = image.tvec.reshape(3,1)
			# print("Rotation Matrix: ", R)
			camera_loc = -np.linalg.inv(R) @ T  #get the normalized camera location
			camera_loc_dct[image.name] = camera_loc.reshape(3,)

		    # camera_center = T + R @ np.array([0, 0, 0]).reshape(3,1)
		    # camera_pose = list(camera_loc.reshape(3,))
		    # camera_ang = (np.arctan2(camera_pose[1], camera_pose[0]) + np.pi) * 180.0 / np.pi
		    # camera_ang_t = (np.arctan2(T.reshape(3,)[1], T.reshape(3,)[0]) + np.pi) * 180.0 / np.pi
		    # camera_ang_c = (np.arctan2(camera_center.reshape(3,)[1], camera_center.reshape(3,)[0]) + np.pi) * 180.0 / np.pi
		    # camera_pose.append(camera_ang)
		    # print(f"Image {image.name} camera angle: ", camera_ang, camera_ang_t, camera_ang_c)

		camera_names = self.natural_sorting(list(camera_loc_dct)) #sort the filenames so that it starts from 0 degree
		# print(camera_names)

		ordered_camera_loc_dct = OrderedDict()
		for name in camera_names:
			ordered_camera_loc_dct[name] = camera_loc_dct[name]

		ordered_camera_angle_dct = OrderedDict()
		center_camera_loc = np.zeros(3)
		for _, loc in camera_loc_dct.items():
			center_camera_loc += loc
		center_camera_loc = 1 / len(camera_names) * center_camera_loc

		for name, loc in ordered_camera_loc_dct.items():
			if camera_names.index(name) == 0:
				ordered_camera_angle_dct[name] = 0.
			else:
				loc_vec1 = ordered_camera_loc_dct[camera_names[camera_names.index(name) - 1]] - center_camera_loc
				loc_vec2 = ordered_camera_loc_dct[name] - center_camera_loc
				ordered_camera_angle_dct[name] = vector_ang(loc_vec1, loc_vec2) * 180.0 / np.pi   #angle between two neighboring cameras, radian to degree


		# camera_angle_dct = {name: (np.arctan2(loc_vec[1], loc_vec[0]) + np.pi) * 180.0 / np.pi for name, loc_vec in camera_loc_dct.items()}

		ordered_abs_angle_dct = OrderedDict()
		angle = 0
		for name, ang in ordered_camera_angle_dct.items():
			angle += ang
			ordered_abs_angle_dct[name] = angle


		return list(ordered_abs_angle_dct.values())


	def get_camera_intrinsics(self, is_dense=True):
		"""Reads from reconstruction results, returns the intrinsic params for the estimated camera type at each view as dict"""
		if is_dense: #dense reconstruction
			reconstruction_path = osp.join(self.output_path, "dense/0/sparse")
		else:  #sparse reconstruction
			reconstruction_path = osp.join(self.output_path, "sparse/0")
		reconstruction = pycolmap.Reconstruction(reconstruction_path)
		reconstruction.export_CAM(osp.join(self.output_path, "sparse"), skip_distortion=False)
		print(reconstruction.summary())

		camera_params_dct = {}
		for camera_id, camera in reconstruction.cameras.items():
			camera_params_dct[camera_id] = camera.params
		    #get intrinsic camera params
			print("Image center to camera frame: ", camera.image_to_world((0.5*self.w, 0.5*self.h)))  #should be (0, 0) in the camera frame for pinhole
		
		return camera_params_dct

	@staticmethod
	def quaternion_to_rotation(qvec):
		q0 = qvec[0]
		q1 = qvec[1]
		q2 = qvec[2]
		q3 = qvec[3]
	     
		r00 = 2 * (q0 * q0 + q1 * q1) - 1
		r01 = 2 * (q1 * q2 - q0 * q3)
		r02 = 2 * (q1 * q3 + q0 * q2)
	     
		r10 = 2 * (q1 * q2 + q0 * q3)
		r11 = 2 * (q0 * q0 + q2 * q2) - 1
		r12 = 2 * (q2 * q3 - q0 * q1)
	     
		r20 = 2 * (q1 * q3 - q0 * q2)
		r21 = 2 * (q2 * q3 + q0 * q1)
		r22 = 2 * (q0 * q0 + q3 * q3) - 1
	     
		rotation_matrix = np.array([[r00, r01, r02],
                           			[r10, r11, r12],
                           			[r20, r21, r22]])
	                            
		return rotation_matrix

	@staticmethod
	def natural_sorting(l):
		#### https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/
		convert = lambda text: int(text) if text.isdigit() else text
		alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
		return sorted(l, key=alphanum_key)