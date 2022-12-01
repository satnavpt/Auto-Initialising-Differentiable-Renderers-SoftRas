# """
# hight2obj

# Converts a heightmap to a 3d mesh.

# Usage:

# hight2obj SOURCE DESTINATION SCALE

# Parameters:

# * SOURCE: Filename of the source hightmap. This must be a 8 bpp greyscale
#           bitmap-type image. PNG is tested an recommended.
# * DESTINATION: Filename of the destination file into which the 3d data is
#                written.
# * SCALE: Scale factor for the z-coordinate. This value is the maximum height
#          of the resulting mesh relative to the short side side of the SOURCE
#          image.
# """

# from PIL import Image

# def create_vertices(image, scale):
#     """Returns a list of vertices for a given heightmap."""
#     norm = min(image.size) - 1 # normalize coordinates to shortest image length
#     vertices = []
#     for y in range(image.height):
#         for x in range(image.width):
#             vertex = (x / norm, y / norm, image.getpixel((x, y))[0] / 255 * scale)
#             vertices.append(vertex)
#     return vertices

# def create_faces(image, width, height):
#     """Returns a regular mesh of a given size. triangles are defined by vertex
#     indices."""
#     faces = []
#     for y in range(height - 1):
#         for x in range(width - 1):
#             offset = y * width + 1 # vertex index in obj-format starts at 1
#             # relative vertex positions: [u]pper, [l]ower / [l]eft, [r]ight
#             ul, ur, ll, lr = (offset + x, offset + x + 1,
#                             offset + width + x, offset + width + x + 1)
#             # two triangles per square: upper left and lower right
#             faces.append((ul, ur, ll))
#             faces.append((ll, ur, lr))
#     return faces

# def get_front_and_sides(image):
#     vertices = []
#     edgeVertices = []
#     inv = False
#     inh = False
#     vertCount = 0
#     for y in range(image.height):
#         for x in range(image.width):
#             pixel = image.getpixel((x, y))[0]
#             if pixel > 0:
#                 vertices.append((x, y, 50))
#                 inh = True
#                 inv = True
#             else:
#                 vertices.append((x, y, 0))
#                 if inv:
#                     inv = False
#                     edgeVertices.append(vertCount)
#                 elif inh:
#                     inh = False
#                     edgeVertices.append(vertCount)
#             vertCount += 1

#     faces = []
#     for y in range(image.height - 1):
#         for x in range(image.width - 1):
#             offset = y * image.width + 1 # vertex index in obj-format starts at 1
#             # relative vertex positions: [u]pper, [l]ower / [l]eft, [r]ight
#             ul, ur, ll, lr = (offset + x, offset + x + 1,
#                             offset + image.width + x, offset + image.width + x + 1)
#             # two triangles per square: upper left and lower right
#             faces.append((ul, ur, ll))
#             faces.append((ll, ur, lr))

#     verticesToRemove = []
#     for i, v in enumerate(vertices):
#         if v[2] == 0:
#             verticesToRemove.append(i)
#     verticesToRemove = set(verticesToRemove) - set(edgeVertices)

#     facesToRemove = []
#     for i, f in enumerate(faces):
#         if set(f).issubset(verticesToRemove):
#             facesToRemove.append(i)
    
#     # verticesToKeep = set(range(len(vertices))) - set(verticesToRemove)
#     # newVertices = [vertices[i] for i in verticesToKeep]
    
#     facesToKeep = set(range(len(faces))) - set(facesToRemove)
#     newFaces = [faces[i] for i in facesToKeep]

#     return vertices, newFaces

# def get_back(image, vertices, offset):
#     faces = []
#     for y in range(image.height - 1):
#         for x in range(image.width - 1):
#             offset = y * image.width + 1 # vertex index in obj-format starts at 1
#             # relative vertex positions: [u]pper, [l]ower / [l]eft, [r]ight
#             ul, ur, ll, lr = (offset + x, offset + x + 1,
#                             offset + image.width + x, offset + image.width + x + 1)
#             # two triangles per square: upper left and lower right
#             faces.append((ul, ur, ll))
#             faces.append((ll, ur, lr))

#     verticesToRemove = []
#     for i, v in enumerate(vertices):
#         if v[2] > 0:
#             verticesToRemove.append(i)
#     verticesToRemove = set(verticesToRemove) - set(edgeVertices)

#     facesToRemove = []
#     for i, f in enumerate(faces):
#         if not set(f).isdisjoint(verticesToRemove):
#             facesToRemove.append(i)
    
#     # verticesToKeep = set(range(len(vertices))) - set(verticesToRemove)
#     # newVertices = [vertices[i] for i in verticesToKeep]
    
#     facesToKeep = set(range(len(faces))) - set(facesToRemove)
#     newFaces = [faces[i] for i in facesToKeep]

#     newNewFaces = []
#     for f in newFaces:
#         newF = (f[0] + offset, f[1] + offset, f[2] + offset)
#         newNewFaces.append(newF)

#     return vertices, newFaces

# def main(im, outpath, scale):
#     im = Image.open(im)#.resize((64,64))
    
#     scale = float(scale)
#     print('Image: size={}x{}, mode={}'.format(*im.size, im.mode))
#     print('Polygon count: {}'.format(2 * (im.width -1) * (im.height - 1)))
#     # vertices = create_vertices(im, scale)
#     # faces = create_faces(im, *im.size)
#     vertices, faces = get_front_and_sides(im)
#     back_v, back_f = get_back(im, len(vertices))
#     vertices = vertices + back_v
#     faces = faces + back_f
#     with open(outpath, 'w') as outfile:
#         for v in vertices:
#             outfile.write('v {} {} {}\n'.format(*v))
#         for f in faces:
#             outfile.write('f {} {} {}\n'.format(*f))

# main('data/aloi/sil_grey_256/1/1_r0.png', 'testmesh/out.obj', 1.)

import numpy as np
import torch

class Initialiser:
    def __init__(self, image, im_dim=256):
        self.visible = (image[0] > 0).astype(int)
        self.objheight, self.top = self.vert()
        self.objwidth, self.left = self.horiz()
        self.im_dim = im_dim

    def vert(self):
        rows = []
        for i, row in enumerate(self.visible):
            if np.isin(1, row):
                rows.append(i)
        return rows[len(rows)-1] - rows[0], rows[0]

    def horiz(self):
        cols = []
        for i, col in enumerate(np.transpose(self.visible)):
            if np.isin(1, col):
                cols.append(i)
        return cols[len(cols)-1] - cols[0], cols[0]

    def initialise(self, mesh, orig_rad = 90):
        width_scale = orig_rad / (self.objwidth/2)
        height_scale = orig_rad / (self.objheight/2)

        self.horizontal_scale(mesh, width_scale)
        self.vertical_scale(mesh, height_scale)
        
        orig_top = (self.im_dim/2) - orig_rad/height_scale
        orig_left = (self.im_dim/2) - orig_rad/width_scale
        shift_per_pixel = 1/orig_rad
        horiz_shift = (orig_left - self.left) * shift_per_pixel
        vert_shift = (orig_top - self.top) * shift_per_pixel

        self.horizontal_shift(mesh, horiz_shift)
        self.vertical_shift(mesh, vert_shift)

        print("\n")
        print("w scale: ", width_scale)
        print("orig_rad: ", orig_rad)
        print("img left: ", self.left)
        print("sphere left: ", orig_left)
        print("horiz shift: ", horiz_shift)
        print("width per pixel: ", shift_per_pixel)
        print("\n")
        print("h scale: ", height_scale)
        print("orig_rad: ", orig_rad)
        print("img top: ", self.top)
        print("sphere top: ", orig_top)
        print("vert shift: ", vert_shift)
        print("height per pixel: ", shift_per_pixel)
        print("\n")

    def horizontal_scale(self, mesh, scale):
        x = torch.transpose(mesh.vertices[0], 0, 1)[0]
        x = torch.unsqueeze((x / scale), 0)
        i = torch.arange(mesh.vertices[0].size(0)).long()
        mesh.vertices[0,i,0] = x

    def vertical_scale(self, mesh, scale):
        y = torch.transpose(mesh.vertices[0], 0, 1)[1]
        y = torch.unsqueeze((y / scale), 0)
        i = torch.arange(mesh.vertices[0].size(0)).long()
        mesh.vertices[0,i,1] = y

    def horizontal_shift(self, mesh, shift):
        x = torch.transpose(mesh.vertices[0], 0, 1)[0]
        x = torch.unsqueeze((x - shift), 0)
        i = torch.arange(mesh.vertices[0].size(0)).long()
        mesh.vertices[0,i,0] = x

    def vertical_shift(self, mesh, shift):
        y = torch.transpose(mesh.vertices[0], 0, 1)[1]
        y = torch.unsqueeze((y + shift), 0)
        i = torch.arange(mesh.vertices[0].size(0)).long()
        mesh.vertices[0,i,1] = y