import matplotlib.pyplot as plt
import numpy as np

dfile = "data/datasets/02691156_test_images.npz"
images = np.load(dfile)["arr_0"][0]
print(images.shape)

for im in images:
    plt.imshow(im[0])
    plt.show()
    break