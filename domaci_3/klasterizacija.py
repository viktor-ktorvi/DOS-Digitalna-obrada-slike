import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from sklearn.cluster import KMeans
import time
from skimage.color import rgb2gray
import skimage.measure

# %%
import numpy as np
from numpy.lib.stride_tricks import as_strided


def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                    (A.shape[1] - kernel_size) // stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape=output_shape + kernel_size,
                     strides=(stride * A.strides[0],
                              stride * A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1, 2)).reshape(output_shape)


# %% Ucitavanje

test_frames = []
file_name = 'test'
extension = '.jpg'

for i in range(1, 7):
    test_frames.append(io.imread('sekvence/test_frames/' + file_name + str(i) + extension))

figsize = (10, 7)
fontsize = 20
# %%

img = test_frames[1]
img_double = img / np.amax(img)

img_step = 50
img_double = img_double[::img_step, ::img_step]

plt.figure()
plt.imshow(img_double)
plt.show()
# %%

N = img_double.shape[0]
M = img_double.shape[1]
num_colors = img_double.shape[2]

start = time.time()
kmeans = KMeans(n_clusters=2, random_state=0).fit(img_double.reshape((N * M, num_colors)))
end = time.time()
print("Vreme = ", end - start, " sec")
# %%
print(kmeans.labels_.shape)
print(kmeans.labels_)
mask = kmeans.labels_.reshape((N, M))

# %%
plt.figure()
plt.imshow(mask, cmap='gray')
plt.show()
