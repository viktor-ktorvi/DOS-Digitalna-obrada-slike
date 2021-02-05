# %% importi
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
import skvideo.io
import time
from numpy.lib.stride_tricks import as_strided
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import cv2
import imageio
from skimage import filters
from skimage.morphology import disk


def distance_einsum(Y, sigma_inv, M):
    Z = Y - M
    W = np.einsum("ijk,kl->ijl", Z, sigma_inv)
    d = np.einsum("ijk,ijk->ij", W, Z)
    return d

# TODO Ovo mora opasno da se refaktorise
def segment_by_sample_with_resizing(img, img_step, samples, ones_img_small,
                                    zeros_img_small, median_radius, invert_mask_flg, hist_range_high, division):
    img_double = img / np.amax(img)
    old_mask = np.array(ones_img_small)

    for sample in samples:
        # mahalanobis za svaki img_step-ti piksel
        dist = distance_einsum(img_double[::img_step, ::img_step, :], sample.sigma_inv, sample.M)

        # hist_range_high = np.amax(dist)
        hist_f, _ = np.histogram(dist.flatten(),
                                 bins=256, range=(0.0, hist_range_high))
        # ovo je neko pronalazenje praga, moze ovo bolje
        sample.threshold = get_mass_division(hist_f, division) / len(hist_f) * hist_range_high

        # binarizacija
        old_mask[dist < sample.threshold] = zeros_img_small[dist < sample.threshold]

    # posto je originalna binarizacija uradjena da se izbaci sempl, posto je sempl sad put mora maska da se kontrira
    if invert_mask_flg:
        old_mask = 1 - old_mask

    # da se izbace tackice medijan filtar koji radi sa 0..255 vrednostima
    median_mask = np.array(old_mask)
    median_mask *= 255
    median_mask = median_mask.astype(np.uint8)

    median_mask = filters.median(median_mask, disk(median_radius), mode='mirror')

    # vracamo masku na 0/1 i cuvamo na najmanje moguce bita u matlabu koliko znam 8
    median_mask = median_mask / 255.0
    new_mask = median_mask.astype(np.uint8)

    # madjija za skaliranje slike
    # smanjili sliku img_step puta pa povecavamo za img_step
    # koliko kontam fja vraca pogled dimenzija male slike X img_step^2 matricice, znaci oko svakog piksela se
    # ponavlja taj piksel i onda taj pogled lepo preoblikujemo
    new_mask_resized = np.broadcast_to(new_mask[:, None, :, None],
                                       (new_mask.shape[0], img_step, new_mask.shape[1], img_step)).reshape(
        new_mask.shape[0] * img_step, new_mask.shape[1] * img_step)

    # posto originalna slika vrv nije bila deljiva sa img_step odsecamo ono sitno sto prelazi dimenzije slike
    new_mask_resized = new_mask_resized[0:img_double.shape[0], 0:img_double.shape[1]]

    # primena maske
    return img_double * new_mask_resized[:, :, np.newaxis], new_mask_resized, old_mask


# TODO Prag nalaziti pametnije. Videti sta Mezeni radi
def get_mass_division(hist, division):
    for i in range(len(hist)):
        if np.sum(hist[0:i]) / np.sum(hist) > division:
            return i


# %% Ucitavanje

test_frames = []
file_name = 'test'
extension = '.jpg'

for i in range(1, 7):
    test_frames.append(io.imread('sekvence/test_frames/' + file_name + str(i) + extension))

figsize = (10, 7)
fontsize = 20


# for i in range(len(test_frames)):
#     plt.figure(figsize=figsize)
#     plt.imshow(test_frames[i])
#     plt.show()


# %% Sample
class Sample:
    def __init__(self, img, height, width, threshold):
        self.threshold = threshold
        self.height = height
        self.width = width
        self.img = img

        self.sample = self.img[self.height[0]: self.height[1], self.width[0]: self.width[1], :]
        # self.sample = cv2.medianBlur(self.sample, 5)
        self.sample = self.sample / np.amax(self.sample)
        X = self.sample.reshape((self.sample.shape[0] * self.sample.shape[1], 3)).T

        sigma = np.cov(X)
        self.M = np.mean(X, 1)
        self.sigma_inv = np.linalg.inv(sigma)


road_sample_height = (500, 600)
road_sample_width = (420, 920)

road = Sample(test_frames[3], road_sample_height, road_sample_height, threshold=15)

sky_sample_height = (10, 200)
sky_sample_width = (10, 200)

sky = Sample(test_frames[0], sky_sample_height, sky_sample_width, threshold=750)

trees_sample_height = (280, 320)
trees_sample_width = (450, 550)

trees = Sample(test_frames[0], trees_sample_height, trees_sample_width, threshold=15)

branches_sample_height = (200, 300)
branches_sample_width = (100, 250)

branches = Sample(test_frames[4], branches_sample_height, branches_sample_width, threshold=3)

more_branches_sample_height = (0, 90)
more_branches_sample_width = (600, 700)

more_branches = Sample(test_frames[4], more_branches_sample_height, more_branches_sample_width, threshold=2)

yellow_grass_sample_height = (310, 420)
yellow_grass_sample_width = (1075, 1270)

yellow_grass = Sample(test_frames[1], yellow_grass_sample_height, yellow_grass_sample_width, threshold=5)

# %% Test Image

img = test_frames[1]
img_double = img / np.amax(img)

img_step = 15
median_radius = 7
sample = road

invert_mask_flg = True
# %% Sample testing

dist = distance_einsum(img_double[::img_step, ::img_step, :], sample.sigma_inv, sample.M)
plt.figure(figsize=figsize)
plt.imshow(dist, cmap='gray')
plt.show()

small_mask = np.ones_like(dist)
small_mask[dist < sample.threshold] = np.zeros_like(dist)[dist < sample.threshold]

start_time = time.time()

big_mask = np.broadcast_to(small_mask[:, None, :, None],
                           (small_mask.shape[0], img_step, small_mask.shape[1], img_step)).reshape(
    small_mask.shape[0] * img_step, small_mask.shape[1] * img_step)

print("%.3f" % (time.time() - start_time), ' sec')

plt.figure(figsize=figsize)
plt.imshow(small_mask, cmap='gray')
plt.show()

plt.figure(figsize=figsize)
plt.imshow(big_mask, cmap='gray')
plt.show()

start_time = time.time()

hist_range_high = np.amax(dist)
division = 0.33
hist_f, bin_edges = np.histogram(dist.flatten(),
                                 bins=256, range=(0.0, hist_range_high))

# fig = plt.figure(figsize=figsize)
# plt.bar(bin_edges[0:-1], hist_f)

road.threshold = get_mass_division(hist_f, division) / len(hist_f) * hist_range_high
print("%.3f" % (time.time() - start_time), ' sec')

print('threshold = ', road.threshold)

# %% Example frame
# samples = [sky, branches, trees, more_branches]
samples = [road]
for sample in samples:
    plt.figure(figsize=figsize)
    plt.imshow(sample.sample)
    plt.show()

ones_img_small = np.ones((img_double[::img_step, ::img_step].shape[0], img_double[::img_step, ::img_step].shape[1]))
zeros_img_small = np.zeros((img_double[::img_step, ::img_step].shape[0], img_double[::img_step, ::img_step].shape[1]))

start_time = time.time()
[segmented, new_mask, old_mask] = segment_by_sample_with_resizing(img=img,
                                                                  img_step=img_step,
                                                                  samples=samples,
                                                                  ones_img_small=ones_img_small,
                                                                  zeros_img_small=zeros_img_small,
                                                                  median_radius=median_radius,
                                                                  invert_mask_flg=invert_mask_flg,
                                                                  hist_range_high=hist_range_high,
                                                                  division=division)

print("%.3f" % (time.time() - start_time), 'test sec')
plt.figure(figsize=figsize)
plt.imshow(img)
plt.show()
plt.figure(figsize=figsize)
plt.imshow(old_mask, cmap='gray')
plt.show()
plt.figure(figsize=figsize)
plt.imshow(new_mask, cmap='gray')
plt.show()
plt.figure(figsize=figsize)
plt.imshow(segmented)
plt.show()

# %% Load video
process_all_flg = False
if process_all_flg:
    videodata = skvideo.io.vread("sekvence/video_road.mp4")
    print(videodata.shape)
# %% Parameters and helpers
if process_all_flg:
    img = np.array(videodata[0])
    img_double = img / np.amax(img)

    img_step = 15
    median_radius = 7

    hist_range_high = 1000
    division = 0.4

    ones_img_small = np.ones((img_double[::img_step, ::img_step].shape[0], img_double[::img_step, ::img_step].shape[1]))
    zeros_img_small = np.zeros(
        (img_double[::img_step, ::img_step].shape[0], img_double[::img_step, ::img_step].shape[1]))

    new_video = np.zeros_like(videodata)
    start_time = time.time()
# %% Process all frames
if process_all_flg:
    for i in range(videodata.shape[0]):
        [frame, _, _] = segment_by_sample_with_resizing(img=videodata[i],
                                                        img_step=img_step,
                                                        samples=samples,
                                                        ones_img_small=ones_img_small,
                                                        zeros_img_small=zeros_img_small,
                                                        median_radius=median_radius,
                                                        invert_mask_flg=invert_mask_flg,
                                                        hist_range_high=hist_range_high,
                                                        division=division)
        frame *= 255
        new_video[i, :, :, :] = frame.astype(np.uint8)
        if i % 10 == 0:
            print("%3.1f" % ((i + 1) / videodata.shape[0] * 100), " %", end="\r")
    print("%.3f" % (time.time() - start_time), "sec za video")

# %% Make video
if process_all_flg:
    imageio.mimwrite(
        'C:\\Users\\HP\\PycharmProjects\\Digitalna obrada slike Elektronika\\Domaci\\Domaci repo\\Domaci 3\\out_vid.mp4',
        new_video, fps=24.0)
