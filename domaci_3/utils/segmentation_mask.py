import numpy as np
from skimage import filters
from skimage.morphology import disk
from domaci_3.utils.distance import distance_einsum
from domaci_3.utils.division import get_mass_division


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

    # vracamo masku na 0/1 i cuvamo na najmanje moguce bita u pajtonu koliko znam 8
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