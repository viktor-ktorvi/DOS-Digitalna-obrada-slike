import numpy as np
from domaci_3.utils.array_4d import make_4d_array_custom


def spatial_kernel(r, sigma_s):
    A = np.arange(-r, r + 1)
    A = np.einsum("i,i->i", A, A)
    kernel = np.exp(-(np.tile(A, (2 * r + 1, 1)) + np.tile(A, (2 * r + 1, 1)).T) / (2 * sigma_s ** 2))
    return kernel / np.sum(kernel)


def filter_gauss(img, r, sigma):
    # ovo dobijanje komponenti može efikasnije sigurno. Čitao sam da se dobija preko paskalovog trougla efikasno.
    kernel = spatial_kernel(r, sigma)

    X = np.diag(kernel) ** (1 / 2)
    X = X.reshape((X.size, 1))

    wx = X.T
    wy = X

    # wx == wy
    w_horizontal = wx
    w_vertical = wy

    # ovo može da se prosledi
    img_double = img / np.amax(img)
    padded_sides = np.pad(img_double, ((0, 0), (r, r)),
                          mode="edge")  # prosiruje sa strane jer prvo radimo horizontalni prolaz

    N = img_double.shape[0]
    M = img_double.shape[1]

    a = 1
    b = 2 * r + 1  # lokalno susedstvo (1 x 2r+1)

    # matrica jednodimenzionalnih lokalnih susedstava (1 x 2r+1)
    img_4d_horizontal = make_4d_array_custom(padded_sides, N, M, a, b)

    # slika filtrirana horizontalno
    horizontal_result = np.einsum('ijkl, kl->ij', img_4d_horizontal, w_horizontal)

    # sad tu filtriranu horizontalno pripremamo za vertikalno filtriranje
    padded_top_and_bottom = np.pad(horizontal_result, ((r, r), (0, 0)), mode="edge")  # prosirivanje gore i dole

    a = 2 * r + 1
    b = 1  # lokalno susedstvo (2r+1 x 1)

    # matrica jednodimenzionalnih lokalnih susedstava (1 x 2r+1)
    img_4d_vertical = make_4d_array_custom(padded_top_and_bottom, N, M, a, b)

    # slika filtrirana prvo horizontalno pa konacno i vertikalno, to i vracamo. Moze ovde samo return radi efikasnosti
    vertical_result = np.einsum('ijkl, kl->ij', img_4d_vertical, w_vertical)

    return vertical_result
