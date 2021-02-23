from numpy.lib.stride_tricks import as_strided


def make_4d_array_custom(arr_padded, N, M, a, b):
    return as_strided(arr_padded, shape=(N, M, a, b), strides=arr_padded.strides * 2)
