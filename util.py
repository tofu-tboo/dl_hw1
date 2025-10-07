import numpy as np

def get_filtered_shape(width, height, kernel_width, kernel_height, stride, padding):
    out_width  = (width  + 2 * padding - kernel_width)  // stride + 1
    out_height = (height + 2 * padding - kernel_height) // stride + 1
    return out_width, out_height

def im2col(x, kernel_width, kernel_height, stride, padding):
    batch_size, channels, width, height = x.shape
    out_width, out_height = get_filtered_shape(width, height, kernel_width, kernel_height, stride, padding)

    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    columns = np.zeros((batch_size, channels, kernel_width, kernel_height, out_width, out_height))
    for i in range(kernel_width):
        i_max = i + stride * out_width
        for j in range(kernel_height):
            j_max = j + stride * out_height

            columns[:, :, i, j, :, :] = x_padded[:, :, i:i_max:stride, j:j_max:stride]

    columns = columns.transpose(0, 4, 5, 1, 2, 3).reshape(batch_size * out_width * out_height, channels * kernel_width * kernel_height)
    return columns, out_width, out_height

def col2im(columns, x_shape, kernel_width, kernel_height, stride, padding):
    batch_size, channels, width, height = x_shape
    out_width, out_height = get_filtered_shape(width, height, kernel_width, kernel_height, stride, padding)

    x_padded = np.zeros((batch_size, channels, width + 2 * padding, height + 2 * padding))

    columns_reshaped = columns.reshape(batch_size, out_width, out_height, channels, kernel_width, kernel_height).transpose(0, 3, 4, 5, 1, 2)
    for i in range(kernel_width):
        i_max = i + stride * out_width
        for j in range(kernel_height):
            j_max = j + stride * out_height
            x_padded[:, :, i:i_max:stride, j:j_max:stride] += columns_reshaped[:, :, i, j, :, :]

    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]