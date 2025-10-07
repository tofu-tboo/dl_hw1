import numpy as np

def get_out_shape(H, W, KH, KW, stride, pad):
    OH = (H + 2*pad - KH)//stride + 1
    OW = (W + 2*pad - KW)//stride + 1
    return OH, OW

def im2col(X, KH, KW, stride=1, pad=0):
    # X: (N,C,H,W) -> col: (N*OH*OW, C*KH*KW)
    N, C, H, W = X.shape
    OH, OW = get_out_shape(H, W, KH, KW, stride, pad)
    Xpad = np.pad(X, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
    col = np.zeros((N, C, KH, KW, OH, OW), dtype=X.dtype)
    for i in range(KH):
        i_max = i + stride*OH
        for j in range(KW):
            j_max = j + stride*OW
            col[:, :, i, j, :, :] = Xpad[:, :, i:i_max:stride, j:j_max:stride]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*OH*OW, C*KH*KW)
    return col, OH, OW

def col2im(col, X_shape, KH, KW, stride=1, pad=0):
    # col: (N*OH*OW, C*KH*KW) -> X: (N,C,H,W)
    N, C, H, W = X_shape
    OH, OW = get_out_shape(H, W, KH, KW, stride, pad)
    col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)
    Xpad = np.zeros((N, C, H + 2*pad, W + 2*pad), dtype=col.dtype)
    for i in range(KH):
        i_max = i + stride*OH
        for j in range(KW):
            j_max = j + stride*OW
            Xpad[:, :, i:i_max:stride, j:j_max:stride] += col[:, :, i, j, :, :]
    if pad == 0:
        return Xpad
    return Xpad[:, :, pad:-pad, pad:-pad]
