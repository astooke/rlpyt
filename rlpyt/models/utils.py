

def conv2d_output_shape(h, w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    Returns output H, W after convolution/pooling on input H, W.
    """
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    if not isinstance(stride, tuple):
        stride = (stride, stride)
    if not isinstance(padding, tuple):
        padding = (padding, padding)
    h = (h + (2 * padding[0]) - (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    w = (w + (2 * padding[1]) - (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1
    return h, w
