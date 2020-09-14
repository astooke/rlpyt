
import numpy as np
import torch
import cv2
import copy
# from scipy.ndimage.filters import gaussian_filter


##############################################################################
# Helpers
##############################################################################


def numpify(func):
    """Wrapper so that the augmentation function always works on a numpy
    array, but if the input `imgs` is a torch tensor, a torch tensor will be
    returned. Assumes first input and first output of the function is the
    images array/tensor, and only operates on that."""
    def numpified_aug(imgs, *args, **kwargs):
        _numpify = isinstance(imgs, torch.Tensor)
        if _numpify:
            imgs = imgs.numpy()
        ret = func(imgs, *args, **kwargs)
        if _numpify:
            if isinstance(ret, tuple):
                # Assume first is the augmented images.
                ret = (torch.from_numpy(ret[0]), *ret[1:])
            else:
                ret = torch.from_numpy(ret)
        return ret
    return numpified_aug


######################################
#  TRANSLATES
######################################

@numpify
def center_translate(imgs, size):
    n, c, h, w = imgs.shape
    assert size >= h and size >= w
    outs = np.zeros((n, c, size, size), dtype=imgs.dtype)
    h1 = (size - h) // 2
    w1 = (size - w) // 2
    outs[:, :, h1:h1 + h, w1:w1 + w] = imgs
    return outs


@numpify
def random_translate(imgs, size, return_random_idxs=False, h1s=None, w1s=None):
    n, c, h, w = imgs.shape
    assert size >= h and size >= w
    outs = np.zeros((n, c, size, size), dtype=imgs.dtype)
    h1s = np.random.randint(0, size - h + 1, n) if h1s is None else h1s
    w1s = np.random.randint(0, size - w + 1, n) if w1s is None else w1s
    for out, img, h1, w1 in zip(outs, imgs, h1s, w1s):
        out[:, h1:h1 + h, w1:w1 + w] = img
    if return_random_idxs:  # So can do the same to another set of imgs.
        return outs, dict(h1s=h1s, w1s=w1s)
    return outs


##############################################################################
# WINDOWS
##############################################################################

@numpify
def center_window(imgs, window_size, in_place=False):
    n, c, h, w = imgs.shape
    zeros_h = h - window_size
    zeros_w = w - window_size
    min_h = zeros_h // 2
    max_h = h - (zeros_h // 2)
    min_w = zeros_w // 2
    max_w = w - (zeros_w // 2)
    if max_w - min_w > window_size:
        max_w -= 1  # case of odd number
    if max_h - min_h > window_size:
        max_h -= 1  # case of odd number
    imgs = imgs if in_place else imgs.copy()
    imgs[:, :, :min_h, :] = 0.
    imgs[:, :, max_h:, :] = 0.
    imgs[:, :, :, :min_w] = 0.
    imgs[:, :, :, max_w:] = 0.
    return imgs


@numpify
def random_window(imgs, window_size, return_random_idxs=False, in_place=False,
        h1s=None, w1s=None):
    n, c, h, w = imgs.shape
    zeros_h = h - window_size
    zeros_w = w - window_size
    h1s = np.random.randint(low=0, high=zeros_h + 1, size=n) if h1s is None else h1s
    w1s = np.random.randint(low=0, high=zeros_w + 1, size=n) if w1s is None else w1s
    h2s = h1s + window_size
    w2s = w1s + window_size
    imgs = imgs if in_place else imgs.copy()
    for img, h1, h2, w1, w2 in zip(imgs, h1s, h2s, w1s, w2s):
        img[:, :h1, :] = 0.
        img[:, h2:, :] = 0.
        img[:, :, :w1] = 0.
        img[:, :, w2:] = 0.
    if return_random_idxs:
        return imgs, dict(h1s=h1s, w1s=w1s)
    return imgs


##############################################################################
# CROPS
##############################################################################

@numpify
def subpixel_shift(imgs, max_shift=1.):
    """
    Pad input images by 1 using "edge" mode, and then do a nearest-neighbor
    averaging scheme, centered at a random location for each image, up to
    max_shift away from the origin in each x and y.  Each output pixel will
    be a linear interpolation of the surrounding 2x2 input pixels.
    """
    if imgs.dtype == np.uint8:
        raise NotImplementedError
    assert max_shift <= 1.
    b, c, h, w = imgs.shape
    padded = np.pad(
        imgs,
        pad_width=((0, 0), (0, 0), (1, 1), (1, 1)),
        mode="edge",
    )
    xx = np.array([[-1., 0., 1.]])  # [1,3]

    rand_x = max_shift * (2 * np.random.rand(b, 1) - 1)  # [B,1]
    rand_y = max_shift * (2 * np.random.rand(b, 1) - 1)  # [B,1]

    wx = np.maximum(0., 1 - np.abs(xx - rand_x))  # [B,3]
    wy = np.maximum(0., 1 - np.abs(xx - rand_y))  # [B,3]
    weight = wx.reshape(b, 1, 3) * wy.reshape(b, 3, 1)  # [B,1,3]x[B,3,1]->[B,3,3]

    shifted = np.zeros_like(imgs)
    for dy in [0, 1, 2]:
        for dx in [0, 1, 2]:
            shifted += (weight[:, dy, dx].reshape(-1, 1, 1, 1) *
                padded[:, :, dy:h + dy, dx:w + dx])

    return shifted


@numpify
def quick_pad_random_crop(imgs, crop_size=None, pad=None, prob=1., h1s=None, w1s=None):
    t = b = c = 1
    shape_len = len(imgs.shape)
    if shape_len == 2:  # Could also make all this logic into a wrapper.
        h, w = imgs.shape
    elif shape_len == 3:
        c, h, w = imgs.shape
    elif shape_len == 4:
        b, c, h, w = imgs.shape
    elif shape_len == 5:
        t, b, c, h, w = imgs.shape  # Apply same crop to all T
        imgs = imgs.transpose(1, 0, 2, 3, 4)
        _c = c
        c = t * c
        # imgs = imgs.reshape(b, t * c, h, w)
    imgs = imgs.reshape(b, c, h, w)

    # assume square NO LONGER: default to original.
    crop_h = h if crop_size is None else crop_size
    crop_w = w if crop_size is None else crop_size

    # crop_size = h if crop_size is None else crop_size  # default to original

    if pad is not None:
        imgs = np.pad(
            imgs,
            pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)),
            mode="edge",
        )
        b, c, h, w = imgs.shape

    h_max = h - crop_h + 1
    w_max = w - crop_w + 1
    if prob < 1.:
        which_no_crop = np.random.rand(b) > prob
    if h1s is None:
        h1s = np.random.randint(0, h_max, b)
        if prob < 1.:
            h1s[which_no_crop] = pad
    if w1s is None:
        w1s = np.random.randint(0, w_max, b)
        if prob < 1.:
            w1s[which_no_crop] = pad

    cropped = np.empty((b, c, crop_h, crop_w), dtype=imgs.dtype)
    for i, (img, h1, w1) in enumerate(zip(imgs, h1s, w1s)):
        cropped[i] = img[:, h1:h1 + crop_h, w1:w1 + crop_w]

    if shape_len == 2:
        cropped = cropped.reshape(crop_h, crop_w)
    elif shape_len == 3:
        cropped = cropped.reshape(c, crop_h, crop_w)
    elif shape_len == 5:
        cropped = cropped.reshape(b, t, _c, crop_h, crop_w)
        cropped = cropped.transpose(1, 0, 2, 3, 4)
        # cropped = cropped.reshape(t, b, c, crop_size, crop_size)

    return cropped, dict(h1s=h1s, w1s=w1s)


@numpify
def random_crop(imgs, crop_size, resize=None, pad_size=None, pad=None,
        return_random_idxs=False, h1s=None, w1s=None):
    """
    Assumes transformations are into square (h_out=w_out), but input can
    be rectangle.
    args:
        imgs: input images, [N,C,H,W]
        crop_size: int; output H and W
        resize: None, int or 2-tuple of ints; size or range of possible sizes
            (inclusive) to resize imgs into before crop
        pad_size: None or int, how big to make imgs by
            padding (after resize) before crop
        pad: None or int, how much to pad each dim before crop
            (cannot use with pad_size)
        """
    assert pad_size is None or pad is None
    n, c, h, w = imgs.shape

    if resize is not None:
        # resize to square sizes only
        if isinstance(resize, (tuple, list)):  # json doesn't make tuple
            # Sample a random resize within the range.
            assert len(resize) == 2
            resize = np.random.randint(resize[0], resize[1] + 1)
        # Different method prefered for zoom vs shrink.
        # interp = cv2.INTER_AREA if resize < h else cv2.INTER_LINEAR
        temp = np.empty((n, c, resize, resize), dtype=imgs.dtype)
        for i, obs in enumerate(imgs):
            for j, frame in enumerate(obs):
                cv2.resize(src=frame, dst=temp[i][j],
                    dsize=(resize, resize),
                    # interpolation=interp,
                )
        imgs = temp
        h = w = resize

    if pad_size is not None or pad is not None:
        pad_h = pad if pad_size is None else (pad_size - h) // 2
        pad_w = pad if pad_size is None else (pad_size - w) // 2
        imgs = np.pad(imgs,
            pad_width=((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode='edge',
            # constant_values=0,  # if mode "constant"
        )
        n, c, h, w = imgs.shape

    h_max = h - crop_size + 1
    w_max = w - crop_size + 1
    h1s = np.random.randint(0, h_max, n) if h1s is None else h1s
    w1s = np.random.randint(0, w_max, n) if w1s is None else w1s

    cropped = np.empty((n, c, crop_size, crop_size), dtype=imgs.dtype)
    for i, (img, h1, w1) in enumerate(zip(imgs, h1s, w1s)):
        cropped[i] = img[:, h1:h1 + crop_size, w1:w1 + crop_size]

    if return_random_idxs:
        random_idxs_kwargs = dict(
            h1s=h1s,
            w1s=w1s,
            resize=resize,
        )
        return cropped, random_idxs_kwargs
    return cropped


@numpify
def random_crop_rect(imgs, crop_size, resize=None, pad_size=None,
        random_resize=False, return_random_idxs=False, h1_samples=None,
        w1_samples=None):
    """
    Slightly more complicated, allows transformations to be rectangular.
    args:
        imgs: input images, [N,C,H,W]
        crop_size: int or 2-tuple of ints; output H and W
        resize: None, int or 2-tuple of ints; size or range of possible sizes
            (inclusive) to resize imgs into before crop
        pad_size: None, int, or 2-tuple of ints; int, how big to make imgs by
            padding (after resize) before crop
    """
    n, c, h, w = imgs.shape

    if resize is not None:
        # resize to square sizes only
        if isinstance(resize, tuple):
            # Sample a random resize within the range.
            assert len(resize) == 2
            if random_resize:
                if isinstance(resize[0], tuple):
                    assert len(resize[0]) == len(resize[1]) == 2
                    resize_h = np.random.randint(resize[0][0], resize[0][1] + 1)
                    resize_w = np.random.randint(resize[1][0], resize[1][1] + 1)
                else:
                    resize_h = resize_w = np.random.randint(resize[0], resize[1] + 1)
            else:
                resize_h, resize_w = resize
        # Different method prefered for zoom vs shrink.
        # interp = cv2.INTER_AREA if resize_h < h else cv2.INTER_LINEAR
        temp = np.empty((n, c, resize_h, resize_w), dtype=imgs.dtype)
        for i, obs in enumerate(imgs):
            for j, frame in enumerate(obs):
                cv2.resize(src=frame, dst=temp[i][j],
                    dsize=(resize_w, resize_h),  # I think the order reverses.
                    # interpolation=interp,
                )
        imgs = temp
        h = w = resize

    if pad_size is not None:
        if isinstance(pad_size, tuple):
            assert len(pad_size) == 2
            h_size, w_size = pad_size
        else:
            h_size = w_size = pad_size
        pad_h = (h_size - h) // 2
        pad_w = (w_size - w) // 2

        imgs = np.pad(imgs,
            pad_width=((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode="edge",
            # constant_values=0,  # if "constant" mode
        )
        n, c, h, w = imgs.shape

    if isinstance(crop_size, tuple):
        assert len(crop_size) == 2
        h_crop, w_crop = crop_size  # rectangle
    else:
        h_crop = w_crop = crop_size  # square
    h_max = h - h_crop + 1
    w_max = w - w_crop + 1
    h1_samples = h1_samples or np.random.randint(0, h_max, n)
    w1_samples = w1_samples or np.random.randint(0, w_max, n)

    cropped = np.empty((n, c, h_crop, w_crop), dtype=imgs.dtype)
    for i, (img, w1, h1) in enumerate(zip(imgs, w1_samples, h1_samples)):
        cropped[i] = img[:, h1:h1 + h_crop, w1:w1 + w_crop]

    if return_random_idxs:
        return cropped, (h1_samples, w1_samples, (resize_h, resize_w))
    return cropped


@numpify
def center_crop(imgs, crop_size, resize=None, pad_size=None, pad=None):
    """
    Assumes transformations are into square (h_out=w_out), but input can
    be rectangle.
    args:
        imgs: input images, [N,C,H,W]
        crop_size: int; output H and W
        resize: None, int or 2-tuple of ints; size or range of possible sizes
            (inclusive) to resize imgs into before crop
        pad_size: None or int, how big to make imgs by
            padding (after resize) before crop
        pad: None or int, how much to pad each dim before crop 
            (cannot use with pad_size)
    """
    assert pad_size is None or pad is None
    n, c, h, w = imgs.shape

    if resize is not None:
        # resize to square sizes only
        if isinstance(resize, (tuple, list)):  # json doesn't make tuple
            # Sample a random resize within the range.
            assert len(resize) == 2
            resize = np.random.randint(resize[0], resize[1] + 1)
        # Different method prefered for zoom vs shrink.
        # interp = cv2.INTER_AREA if resize < h else cv2.INTER_LINEAR
        temp = np.empty((n, c, resize, resize), dtype=imgs.dtype)
        for i, obs in enumerate(imgs):
            for j, frame in enumerate(obs):
                cv2.resize(src=frame, dst=temp[i][j],
                    dsize=(resize, resize),
                    # interpolation=interp,
                )
        imgs = temp
        h = w = resize

    if pad_size is not None or pad is not None:
        pad_h = pad if pad_size is None else (pad_size - h) // 2
        pad_w = pad if pad_size is None else (pad_size - w) // 2
        imgs = np.pad(imgs,
            pad_width=((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode="edge",
            # constant_values=0,  # if "constant" mode
        )
        n, c, h, w = imgs.shape

    h_max = h - crop_size + 1
    w_max = w - crop_size + 1
    h1 = h_max // 2
    w1 = w_max // 2

    cropped = np.empty((n, c, crop_size, crop_size), dtype=imgs.dtype)
    for i, img in enumerate(imgs):
        cropped[i] = img[:, h1:h1 + crop_size, w1:w1 + crop_size]

    return cropped


@numpify
def center_crop_one(img, crop_size, resize=None, pad_size=None, pad=None):
    """
    Assumes transformations are into square (h_out=w_out), but input can
    be rectangle.
    args:
        imgs: input image, [H,W]
        crop_size: int; output H and W
        resize: None, int or 2-tuple of ints; size or range of possible sizes
            (inclusive) to resize imgs into before crop
        pad_size: None or int, how big to make imgs by
            padding (after resize) before crop
        pad: None or int, how much to pad each dim before crop 
            (cannot use with pad_size)
    """
    assert pad_size is None or pad is None
    h, w = img.shape

    if resize is not None:
        # resize to square sizes only
        if isinstance(resize, (tuple, list)):  # json doesn't make tuple
            # Sample a random resize within the range.
            assert len(resize) == 2
            resize = np.random.randint(resize[0], resize[1] + 1)
        # Different method prefered for zoom vs shrink.
        # interp = cv2.INTER_AREA if resize < h else cv2.INTER_LINEAR
        img = cv2.resize(img, (resize, resize),
            # interpolation=interp,
        )
        h = w = resize

    if pad_size is not None or pad is not None:
        pad_h = pad if pad_size is None else (pad_size - h) // 2
        pad_w = pad if pad_size is None else (pad_size - w) // 2
        img = np.pad(img,
            pad_width=((pad_h, pad_h), (pad_w, pad_w)),
            mode="edge",
            # constant_values=0,  # if "constant" mode
        )
        h, w = img.shape

    h_max = h - crop_size + 1
    w_max = w - crop_size + 1
    h1 = h_max // 2
    w1 = w_max // 2

    cropped = np.empty((crop_size, crop_size), dtype=img.dtype)
    cropped[:] = img[h1:h1 + crop_size, w1:w1 + crop_size]

    return cropped


##############################################################################
# CUTOUTS
##############################################################################


@numpify
def random_cutout(imgs, min_cut, max_cut, random_gray=False, in_place=False):
    n, c, h, w = imgs.shape
    w_cut = np.random.randint(min_cut, max_cut + 1, n)  # random size cut
    h_cut = np.random.randint(min_cut, max_cut + 1, n)  # rectangular shape
    cutouts = imgs if in_place else imgs.copy()
    if random_gray:
        if imgs.dtype == np.uint8:
            fills = np.random.randint(0, 255, n).astype(np.uint8)
        else:
            fills = np.random.rand(n)
    else:
        fills = [0] * n
    for cut, wc, hc, fill in zip(cutouts, w_cut, h_cut, fills):
        w1 = np.random.randint(0, w - wc + 1)  # uniform over interior
        h1 = np.random.randint(0, h - hc + 1)
        cut[:, h1:h1 + hc, w1:w1 + wc] = fill

    return cutouts


##############################################################################
# BLURS (using scipy on cpu, very slow)
##############################################################################


@numpify
def gaussian_blur(imgs, sigma=1.0, in_place=False):
    n, c, h, w = imgs.shape
    blurred = imgs if in_place else np.empty(imgs.shape, dtype=imgs.dtype)
    for blur, img in zip(blurred, imgs):
        # Possibly change image to higher precision before this?
        blur[:] = gaussian_filter(img, sigma=(0, sigma, sigma))
    return blurred


@numpify
def random_gaussian_blur(imgs, min_sigma=0.1, max_sigma=2.0, prob=0.5,
        in_place=False):
    n, c, h, w = imgs.shape
    blurred = imgs if in_place else np.empty(imgs.shape, dtype=imgs.dtype)
    sigmas = np.random.rand(n) * (max_sigma - min_sigma) + min_sigma
    do_blurs = np.random.rand(n) < prob
    for blur, sig, do in zip(blurred, sigmas, do_blurs):
        if do:
            #         # Possibly change image to higher precision before this?
            blur[:] = gaussian_filter(blur, sigma=(0, sig, sig))
    return blurred


@numpify
def motion_blur(imgs, sigma=1.0, in_place=False):
    n, c, h, w = imgs.shape
    blurred = imgs if in_place else np.empty(imgs.shape, dtype=imgs.dtype)
    for blur, img in zip(blurred, imgs):
        # Possibly change image to higher precision before this?
        blur[:] = gaussian_filter(img, sigma=(sigma, 0, 0))
    return blurred


@numpify
def random_motion_blur(imgs, min_sigma=0.1, max_sigma=2.0, prob=0.5,
        in_place=False):
    n, c, h, w = imgs.shape
    blurred = imgs if in_place else imgs.copy()
    sigmas = np.random.rand(n) * (max_sigma - min_sigma) + min_sigma
    do_blurs = np.random.rand(n) < prob
    for blur, sig, do in zip(blurred, sigmas, do_blurs):
        if do:
            blur[:] = gaussian_filter(blur, sigma=(sig, 0, 0))
    return blurred


##############################################################################
# CONTRAST, BRIGHTNESS, ETC
##############################################################################


def _contrast(img, A=0., B=1.):
    """Assumes input image is float in range [0,1].
    A transformation mapping input intensities to output intensities.  The
    transformation is piece-wise linear, with zero-intercept placed at A and
    one-intercept placed at B, constant slope between A and B, and saturated
    values outside that range (identity for A=0, B=1)."""
    return np.clip((img - A) / (B - A), 0., 1.)


def _contrast_torch(img, A=0., B=1.):
    return torch.clamp((img - A) / (B - A), 0., 1.)


@numpify
def grey_jitter(imgs, contrast_eps=0.2, gamma_max=2., contrast_prob=0.5,
        gamma_prob=0.5):
    n, c, h, w = imgs.shape
    assert imgs.dtype == np.uint8
    jittered = imgs.astype(np.float32)
    np.multiply(jittered, 1. / 255, out=jittered)

    do_contrasts = np.random.rand(n) < contrast_prob
    do_gammas = np.random.rand(n) < gamma_prob
    contrast_As = np.random.rand(n) * contrast_eps * 2 - contrast_eps
    contrast_Bs = np.random.rand(n) * contrast_eps * 2 - contrast_eps + 1.
    # Uniform probability over gamma_eps, 50/50 gamma=gamma_eps vs 1/gamma_eps.
    gammas = np.random.rand(n) * (gamma_max - 1) + 1.
    invert_gammas = np.random.rand(n) < 0.5
    gammas[invert_gammas] = 1. / gammas[invert_gammas]
    for i, (do_c, do_g, A, B, g) in enumerate(zip(do_contrasts, do_gammas,
            contrast_As, contrast_Bs, gammas)):
        if do_c:
            jittered[i][:] = _contrast(jittered[i], A, B)
        if do_g:
            jittered[i][:] = jittered[i] ** g

    np.multiply(jittered, 255., out=jittered)
    return jittered.astype(np.uint8)


def grey_jitter_torch(imgs, contrast_eps=0.2, gamma_max=2., contrast_prob=0.5,
        gamma_prob=0.5):
    # Vectorizes all into one call to GPU.
    n, c, h, w = imgs.shape
    if imgs.dtype == torch.uint8:
        jittered = imgs.type(torch.float)
        jittered.mul_(1 / 255.)
    else:
        jittered = imgs  # probably should copy?

    do_contrasts = np.random.rand(n) < contrast_prob
    do_gammas = np.random.rand(n) < gamma_prob
    contrast_As = np.random.rand(n) * contrast_eps * 2 - contrast_eps
    contrast_Bs = np.random.rand(n) * contrast_eps * 2 - contrast_eps + 1.
    # Uniform probability over gamma_eps, 50/50 gamma=gamma_eps vs 1/gamma_eps.
    gammas = np.random.rand(n) * (gamma_max - 1) + 1  # uniform [1, gamma_max]
    invert_gammas = np.random.rand(n) < 0.5
    gammas[invert_gammas] = 1. / gammas[invert_gammas]  # half go 1 / gamma
    
    As = np.zeros(n, dtype=np.float32)
    Bs = np.ones(n, dtype=np.float32)
    As[do_contrasts] = contrast_As[do_contrasts]
    Bs[do_contrasts] = contrast_Bs[do_contrasts]
    As = torch.from_numpy(As).view(n, 1, 1, 1).to(jittered.device)
    Bs = torch.from_numpy(Bs).view(n, 1, 1, 1).to(jittered.device)
    jittered = _contrast_torch(jittered, A=As, B=Bs)
    gams = np.ones(n, dtype=np.float32)
    gams[do_gammas] = gammas[do_gammas]
    gams = torch.from_numpy(gams).view(n, 1, 1, 1).to(jittered.device)
    jittered = jittered ** gams

    # for i, (do_c, do_g, A, B, g) in enumerate(zip(do_contrasts, do_gammas,
    #         contrast_As, contrast_Bs, gammas)):
    #     if do_c:
    #         jittered[i][:] = _contrast_torch(jittered[i], A, B)
    #     if do_g:
    #         jittered[i][:] = jittered[i] ** g

    # For now, just leave it as float in range [0-1], model can handle either
    # this or uint8.
    # jittered.mul_(255.)
    # jittered = jittered.type(torch.uint8)
    return jittered


@numpify
def intensity(imgs, sigma=0.05, clip=2., prob=1.):
    shape_len = len(imgs.shape)
    if shape_len == 2:  # Could also make all this logic into a wrapper.
        # h, w = imgs.shape
        noise_shape = (1,)
    elif shape_len == 3:
        # FNISH HERE
        # c, h, w = imgs.shape
        noise_shape = (1,)
    elif shape_len == 4:
        b, c, h, w = imgs.shape
        noise_shape = (b, 1, 1, 1)
    elif shape_len == 5:
        t, b, c, h, w = imgs.shape  # Apply same noise to all T.
        noise_shape = (1, b, 1, 1, 1)

    noise = 1. + sigma * np.clip(np.random.randn(*noise_shape), -clip, clip)
    if prob < 1.:
        which_no_noise = np.random.rand(*noise_shape) > prob
        noise[which_no_noise] = 1.
    if imgs.dtype == np.uint8:
        noisy = imgs.astype(np.float32)
        np.multiply(noisy, 1. / 255, out=noisy)
    else:
        noisy = imgs.copy()
    noisy[:] = noisy * noise

    return noisy


##############################################################################
# FILTERS: GAUSSIAN, SOBEL, etc (fast on GPU)
##############################################################################


def random_apply_filter(imgs, filter, min_blend=0., max_blend=1., prob=0.5):
    n, c, h, w = imgs.shape
    if imgs.dtype == torch.uint8:
        imgs_ = imgs.type(torch.float)
        imgs_.mul_(1 / 255.)
    else:
        imgs_ = imgs
    filtered = filter(imgs_)
    blends = torch.rand(n, 1, 1, 1) * (max_blend - min_blend) + min_blend
    blends[torch.rand(n) > prob] = 0.
    blends = blends.to(imgs.device)
    return blends * filtered + (1 - blends) * imgs_


def make_square_gaussian_filter(sigma=1.0, kernel_size=7):
    coords = torch.arange(kernel_size, dtype=torch.float) - (kernel_size - 1) / 2.
    grid_h, grid_w = torch.meshgrid([coords, coords])
    kernel = torch.exp(-(grid_h ** 2 + grid_w ** 2) / (2 * sigma ** 2))
    kernel = kernel.view(1, 1, kernel_size, kernel_size) / kernel.sum()
    return kernel


class GaussianFilter(torch.nn.Module):

    def __init__(self, sigma=1.0, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 != 0, "Must use odd-sized kernel."
        self.conv = torch.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            padding_mode="replicate",
            bias=False,
        )
        self.conv.weight.data = make_square_gaussian_filter(sigma=sigma,
            kernel_size=kernel_size)
        self.requires_grad_(False)
        self._sigma = sigma

    def forward(self, input):
        n, c, h, w = input.shape
        return self.conv(input.view(n * c, 1, h, w)).view(n, c, h, w)

    def set_sigma(self, sigma):
        if sigma == self._sigma:
            return
        self.conv.weight.data = make_square_gaussian_filter(sigma=sigma,
            kernel_size=self.conv.kernel_size[0]).to(self.conv.weight.device)
        self._sigma = sigma


def make_sobel_filter(horizontal=True, reverse=False):
    kernel = torch.tensor(
        [[1, 0, -1],
         [2, 0, -2],
         [1, 0, -1]],
        dtype=torch.float)
    # kernel = kernel / kernel.abs().sum()
    kernel = kernel.flip(-1) if reverse else kernel
    kernel = kernel if horizontal else kernel.transpose(1, 0)
    return kernel.view(1, 1, 3, 3)


class SobelFilter(torch.nn.Module):

    def __init__(self, scale="normalize"):
        super().__init__()
        self.convx = torch.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="replicate",
            bias=False,
        )
        self.convy = copy.deepcopy(self.convx)
        self.convx.weight.data = make_sobel_filter(horizontal=True)
        self.convy.weight.data = make_sobel_filter(horizontal=False)
        self.requires_grad_(False)
        # Auto-scaling computes one scaling for each batch.
        self.scale = scale
        assert scale in [None, "normalize", "match_input", 
        "normalize_each", "match_input_each"]

    def forward(self, input):
        n, c, h, w = input.shape
        inp = input.view(n * c, 1, h, w)
        gx = self.convx(inp)
        gy = self.convy(inp)
        g = torch.sqrt(gx ** 2 + gy ** 2).view(n, c, h, w)
        if self.scale == "normalize":
            g_max = torch.clamp(g.max(), min=1e-6)
            g = g / g_max  # [0, 1] across the whole batch
        elif self.scale == "normalize_each":
            g_maxs = torch.clamp(g.view(n, -1).max(dim=-1)[0], min=1e-6).view(n, 1, 1, 1)
            g = g / g_maxs  # each along leading dim is [0, 1]
        elif self.scale == "match_input":
            min_in, max_in = input.min(), input.max()
            g_max = torch.clamp(g.max(), min=1e-6)
            g = g / g_max * (max_in - min_in) + min_in  # [min_in, max_in]
        elif self.scale == "match_input_each":
            mins_in = input.view(n, -1).min(dim=-1)[0].view(n, 1, 1, 1)
            maxs_in = input.view(n, -1).max(dim=-1)[0].view(n, 1, 1, 1)
            g_maxs = torch.clamp(g.view(n, -1).max(dim=-1)[0], min=1e-6).view(n, 1, 1, 1)
            g = g / g_maxs * (maxs_in - mins_in) + mins_in
        return g
