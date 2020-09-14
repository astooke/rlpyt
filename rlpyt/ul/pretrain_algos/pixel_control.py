
import os.path as osp
import sys
import torch
import torch.nn.functional as F
import pickle
import numpy as np

from rlpyt.algos.utils import discount_return



def compute_pixel_control_returns(
        frames,
        done,
        discount,
        crop_h,
        crop_w,
        cell_h,
        cell_w,
    ):
    """
    Compute rewards as the absolute pixel value differences, averaged within
    each cell and across channels. Then compute the returns as full MC
    returns--no bootstrapping because we already have the full replay buffer
    (the last trajectories will get cut off, shouldn't affect too many samples
    with small discount factor, typically 0.9).

    Inputs `frame` and `done` can be torch tensors or numpy arrays and the
    outputs will be the same type (the operations are done in pytorch).

    Args:
        frames: new frame at each time-step, shaped [T,B,C,H,W] (e.g. C=1 for greyscale)
        done: indicator when episode is done, shaped [T,B]
        discount (float): discount factor for computing monte carlo returns
        crop_h (int): size of image to use (typically 80, to crop from 84)
        crop_w (int): same idea as crop_h
        cell_h (int): size of cell for averaging, typically 4
        cell_w (int): same idea as cell_w

    Returns `reward, return_` both shaped [T,B,H',C'] (last time-step is bogus)
    
    Example sizes: frames H=84, crop_h=80, cell_h=4 --> H'=80/4=20.
    """
    
    _numpy = isinstance(frames, np.ndarray)  # frames from replay buffer
    if _numpy:
        frames = torch.from_numpy(frames)
        done = torch.from_numpy(done)
    if frames.dtype == torch.uint8:
        frames = frames.type(torch.float)
        frames = frames.mul_(1. / 255)
    T, B, C, H, W = frames.shape
    #  Should just be the newest frame at each time-step.
    # So for Atari (greyscale), C=1.  For RGB, C=3.
    if crop_h < H:
        min_h = (H - crop_h) // 2
        max_h = - min_h  # Not sure if cutoff length not div by 2?
        frames = frames[:, :, :, min_h:max_h]
    if crop_w < W:
        min_w = (W - crop_w) // 2
        max_w = - min_w
        frames = frames[:, :, :, :, min_w:max_w]
    T, B, C, H, W = frames.shape
    # Leave an extra time-step at the end, which will have bogus values but
    # will be easier to handle in the replays because it maintains length T.
    abs_diff = torch.zeros(T, B, C, H, W)
    abs_diff[:-1] = torch.abs(frames[1:] - frames[:-1])
    abs_diff = abs_diff.view(T * B, C, H, W)
    pooled_diff = F.avg_pool2d(
        input=abs_diff,
        kernel_size=(cell_h, cell_w),
    )  # [T*B,C,H',W']
    pooled_diff = pooled_diff.mean(dim=1)  # [T*B,H',W']
    _, Hp, Wp = pooled_diff.shape
    reward = pooled_diff.view(T, B, Hp, Wp)
    # Repeat the last value as probably more correct than zeros:
    reward[-1] = reward[-2]

    # Make reward=0 where done=True (the next frame was from after reset).
    done = done.type(torch.bool)  # Should already be the case
    reward[done] = 0.

    # This function just assumes leading dim is T, rest of dims can all be
    # left alone, as long as `done` broadcasts correctly.
    return_ = discount_return(
        reward=reward,
        done=done.type(reward.dtype).unsqueeze(-1).unsqueeze(-1),
        bootstrap_value=0.,
        discount=discount,
    )
    if _numpy:
        reward = reward.numpy()
        return_ = return_.numpy()
    return reward, return_


def save_pixel_control_returns(
        replay_buffer_filename="replaybuffer.pkl",
        discount=0.9,
        crop_h=80,
        crop_w=80,
        cell_h=4,
        cell_w=4,
        min_c_new_frame=-1,  # Used if NOT frame-based buffer.
        max_c_new_frame=None,
        overwrite=False,
    ):
    """Puts MonteCarlo Pixel Control returns into a replay buffer.
    
    This computes pretty fast, so usually just have all this logic
    inside the FixedReplayBuffer class and compute during initialization
    in a pre-training run. (e.g. Then can use different cell sizes).

    If the replay buffer is frame-based, this will automatically
    grab the new frame from each time-step.  This is usually the
    case for Atari.

    If the replay buffer is not frame-based, this will grab the
    channels from the observation as min_c_new_frame:max_c_new_frame.
    For example, to grab only the last frame (if stored OLDEST to NEWEST),
    leave those as (-1, None), respectively.  For DMControl it's probably
    a history of RGB frames, so grab (-3,None) if stored OLDEST to NEWEST. 
    """
    pc_filename = f"pixel_control_{crop_h}x{crop_w}_{cell_h}x{cell_w}.pkl"
    if osp.isfile(pc_filename) and not overwrite:
        print(f"Already found file {pc_filename} and not overwriting.")
        return
    with open(replay_buffer_filename, "rb") as fh:
        replay_buffer = pickle.load(fh)
    if hasattr(replay_buffer, "samples_new_frames"):
        frames = replay_buffer.samples_new_frames
        if len(frames.shape) == 4:  # [T,B,H,W]
            frames = np.expand_dims(frames, axis=2)  # [T,B,C=1,H,W]
    else:
        frames = replay_buffer.samples.observation[:, :,
            min_c_new_frame:max_c_new_frame]
    done = replay_buffer.samples.done
    pix_ctl_reward, pix_ctl_return = compute_pixel_control_returns(
        frames=frames,
        done=done,
        discount=discount,
        crop_h=crop_h,
        crop_w=crop_w,
        cell_h=cell_h,
        cell_w=cell_w,
    )
    pixel_control = dict(
        reward=pix_ctl_reward,
        return_=pix_ctl_return,
        discount=discount,
        crop_h=crop_h,
        crop_w=crop_w,
        cell_h=cell_h,
        cell_w=cell_w,
    )
    with open(f"pixel_control_{crop_h}x{crop_w}_{cell_h}x{cell_w}.pkl",
            "wb") as fh:
        pickle.dump(pixel_control, fh, protocol=4)


def save_pixel_control_returns_dmlab(
        replay_buffer_filename="replaybuffer.pkl",
        discount=0.9,
        crop_h=72,
        crop_w=96,
        cell_h=4,
        cell_w=4,
        min_c_new_frame=-3,  # Use all channels (just RGB)
        max_c_new_frame=None,
        overwrite=False,
    ):
    save_pixel_control_returns(
        replay_buffer_filename=replay_buffer_filename,
        discount=discount,
        crop_h=crop_h,
        crop_w=crop_w,
        cell_h=cell_h,
        cell_w=cell_w,
        min_c_new_frame=min_c_new_frame,  # Use all channels (just RGB)
        max_c_new_frame=max_c_new_frame,
        overwrite=overwrite,
    )



if __name__ == "__main__":
    save_pixel_control_returns(*sys.argv[1:])