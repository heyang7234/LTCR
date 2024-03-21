import numpy as np
import torch


# def DataTransform(sample, config):

#     weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
#     strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)

#     return weak_aug, strong_aug

def DataTransform(sample):

    weak_aug = scaling(sample, 0.001)
    # strong_aug = jitter(permutation(sample, max_segments=5), 0.001)
    strong_aug = jitter(permutation(sample, max_segments=5), 0.001)

    # weak_aug = scaling(sample, 0.001)
    # strong_aug = jitter(scaling(sample), 0.1)

    return weak_aug, strong_aug


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                # print(split_points) # [216]
                splits = np.split(orig_steps, split_points)
                # print(splits) # [[0-215] [216-255]]
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            # print(len(splits)) # 2 
            # shuffled_indices = np.random.permutation(np.arange(len(splits)))
            shuffled_indices = np.random.permutation(np.arange(len(splits)))
            # print(shuffled_indices)
            shuffled_splits = [splits[i] for i in shuffled_indices]
            # print(shuffled_splits)
            # warp = np.concatenate(np.random.permutation(splits)).ravel()
            warp = np.concatenate(shuffled_splits).ravel() # 
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    # return torch.from_numpy(ret)
    return ret

