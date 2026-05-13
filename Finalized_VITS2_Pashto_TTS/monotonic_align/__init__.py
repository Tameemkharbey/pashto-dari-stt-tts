import numpy as np
import torch


def _maximum_path_c_numpy(paths, values, t_ys, t_xs, max_neg_val=-1e9):
    """Pure NumPy port of the Cython maximum_path_c."""
    for i in range(paths.shape[0]):
        path = paths[i]
        value = values[i]
        t_y, t_x = int(t_ys[i]), int(t_xs[i])
        index = t_x - 1

        for y in range(t_y):
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                v_cur = max_neg_val if x == y else float(value[y - 1, x])
                if x == 0:
                    v_prev = 0.0 if y == 0 else max_neg_val
                else:
                    v_prev = float(value[y - 1, x - 1])
                value[y, x] += max(v_prev, v_cur)

        for y in range(t_y - 1, -1, -1):
            path[y, index] = 1
            if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
                index -= 1


try:
    from .monotonic_align.core import maximum_path_c as _maximum_path_c
    _USE_CYTHON = True
except ImportError:
    _maximum_path_c = None
    _USE_CYTHON = False


def maximum_path(neg_cent, mask):
    """
    neg_cent: [b, t_t, t_s]
    mask: [b, t_t, t_s]
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent_np = neg_cent.data.cpu().numpy().astype(np.float32)
    path = np.zeros(neg_cent_np.shape, dtype=np.int32)

    t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
    t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)

    if _USE_CYTHON:
        _maximum_path_c(path, neg_cent_np, t_t_max, t_s_max)
    else:
        _maximum_path_c_numpy(path, neg_cent_np, t_t_max, t_s_max)

    return torch.from_numpy(path).to(device=device, dtype=dtype)
