import numpy as np
from scipy.ndimage.morphology import distance_transform_edt


def mask_to_onehot(mask, num_classes=6):

    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)


def onehot_to_mask(mask):

    _mask = np.argmax(mask, axis=0)

    return _mask


def onehot_to_binary_edges(mask, radius, num_classes=6):

    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    edgemap = np.zeros(mask.shape[1:])

    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = np.expand_dims(edgemap, axis=0)
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap
