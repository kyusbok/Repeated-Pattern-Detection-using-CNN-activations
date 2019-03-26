import os
import pickle

import torch
import torch.nn as nn
from torchvision import models, transforms
from AlexNetConvLayers import alexnet_conv_layers

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageFilter
from scipy.ndimage.filters import maximum_filter, gaussian_filter
from scipy.stats import multivariate_normal
from skimage.feature import peak_local_max

from utils import custom_plot


#parameters
sigma_l = []
alfa_l = [5, 7, 15, 15, 15]
fi_prctile = 80
delta = 0.65

subsample_pairs = 10
peaks_max = 10000


preprocess_transform = transforms.Compose([transforms.ToTensor()])

dev = torch.device("cuda")

def load_image(img_path):
    image = Image.open(img_path).convert('RGB')
    return preprocess_transform(image).unsqueeze(0).to(dev)

model = alexnet_conv_layers()
model.to(dev)

img_path = "dotted2.png"
image = load_image(img_path)

#conv features computation
conv_feats = model(image)

#peaks extraction
peaks = []
for li, l in enumerate(conv_feats):
    peaks.append([])
    maps = l.squeeze().detach().cpu().numpy()
    sigma_l.append((image.size(2) / maps.shape[1]) / 2)

    # #visualization
    # for fi, fmap in enumerate(maps[:5]):
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(fmap)
    #     plt.subplot(1, 2, 2)
    #     tmp_max = maximum_filter(fmap, 1)
    #     max_coords = peak_local_max(tmp_max, 1)
    #     plt.imshow(peak_local_max(tmp_max, 1, indices=False))
    #     plt.waitforbuttonpress()
    for fi, fmap in enumerate(maps):
        fmap = np.array(Image.fromarray(fmap).resize((image.size(3), image.size(2))))
        # tmp_max = maximum_filter(fmap, 1)
        # max_coords = peak_local_max(tmp_max, 5)

        # plt.subplot(1, 2, 1)
        # plt.imshow(fmap)

        fmap = gaussian_filter(fmap, sigma=10)
        tmp_max = maximum_filter(fmap, 1)
        max_coords = peak_local_max(tmp_max, 5)

        # plt.subplot(1, 2, 2)
        # plt.imshow(fmap)
        # plt.waitforbuttonpress()

        peaks[li].append(max_coords[np.random.permutation(max_coords.shape[0])[:peaks_max]])

#compute displacement set and voting space
pickefile = "V_" + os.path.basename(img_path) + ".pkl"
if os.path.exists(pickefile):
    with open(pickefile, 'rb') as f:
        V = pickle.load(f)
else:
    quant_r, quant_c = np.mgrid[0:image.size(2):1, 0:image.size(3):1]
    V = np.zeros(quant_r.shape)
    quant_rc = np.empty(quant_r.shape + (2,), dtype=np.float32)
    quant_rc[:, :, 0] = quant_r
    quant_rc[:, :, 1] = quant_c
    disps = []
    for li, p in enumerate(peaks):
        disps.append([])
        for fi, p2 in enumerate(p):
            # pairs_inds = np.asarray([(i, j) for i in range(p2.shape[0]) for j in range(p2.shape[0]) if i != j and j > i])
            pairs_inds = np.asarray(np.meshgrid(np.arange(p2.shape[0]), np.arange(p2.shape[0])), dtype=np.uint8).T.reshape(-1, 2)
            pairs_inds = pairs_inds[pairs_inds[:, 0] > pairs_inds[:, 1]]
            if pairs_inds.shape[0] > 0:
                tmp_disps = np.abs(p2[pairs_inds[:, 0]] - p2[pairs_inds[:, 1]])
            else:
                tmp_disps = np.asarray([[]])
            if tmp_disps.size == 0:
                continue
            tmp_disps = tmp_disps[np.random.permutation(tmp_disps.shape[0])[:subsample_pairs]]
            # disps[li].append(tmp_disps)
            #tmp_disps è Dfl
            for ij, dij in enumerate(tmp_disps):
                tmp_Vfiij = multivariate_normal.pdf(quant_rc, mean=dij
                                                    , cov=np.asarray([[sigma_l[li], 0]
                                                                     , [0, sigma_l[li]]], dtype=np.float32))
                tmp_Vfiij /= tmp_disps.shape[0]
                V += tmp_Vfiij

    with open(pickefile, 'wb') as handle:
        pickle.dump(V, handle, protocol=pickle.HIGHEST_PROTOCOL)


#find best step
starting_ind = 10
#TODO qualcosa per pesare di più gli step più piccoli
# dstar = np.asarray(((V[:, 0] / np.arange(0, V.shape[0], 1))[starting_ind:].argmax() + starting_ind
#                    , (V[0, :] / np.arange(0, V.shape[1], 1))[starting_ind:].argmax() + starting_ind))

dstar = np.asarray((V[starting_ind:, 0].argmax() + starting_ind
                   , V[0, starting_ind:].argmax() + starting_ind))

#compute consistent votes to compute fi
fi_acc = []
for li, p in enumerate(peaks):
    for fi, p2 in enumerate(p):
        pairs_inds = np.asarray(np.meshgrid(np.arange(p2.shape[0]), np.arange(p2.shape[0])), dtype=np.uint8).T.reshape(
            -1, 2)
        pairs_inds = pairs_inds[pairs_inds[:, 0] > pairs_inds[:, 1]]
        if pairs_inds.shape[0] > 0:
            tmp_disps = np.abs(p2[pairs_inds[:, 0]] - p2[pairs_inds[:, 1]])
        else:
            fi_acc.append(0)
            continue
        tmp_disps = tmp_disps[np.random.permutation(tmp_disps.shape[0])[:subsample_pairs]]
        fi_acc.append(len([1 for dij in tmp_disps if (np.linalg.norm(dij - dstar)) < 3 * alfa_l[li]]))
#is this correct??
param_fi = np.percentile(fi_acc, fi_prctile)

#find weights for filters
disps_star = []
weights = []
for li, p in enumerate(peaks):
    disps_star.append([])
    weights.append([])
    for fi, p2 in enumerate(p):
        # pairs_inds = np.asarray([(i, j) for i in range(p2.shape[0]) for j in range(p2.shape[0]) if i != j and j > i])
        pairs_inds = np.asarray(np.meshgrid(np.arange(p2.shape[0]), np.arange(p2.shape[0])), dtype=np.uint8).T.reshape(
            -1, 2)
        pairs_inds = pairs_inds[pairs_inds[:, 0] > pairs_inds[:, 1]]
        if pairs_inds.shape[0] > 0:
            tmp_disps = np.abs(p2[pairs_inds[:, 0]] - p2[pairs_inds[:, 1]])
        else:
            tmp_disps = np.asarray([[]])
        weights[li].append(0)
        if tmp_disps.size == 0:
            continue
        tmp_disps = tmp_disps[np.random.permutation(tmp_disps.shape[0])[:subsample_pairs]]
        # disps_star[li].append(tmp_disps)
        # tmp_disps è Dfl

        for ij, dij in enumerate(tmp_disps):
            tmp_diff = np.linalg.norm(dij - dstar)
            if tmp_diff < 3 * alfa_l[li]:
                # φ è 80esimo percentile, bisogna sommare i pesi per calcolare per ogni filtro
                wijfl = np.exp(-(tmp_diff ** 2)
                               / (2 * (alfa_l[li] ** 2))) \
                        / (tmp_disps.shape[0] + param_fi)
                weights[li][-1] += wijfl

#find filters with weights higher than threshold
selected_filters = []
for li, w in enumerate(weights):
    tmp_weight_thr = delta * max(w)
    selected_filters.append([fi for fi, w2 in enumerate(w) if w2 > tmp_weight_thr])

#accumulate origin coordinates loss
acc_origin = []
acc_origin_weights = []
for li, w in enumerate(weights):
    for fi in selected_filters[li]:
        p2 = peaks[li][fi]
        pairs_inds = np.asarray(np.meshgrid(np.arange(p2.shape[0]), np.arange(p2.shape[0])), dtype=np.uint8).T.reshape(
            -1, 2)
        pairs_inds = pairs_inds[pairs_inds[:, 0] > pairs_inds[:, 1]]
        if pairs_inds.shape[0] > 0:
            tmp_disps = np.abs(p2[pairs_inds[:, 0]] - p2[pairs_inds[:, 1]])
        else:
            fi_acc.append(0)
            continue
        cons_disps = [dij for ij, dij in enumerate(tmp_disps)
                                    if (np.linalg.norm(dij - dstar)) < 3 * alfa_l[li]]
        cons_disps_weights = [np.exp(-(np.linalg.norm(dij - dstar) ** 2)/ (2 * (alfa_l[li] ** 2))) / (tmp_disps.shape[0] + param_fi)
                              for dij in cons_disps]
        acc_origin.extend(cons_disps)
        acc_origin_weights.extend(cons_disps_weights)

o_r = np.linspace(-dstar[0], dstar[0], 10)
o_c = np.linspace(-dstar[1], dstar[1], 10)
min_rc = (-1, -1)
min_val = np.inf
for r in o_r:
    for c in o_c:
        tmp_orig = np.asarray([r, c])
        tmp_val = [np.linalg.norm(np.mod((dij - tmp_orig), dstar) - (dstar / 2)) * acc_origin_weights[ij]
                      for ij, dij in enumerate(acc_origin)]
        tmp_val = np.sum(tmp_val)
        if tmp_val < min_val:
            min_val = tmp_val
            min_rc = (r, c)


boxes = []
tmp_img = np.array(Image.open(img_path))
for ri in range(100):
    min_r = min_rc[0] + (dstar[0] * ri) - (dstar[1] / 2)
    if min_r > tmp_img.shape[0]:
        break
    for ci in range(100):
        min_c = min_rc[1] + (dstar[1] * ci) - dstar[0] / 2
        if min_c > tmp_img.shape[1]:
            break
        tmp_box = np.asarray([min_c, min_r, dstar[1], dstar[0]])
        boxes.append(tmp_box)

custom_plot(tmp_img, boxes)

pass