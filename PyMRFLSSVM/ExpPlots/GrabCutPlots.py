# -*- coding: utf-8 -*-
import cv2
from skimage.segmentation import quickshift, slic, felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import pickle

__author__ = 'spacegoing'

############################ GrabCut Exps ############################
stats_data_dir = '/Users/spacegoing/macCodeLab-MBP2015/Python/ExpResults_PyMRFLSSVM/'
with open(stats_data_dir + '50_results.pickle', 'rb') as f:
    name_infLabel_loss_y_dict = pickle.load(f)

# %%%%%%%%%%%%%%%%% Accuracy Histogram %%%%%%%%%%%%%%%%%%%%
# acc_list = list()
# for name, i in name_infLabel_loss_y_dict.items():
#     acc_list.append(1 - i['loss'])
# acc_list = np.asarray(acc_list)
# a = np.sum(acc_list>=0.90)
# np.mean(acc_list)
# plt.hist(acc_list)

avg_error = 0
worst_loss_list = list()
worst_name_list = list()
for name, i in name_infLabel_loss_y_dict.items():
    print(i['loss'])
    if i['loss'] > 0.05:
        worst_loss_list.append(1 - i['loss'])
        worst_name_list.append(str(name))
    else:
        avg_error += i['loss']

print(1 - avg_error / len(name_infLabel_loss_y_dict))

pami_imgs_name = ['326038', '124080', 'person5', '24077']
pami_imgs_ext = ['.jpg', '.jpg', '.jpg', '.jpg']


# %%%%%%%%%%%%%%%% GrabCut Plot %%%%%%%%%%%%%%%
def plot_img(name, ext, mask):
    img_path = './GrabCut/Data/grabCut/images/' + name + ext

    img = cv2.imread(img_path)
    mask = mask.astype('uint8')
    img = img * mask[:, :, np.newaxis]
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.savefig("./StatsData/" + name + ".png")


for name, ext in zip(pami_imgs_name, pami_imgs_ext):
    mask = name_infLabel_loss_y_dict[name]["inferred_label"]
    plot_img(name, ext, mask)

# worst_name = "189080"
# worst_ext = ".jpg"
# worst_mask = name_infLabel_loss_y_dict[worst_name]["inferred_label"]
# plot_img(worst_name, worst_ext, worst_mask)

############################ Superpixel ##############################
img_path = "/Users/spacegoing/macCodeLab-MBP2015/Python/" \
           "MRFLSVM/PyMRFLSSVM/GrabCut/Data/grabCut/images/banana1.bmp"
img = io.imread(img_path)
image = img_as_float(img)
segments = slic(img, 300, sigma=5)

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(mark_boundaries(image, segments))
plt.axis('off')

# %%%%%%%%%%%%%%%% GrabCut Plot %%%%%%%%%%%%%%%

name = 'person2'
img_path = './GrabCut/Data/grabCut/images/' + name + '.bmp'
mask_path = './GrabCut/Data/grabCut/labels/' + name + '.bmp'

img = cv2.imread(img_path)
mask = cv2.imread(mask_path)
mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = img * mask[:, :, :]
plt.axis("off")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
