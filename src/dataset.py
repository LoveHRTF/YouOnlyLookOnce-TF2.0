from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import numpy as np
import random
import config as cfg


class Dataset(object):

    def __init__(self, common_params, file, train=True):
        super(Dataset, self).__init__()
        self.image_size = common_params['image_size']
        self.batch_size = common_params['batch_size']
        self.file = file
        self.grid_num = common_params['output_size']

        self.train = train
        self.fnames = []
        self.boxes = []
        self.labels = []
        self.mean = (103.939, 116.779, 123.68)  # bgr
        fs_input = open(self.file, 'r')

        for line in fs_input.readlines():
            line = line.strip().split(' ')
            self.fnames.append(line[0])
            num_boxes = (len(line) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                x = float(line[1 + 5 * i])
                y = float(line[2 + 5 * i])
                x2 = float(line[3 + 5 * i])
                y2 = float(line[4 + 5 * i])
                c = line[5 + 5 * i]
                box.append([x, y, x2, y2])
                label.append(int(c) + 1)
            self.boxes.append(box)
            self.labels.append(label)
            fs_input.close()

        self.record_point = 0
        self.total_samples = len(self.fnames)
        self.num_batch_per_epoch = int(self.total_samples / self.batch_size)

    def parse_data(self, idx):
        fname = self.fnames[idx]
        img = cv2.imread(fname)
        boxes = np.array(self.boxes[idx]).copy()
        labels = np.array(self.labels[idx]).copy()

        if self.train:
            img, boxes = self.random_flip(img, boxes)
            img, boxes = self.random_scale(img, boxes)
            img = self.random_blur(img)
            img = self.random_brightness(img)
            img = self.random_hue(img)
            img = self.random_saturation(img)
            img, boxes, labels = self.random_shift(img, boxes, labels)
            img, boxes, labels = self.random_crop(img, boxes, labels)

        h, w, _ = img.shape
        boxes /= [w, h, w, h]
        # img = self.sub_mean(img, self.mean)
        img = cv2.resize(img, (self.image_size, self.image_size))
        target = self.encode(boxes, labels)
        return img, target, fname

    def encode(self, boxes, labels):
        """
        :param boxes: a 2D array holding the bounding boxes
        :param labels: ground truth
        :return: a 3D array of size [grid_num, grid_num, 30], where each cell has a gt vector of size 30 containing
        [x, y, w, h, confidence] * 2 + one-hot vector of 20 classes
        """
        target = np.zeros((self.grid_num, self.grid_num, 30))
        cell_size = 1. / self.grid_num
        wh = (boxes[:, 2:] - boxes[:, 0:2])
        cxcy = (boxes[:, 2:] + boxes[:, 0:2]) / 2

        for i in range(cxcy.shape[0]):
            _cxcy = cxcy[i]
            ij = (np.ceil(_cxcy / cell_size) - 1).astype(np.int32)
            target[ij[1], ij[0], 4] = 1
            target[ij[1], ij[0], 9] = 1
            target[ij[1], ij[0], int(labels[i]) + 9] = 1
            xy = ij * cell_size
            delta_xy = (_cxcy - xy) / cell_size
            target[ij[1], ij[0], 2:4] = wh[i]
            target[ij[1], ij[0], 0:2] = delta_xy
            target[ij[1], ij[0], 7:9] = wh[i]
            target[ij[1], ij[0], 5:7] = delta_xy
        return target

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def RGB2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def sub_mean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h, w, _ = im.shape
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return im_lr, boxes
        return im, boxes

    def random_scale(self, bgr, boxes):
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            height, width, c = bgr.shape
            bgr = cv2.resize(bgr, (int(width * scale), height))
            scale_boxes = [scale, 1, scale, 1]
            boxes = boxes * scale_boxes
            return bgr, boxes
        return bgr, boxes

    def random_blur(self, bgr):
        if random.random() < 0.5:
            bgr = cv2.blur(bgr, (5, 5))
        return bgr

    def random_brightness(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def random_hue(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            h = h * adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def random_saturation(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s * adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def random_shift(self, bgr, boxes, labels):
        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        if random.random() < 0.5:
            height, width, c = bgr.shape
            after_shfit_image = np.zeros((height, width, c), dtype=bgr.dtype)
            after_shfit_image[:, :, :] = (104, 117, 123)  # bgr
            shift_x = random.uniform(-width * 0.2, width * 0.2)
            shift_y = random.uniform(-height * 0.2, height * 0.2)

            if shift_x >= 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, int(shift_x):, :] = bgr[:height - int(shift_y), :width - int(shift_x),
                                                                     :]
            elif shift_x >= 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), int(shift_x):, :] = bgr[-int(shift_y):, :width - int(shift_x),
                                                 :]
            elif shift_x < 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, :width + int(shift_x), :] = bgr[:height - int(shift_y), -int(shift_x):,
                                                                             :]
            elif shift_x < 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), :width + int(shift_x), :] = bgr[-int(shift_y):,
                                                                                      -int(shift_x):, :]

            shift_xy = [int(shift_x), int(shift_y)]
            center = center + shift_xy
            mask1 = np.where((center[:, 0] > 0) & (center[:, 0] < width))[0]
            mask2 = np.where((center[:, 1] > 0) & (center[:, 1] < height))[0]
            mask = np.intersect1d(mask1, mask2)
            boxes_in = boxes[mask]
            if len(boxes_in) == 0:
                return bgr, boxes, labels
            box_shift = [int(shift_x), int(shift_y), int(shift_x), int(shift_y)]
            boxes_in = boxes_in + box_shift
            labels_in = labels[mask]
            return after_shfit_image, boxes_in, labels_in
        return bgr, boxes, labels

    def random_crop(self, bgr, boxes, labels):
        if random.random() < 0.5:
            center = (boxes[:, 2:] + boxes[:, :2]) / 2
            height, width, c = bgr.shape
            h = random.uniform(0.6 * height, height)
            w = random.uniform(0.6 * width, width)
            x = random.uniform(0, width - w)
            y = random.uniform(0, height - h)
            x, y, h, w = int(x), int(y), int(h), int(w)

            center = center - [x, y]
            mask1 = np.where((center[:, 0] > 0) & (center[:, 0] < w))[0]
            mask2 = np.where((center[:, 1] > 0) & (center[:, 1] < h))[0]
            mask = np.intersect1d(mask1, mask2)

            boxes_in = boxes[mask]
            if (len(boxes_in) == 0):
                return bgr, boxes, labels
            box_shift = [x, y, x, y]

            boxes_in = boxes_in - box_shift
            boxes_in[:, 0] = boxes_in[:, 0].clip(min=0, max=w)
            boxes_in[:, 2] = boxes_in[:, 2].clip(min=0, max=w)
            boxes_in[:, 1] = boxes_in[:, 1].clip(min=0, max=h)
            boxes_in[:, 3] = boxes_in[:, 3].clip(min=0, max=h)

            labels_in = labels[mask]
            img_croped = bgr[y:y + h, x:x + w, :]
            return img_croped, boxes_in, labels_in
        return bgr, boxes, labels

    def batch(self):
        """Get a batch of data for training or testing"""
        if self.record_point % self.num_batch_per_epoch == 0:
            self.shuffle_idx = np.random.permutation(self.total_samples) if self.train else np.arange(self.total_samples)
            self.record_point = 0

        images, targets, fnames = [], [], []
        idxs = self.shuffle_idx[self.record_point * self.batch_size: (self.record_point + 1) * self.batch_size]
        for idx in idxs:
            image, target, fname = self.parse_data(idx)
            images.append(image)
            targets.append(target)
            fnames.append(fname)
        images = np.asarray(images, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)

        self.record_point += 1
        return images, targets, fnames


if __name__ == '__main__':
    # Example of getting a batch of training data
    # To get test data, pass another param train=False to Dataset constructor
    dataset = Dataset(cfg.common_params, cfg.dataset_params['train_file'])
    for i in range(100):
        images, targets, _ = dataset.batch()
        print(targets[0, :, :, 0:5])
        print(images.shape, targets.shape)
