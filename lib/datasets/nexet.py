# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

from datasets.imdb import imdb
import datasets.nexet_challenge_eval
import datasets.ds_utils as ds_utils
from fast_rcnn.config import cfg
import os.path as osp
import sys
import os
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import json
import uuid


class nexet(imdb):
    def __init__(self, image_set, year, integrate_classes=True):
        name = 'nexet_' + year + '_' + image_set
        if image_set == "train":
            if integrate_classes:
                name += "_1label"
            else:
                name += "_5labels"
        imdb.__init__(self, name)

        # name, paths
        self._image_set = image_set
        self._data_path = osp.join(cfg.DATA_DIR, "nexet" + year, image_set)
        self._integrate_classes = integrate_classes
        self._classes = ('__background__',  # always index 0
                         "car", "van", "truck", "bus", "pickup_truck") \
            if not self._integrate_classes else ("__bakcground__", "vehicle")
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        #self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        #self._salt = str(uuid.uuid4())
        #self._comp_id = "comp4"

        assert os.path.exists(self._data_path), "Path does not exist: {}".format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        :param i:
        :return:
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier
        :param index:
        :return:
        """
        # TODO : Need folder JPEGImages?
        image_path = os.path.join(self._data_path, "images", index)
        assert os.path.exists(image_path), "Path does not exist: {}".format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        :return:
        """
        # image_set_file = os.path.join(self._data_path, "train.csv")
        image_set_file = os.path.join(self._data_path, "train_boxes.csv")

        assert os.path.exists(image_set_file), "Path does not exist: {}".format(image_set_file)

        images = {}
        with open(image_set_file) as f:
            header = True
            for x in f:
                if header:
                    header = False
                    continue
                images[x.split(',')[0]] = True
        return images.keys()

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        :return:
        """
        cache_file = os.path.join(self.cache_path, self.name + "_gt_roidb.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding="bytes")
            print("{} gt roidb loaded from {}".format(self.name, cache_file))
            return roidb

        filename = os.path.join(self._data_path, "train_boxes.csv")

        index2objs = defaultdict(list)

        with open(filename) as f:
            header = True
            for x in f:
                if header:
                    header = False
                    continue

                obj = x.split(",")
                index2objs[obj[0]].append(obj)


            # objs = [x.split(",") for x in f.readlines()[1:] if x.split(",")[0].strip().__eq__(index)]

        gt_roidb = []
        for idx in self.image_index:
            objs = index2objs[idx]

            num_objs = len(objs)

            boxes = np.zeros((num_objs, 4), dtype=np.uint16)
            gt_classes = np.zeros((num_objs), dtype=np.int32)
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

            # Load object boundinng boxes into a data frame.
            for ix, obj in enumerate(objs):
                # TODO : Make pixel indexes 0-based ?
                x1 = float(obj[1]) - 1
                y1 = float(obj[2]) - 1
                x2 = float(obj[3]) - 1
                y2 = float(obj[4]) - 1
                cls = self._class_to_ind[obj[5].lower().strip()] if not self._integrate_classes else self._class_to_ind["vehicle"]
                boxes[ix, :] = [x1, y1, x2, y2]
                gt_classes[ix] = cls
                overlaps[ix, cls] = 1.0

            overlaps = scipy.sparse.csr_matrix(overlaps)

            gt_roidb.append( {"boxes": boxes,
                    "gt_classes": gt_classes,
                    'flipped': False,
                    "gt_overlaps": overlaps})


        with open(cache_file, "wb") as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print("Wrote gt roidb to {}".format(cache_file))

        return gt_roidb

    def _load_nexet_annotation(self, index):
        """
        Load image and bounding boxes info from csv file in the nexet format.
        :param index:
        :return:
        """
        filename = os.path.join(self._data_path, "train_boxes.csv")
        with open(filename) as f:
            objs = [x.split(",") for x in f.readlines()[1:] if x.split(",")[0].strip().__eq__(index)]
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object boundinng boxes into a data frame.
        for ix, obj in enumerate(objs):
            # TODO : Make pixel indexes 0-based ?
            x1 = float(obj[1]) - 1
            y1 = float(obj[2]) - 1
            x2 = float(obj[3]) - 1
            y2 = float(obj[4]) - 1
            cls = self._class_to_ind[obj[5].lower().strip()] if not self._integrate_classes else self._class_to_ind["vehicle"]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {"boxes": boxes,
                "gt_classes": gt_classes,
                'flipped': False,
                "gt_overlaps": overlaps}

    def _get_nexet_results_file_template(self):
        filename = "dt.csv"
        path = os.path.join(self._data_path, filename)
        return path

    def _write_nexet_result_file(self, all_boxes):
        filename = self._get_nexet_results_file_template()
        # classify each 5 vehicle types
        if not self._integrate_classes:
            with open(filename, 'wt') as f:
                f.write("image_filename,x0,y0,x1,y1,label,confidence\n")
                for cls_ind, cls in enumerate(self.classes):
                    if cls == "__background__":
                        continue
                    print("Writing nexet results file ({:s})".format(cls))
                    for im_ind, index in enumerate(self.image_index):
                        dets = all_boxes[cls_ind][im_ind]
                        if dets == []:
                            continue
                        # TODO : the VOCdevkit expects 1-based indices ?
                        for k in range(dets.shape[0]):
                            f.write('{:s},{:.1f},{:.1f},{:.1f},{:.1f},{:s},{:.3f}\n'.
                                    format(index,
                                           dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1,
                                           cls,
                                           dets[k, -1]))
        # Only label VEHICLE
        else:
            with open(filename, 'wt') as f:
                f.write("image_filename,x0,y0,x1,y1,label,confidence\n")
                for cls_ind, cls in enumerate(self.classes):
                    if cls == "__background__":
                        continue
                    print("Writing nexet results file")
                    for im_ind, index in enumerate(self.image_index):
                        dets = all_boxes[cls_ind][im_ind]
                        if dets == []:
                            continue
                        # TODO : the VOCdevkit expects 1-based indices ?
                        for k in range(dets.shape[0]):
                            f.write('{:s},{:.1f},{:.1f},{:.1f},{:.1f},{:s},{:.3f}\n'.
                                    format(index,
                                           dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1,
                                           "vehicle",
                                           dets[k, -1]))

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_nexet_result_file(all_boxes)
        self._do_python_eval(output_dir)
        # if self.config['matlab_eval']:
        #     self._do_matlab_eval(output_dir)
        """
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)
        """

    def _do_python_eval(self, output_dir):
        annopath = os.path.join(self._data_path, "train_boxes.csv")
        resultpath = self._get_nexet_results_file_template()
        result_mAP = datasets.nexet_challenge_eval.evaluation(annopath, resultpath)
        print('mAP',result_mAP)


if __name__ == '__main__':
  from datasets.nexet import nexet

  d = nexet('train', True)
  res = d.roidb
  from IPython import embed;

  embed()
