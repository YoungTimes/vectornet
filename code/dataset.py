from re import S
from numpy.lib.function_base import append
import pandas as pd
import pickle
import sys
import os
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import cascaded_union

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap

from argoverse.utils.centerline_utils import (
    remove_overlapping_lane_seq,
)
from argoverse.utils.mpl_plotting_utils import visualize_centerline

import multiprocessing

import random
import pandas as pd
import math
import datetime

from shapely.geometry import Point, Polygon, LineString, LinearRing
from shapely.affinity import affine_transform, rotate

import shutil

import argparse

class ArgoverseData(object):
    def __init__(self, args):
        super().__init__()

        self.num_batchs = 0
        self.batch_pointer = 0
        self.feature_file_pointer = 0
        self.feature_file_list = []

        self.features = []
        self.labels = []
        self.masks = []
        self.graph_masks = []
        self.params = []
        self.seq_ids = []

        self.pids = []

        self.node_masks = []

        self.origin_features = []
        self.origin_input_trajs = []

        self.args = args

        self.feature_data_dir = self.args.data_dir + "features/"

        self.max_elems_in_sub_graph = 100
        self.max_features_in_elems = 100
        self.max_feature_dim = 9

        self.feature_small_size = 3000

        self.sequeues = []

    def next_batch(self):
        x_batch = []
        y_batch = []
        mask_batch = []
        graph_mask_batch = []
        param_batch = []
        origin_feature_batch = []
        origin_input_traj_batch = []

        pid_batch = []
        node_mask_batch = []

        # print(self.masks.shape)

        i = 0
        while i < self.args.batch_size:
            if self.batch_pointer >= len(self.features):
                self.batch_pointer = 0
                self.feature_file_pointer = self.feature_file_pointer + 1
                if self.feature_file_pointer >= len(self.feature_file_list):
                    self.feature_file_pointer = 0
                    self.load_feature_file(self.feature_file_list[self.feature_file_pointer])
                    

            # print("pointer:" + str(self.batch_pointer) + ", feature len:" + str(len(self.features)))

            # print("===============")
            # print(i)
            # print(self.batch_pointer)

            x_batch.append(self.features[self.batch_pointer])
            y_batch.append(self.labels[self.batch_pointer])
            mask_batch.append(self.masks[self.batch_pointer])
            graph_mask_batch.append(self.graph_masks[self.batch_pointer])
            param_batch.append(self.params[self.batch_pointer])
            origin_feature_batch.append(self.origin_features[self.batch_pointer])
            origin_input_traj_batch.append(self.origin_input_trajs[self.batch_pointer])
            pid_batch.append(self.pids[self.batch_pointer])
            node_mask_batch.append(self.node_masks[self.batch_pointer])

            i = i + 1
            self.batch_pointer = self.batch_pointer + 1

        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        mask_batch = np.array(mask_batch).astype(bool)
        graph_mask_batch = np.array(graph_mask_batch).astype(bool)
        param_batch = np.array(param_batch)
        origin_feature_batch = np.array(origin_feature_batch)
        origin_input_traj_batch = np.array(origin_input_traj_batch)

        pid_batch = np.array(pid_batch)
        node_mask_batch = np.array(node_mask_batch)

        # print(origin_feature_batch)
        # print(self.origin_features.shape)

        random_indices = np.arange(self.args.batch_size)
        random.shuffle(random_indices)

        # print(random_indices)
        # print(x_batch)

        # sys.exit(0)

        # print(mask_batch.shape)
        # print(mask_batch)

        x_batch = x_batch[random_indices, :]
        y_batch = y_batch[random_indices, :]
        mask_batch = mask_batch[random_indices, :]
        graph_mask_batch = graph_mask_batch[random_indices, :]
        param_batch = param_batch[random_indices, :]
        origin_feature_batch = origin_feature_batch[random_indices]
        origin_input_traj_batch = origin_input_traj_batch[random_indices]

        pid_batch = pid_batch[random_indices, :]
        node_mask_batch = node_mask_batch[random_indices, :]

        return x_batch, y_batch, mask_batch, graph_mask_batch, param_batch, origin_input_traj_batch, pid_batch, node_mask_batch, origin_feature_batch

    def load_feature_file(self, pickle_filename):
        with open(pickle_filename, 'rb') as fp:
            data = pickle.load(fp)
            self.features = data['features']
            self.labels = data['labels']
            self.masks = data['masks']
            self.graph_masks = data['graph_masks']
            self.params = data['params']
            self.origin_features = data["origin_features"]
            self.origin_input_trajs = data["origin_input_trajs"]
            self.pids = data["pids"]
            self.node_masks = data["node_masks"]

    def load_data(self):

        if (not os.path.exists(self.feature_data_dir)) or len(os.listdir(self.feature_data_dir)) <= 0:
            print("empty trainning data")
            return

        pickle_filename = 

            # self.load_data_from_origin()

        
            # with open(self.pickle_filename, 'rb') as fp:
            #     data = pickle.load(fp)
            #     self.features = data['features']
            #     self.labels = data['labels']
            #     self.masks = data['masks']
            #     self.graph_masks = data['graph_masks']
            #     self.params = data['params']
            #     self.origin_features = data["origin_features"]
            #     self.origin_input_trajs = data["origin_input_trajs"]
            #     self.pids = data["pids"]
            #     self.node_masks = data["node_masks"]

            #     print("features len:" + str(len(self.features)) + ", batch size:" + str(self.args.batch_size))

        self.num_batchs = (int)(len(self.features) / self.args.batch_size) + 1

        return

