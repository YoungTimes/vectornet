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

import scipy.interpolate as interp

# class ArgoverseData(object):
#     def __init__(self, args):
#         super().__init__()

#         self.num_batchs = 0
#         self.batch_pointer = 0
#         self.feature_file_pointer = 0
#         self.feature_file_list = []
#         # self.batch_size = 5

#         self.features = []
#         self.labels = []
#         self.masks = []
#         self.graph_masks = []
#         self.params = []
#         self.seq_ids = []

#         self.pids = []

#         self.node_masks = []

#         self.origin_features = []
#         self.origin_input_trajs = []

#         self.args = args

#         # self.pickle_filename = args.data_dir +  "train_dataset.pkl"

#         self.feature_data_dir = self.args.data_dir + "features/"

#         self.max_elems_in_sub_graph = 100
#         self.max_features_in_elems = 100
#         self.max_feature_dim = 9

#         self.feature_small_size = 3000

#     def next_batch(self):
#         x_batch = []
#         y_batch = []
#         mask_batch = []
#         graph_mask_batch = []
#         param_batch = []
#         origin_feature_batch = []
#         origin_input_traj_batch = []

#         pid_batch = []
#         node_mask_batch = []

#         # print(self.masks.shape)

#         i = 0
#         while i < self.args.batch_size:
#             if self.batch_pointer >= len(self.features):
#                 self.batch_pointer = 0
#                 self.feature_file_pointer = self.feature_file_pointer + 1
#                 if self.feature_file_pointer >= len(self.feature_file_list):
#                     self.feature_file_pointer = 0
#                     self.load_feature_file(self.feature_file_list[self.feature_file_pointer])
                    

#             # print("pointer:" + str(self.batch_pointer) + ", feature len:" + str(len(self.features)))

#             # print("===============")
#             # print(i)
#             # print(self.batch_pointer)

#             x_batch.append(self.features[self.batch_pointer])
#             y_batch.append(self.labels[self.batch_pointer])
#             mask_batch.append(self.masks[self.batch_pointer])
#             graph_mask_batch.append(self.graph_masks[self.batch_pointer])
#             param_batch.append(self.params[self.batch_pointer])
#             origin_feature_batch.append(self.origin_features[self.batch_pointer])
#             origin_input_traj_batch.append(self.origin_input_trajs[self.batch_pointer])
#             pid_batch.append(self.pids[self.batch_pointer])
#             node_mask_batch.append(self.node_masks[self.batch_pointer])

#             i = i + 1
#             self.batch_pointer = self.batch_pointer + 1

#         x_batch = np.array(x_batch)
#         y_batch = np.array(y_batch)
#         mask_batch = np.array(mask_batch).astype(bool)
#         graph_mask_batch = np.array(graph_mask_batch).astype(bool)
#         param_batch = np.array(param_batch)
#         origin_feature_batch = np.array(origin_feature_batch)
#         origin_input_traj_batch = np.array(origin_input_traj_batch)

#         pid_batch = np.array(pid_batch)
#         node_mask_batch = np.array(node_mask_batch)

#         # print(origin_feature_batch)
#         # print(self.origin_features.shape)

#         random_indices = np.arange(self.args.batch_size)
#         random.shuffle(random_indices)

#         # print(random_indices)
#         # print(x_batch)

#         # sys.exit(0)

#         # print(mask_batch.shape)
#         # print(mask_batch)

#         x_batch = x_batch[random_indices, :]
#         y_batch = y_batch[random_indices, :]
#         mask_batch = mask_batch[random_indices, :]
#         graph_mask_batch = graph_mask_batch[random_indices, :]
#         param_batch = param_batch[random_indices, :]
#         origin_feature_batch = origin_feature_batch[random_indices]
#         origin_input_traj_batch = origin_input_traj_batch[random_indices]

#         pid_batch = pid_batch[random_indices, :]
#         node_mask_batch = node_mask_batch[random_indices, :]

#         return x_batch, y_batch, mask_batch, graph_mask_batch, param_batch, origin_input_traj_batch, pid_batch, node_mask_batch, origin_feature_batch

#     def load_feature_file(self, pickle_filename):
#         with open(pickle_filename, 'rb') as fp:
#             data = pickle.load(fp)
#             self.features = data['features']
#             self.labels = data['labels']
#             self.masks = data['masks']
#             self.graph_masks = data['graph_masks']
#             self.params = data['params']
#             self.origin_features = data["origin_features"]
#             self.origin_input_trajs = data["origin_input_trajs"]
#             self.pids = data["pids"]
#             self.node_masks = data["node_masks"]

#     def load_data(self, force_reproduce = True):

#         if (not os.path.exists(self.feature_data_dir)) or len(os.listdir(self.feature_data_dir)) <= 0 or force_reproduce == True:
#             print("reproduce data...")
#             self.load_data_from_origin()
#             # with open(self.pickle_filename, 'rb') as fp:
#             #     data = pickle.load(fp)
#             #     self.features = data['features']
#             #     self.labels = data['labels']
#             #     self.masks = data['masks']
#             #     self.graph_masks = data['graph_masks']
#             #     self.params = data['params']
#             #     self.origin_features = data["origin_features"]
#             #     self.origin_input_trajs = data["origin_input_trajs"]
#             #     self.pids = data["pids"]
#             #     self.node_masks = data["node_masks"]

#             #     print("features len:" + str(len(self.features)) + ", batch size:" + str(self.args.batch_size))

#         self.num_batchs = (int)(len(self.features) / self.args.batch_size) + 1

#         return


def interpolate_polyline_n(polyline, num_points):
    tck, u = interp.splprep(polyline.T, s=0)
    u = np.linspace(0.0, 1.0, num_points)

    return np.column_stack(interp.splev(u, tck))

def line_length(line):
    return np.linalg.norm(np.diff(line, axis=0))

def interpolate_polyline(polyline):
    # print("======================")
    # print(polyline)

    duplicates = []
    for i in range(1, len(polyline)):
        if np.allclose(polyline[i], polyline[i - 1]):
            duplicates.append(i)
    if len(polyline) - len(duplicates) < 4:
        return polyline
    if duplicates:
        polyline = np.delete(polyline, duplicates, axis=0)

    while len(polyline) < 4:
        new_polyline = [polyline[0]]

        for i in range(1, len(polyline)):
            new_polyline.append((polyline[i - 1] + polyline[i]) / 2.0)
            new_polyline.append(polyline[i])

        polyline = new_polyline

    # print("length:" + str(length))
    # print("polyline len:" + str(len(polyline)))

    length  = line_length(polyline)
    resolution = 0.2
    num_points = int(length / resolution) + 1

    num_points = max(num_points, 2)
    # print(num_points)
    return interpolate_polyline_n(polyline, num_points)

def create_bound_features(args, bound_line, type, lane_id, city_name, avm, sub_graph, sub_graph_mask, sub_pids):
    lane_features = []
    has_traffic_contrl = avm.lane_has_traffic_control_measure(lane_id, city_name)
    turn = avm.get_lane_turn_direction(lane_id, city_name)
    in_intersection = avm.lane_is_in_intersection(lane_id, city_name)
    # origin_center_line = avm.get_lane_segment_centerline(lane_id, city_name)

    for cl_indx in range(1, len(bound_line)):
        lane_feature = []

        cl_0 = bound_line[cl_indx - 1]
        cl_1 = bound_line[cl_indx]

        # offset to the last observed point
        lane_feature.extend([cl_0[0], cl_0[1]])
        lane_feature.extend([cl_1[0], cl_1[1]])

        # object type: bound line
        if type == "left":
            lane_feature.append(3)
        else:
            lane_feature.append(4)

        # lane_feature.append(0)
        # lane_feature.append(0)
        # lane_feature.append(0)

        if has_traffic_contrl == True:
            lane_feature.append(1)
        else:
            lane_feature.append(-1)

        if turn == 'LEFT':
            lane_feature.append(1)
        elif turn == 'RIGHT':
            lane_feature.append(-1)
        else:
            lane_feature.append(0)

        if in_intersection == True:
            lane_feature.append(1)
        else:
            lane_feature.append(-1)

        lane_feature.append(lane_id)
        # lane_feature.append(0)

        lane_features.append(lane_feature)

    if len(lane_features) <= 0:
        print("lane_features no data")
        return

    # print("lane_features:" + str(len(lane_features)) + ", bound line size:" + str(len(bound_line)))
    sub_pids.append([min(np.array(lane_features)[:, 0]), min(np.array(lane_features)[:, 1])])

    lane_features_mask = [True] * len(lane_features)

    if len(lane_features) < args.max_features_in_elems:
        pad_mask = [False] * (args.max_features_in_elems - len(lane_features))
        lane_features_mask.extend(pad_mask)

        pad_feature = [[0] * args.max_feature_dim] * (args.max_features_in_elems - len(lane_features))
        lane_features.extend(pad_feature)
    else:
        lane_features = lane_features[:args.max_features_in_elems]
        lane_features_mask = lane_features_mask[:args.max_features_in_elems]

    sub_graph.append(lane_features)
    sub_graph_mask.append(lane_features_mask)

def get_point_in_polygon_score(lane_seq: List[int],
                                xy_seq: np.ndarray, city_name: str,
                                avm: ArgoverseMap) -> int:
    lane_seq_polygon = cascaded_union([
        Polygon(avm.get_lane_segment_polygon(lane, city_name)).buffer(0)
        for lane in lane_seq
    ])
    point_in_polygon_score = 0
    for xy in xy_seq:
        point_in_polygon_score += lane_seq_polygon.contains(Point(xy))
    return point_in_polygon_score

def sort_lanes_based_on_point_in_polygon_score(
        lane_seqs: List[List[int]],
        xy_seq: np.ndarray,
        city_name: str,
        avm: ArgoverseMap,
) -> List[List[int]]:
    point_in_polygon_scores = []
    for lane_seq in lane_seqs:
        point_in_polygon_scores.append(get_point_in_polygon_score(lane_seq, xy_seq, city_name, avm))
    randomized_tiebreaker = np.random.random(len(point_in_polygon_scores))
    sorted_point_in_polygon_scores_idx = np.lexsort(
        (randomized_tiebreaker, np.array(point_in_polygon_scores)))[::-1]
    sorted_lane_seqs = [
        lane_seqs[i] for i in sorted_point_in_polygon_scores_idx
    ]
    sorted_scores = [point_in_polygon_scores[i]
        for i in sorted_point_in_polygon_scores_idx]
    return sorted_lane_seqs, sorted_scores

def get_candidate_centerlines_for_trajectory(
            xy: np.ndarray,
            city_name: str,
            avm: ArgoverseMap,
            viz: bool = False,
            max_search_radius: float = 50.0,
            seq_len: int = 50
    ) -> List[np.ndarray]:

    # Map Feature computations
    _MANHATTAN_THRESHOLD = 5.0  # meters
    _DFS_THRESHOLD_FRONT_SCALE = 20.0  # meters
    _DFS_THRESHOLD_BACK_SCALE = 15.0  # meters
    _MAX_SEARCH_RADIUS_CENTERLINES = 50.0  # meters

    # Get all lane candidates within a bubble
    curr_lane_candidates = avm.get_lane_ids_in_xy_bbox(xy[-1, 0], xy[-1, 1], city_name, _MANHATTAN_THRESHOLD)

    # Keep expanding the bubble until at least 1 lane is found
    while (len(curr_lane_candidates) < 1 and _MANHATTAN_THRESHOLD < max_search_radius):
        _MANHATTAN_THRESHOLD *= 2
        curr_lane_candidates = avm.get_lane_ids_in_xy_bbox(xy[-1, 0], xy[-1, 1], city_name, _MANHATTAN_THRESHOLD)

    assert len(curr_lane_candidates) > 0, "No nearby lanes found!!"

    # Set dfs threshold
    traj_len = xy.shape[0]

    # Assuming a speed of 50 mps, set threshold for traversing in the front and back
    dfs_threshold_front = _DFS_THRESHOLD_FRONT_SCALE * (seq_len + 1 - traj_len) / 10
    dfs_threshold_back = _DFS_THRESHOLD_BACK_SCALE * (traj_len + 1) / 10

    # DFS to get all successor and predecessor candidates
    obs_pred_lanes: List[Sequence[int]] = []
    for lane in curr_lane_candidates:
        candidates_future = avm.dfs(lane, city_name, 0, dfs_threshold_front)
        candidates_past = avm.dfs(lane, city_name, 0, dfs_threshold_back, True)

        # Merge past and future
        for past_lane_seq in candidates_past:
            for future_lane_seq in candidates_future:
                assert (past_lane_seq[-1] == future_lane_seq[0]), "Incorrect DFS for candidate lanes past and future"
                obs_pred_lanes.append(past_lane_seq + future_lane_seq[1:])

    # Removing overlapping lanes
    obs_pred_lanes = remove_overlapping_lane_seq(obs_pred_lanes)

    # Sort lanes based on point in polygon score
    obs_pred_lanes, scores = sort_lanes_based_on_point_in_polygon_score(
                                    obs_pred_lanes, xy, city_name, avm)

    # print("candidate lane len:" + str(len(obs_pred_lanes)))
    # print(obs_pred_lanes[0])

    # if viz:
    #     avm.draw_lane(obs_pred_lanes[0], city_name, legend=False)

    # return obs_pred_lanes[0]

    # # If the best centerline is not along the direction of travel, re-sort
    # if mode == "test":
    #     candidate_centerlines = self.get_heuristic_centerlines_for_test_set(
    #         obs_pred_lanes, xy, city_name, avm, max_candidates, scores)
    # else:

    if viz:
        candidate_centerlines = avm.get_cl_from_lane_seq(obs_pred_lanes, city_name)

        plt.figure(0, figsize=(8, 7))
        for centerline_coords in candidate_centerlines:
            visualize_centerline(centerline_coords)
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "-",
            color="#d33e4c",
            alpha=1,
            linewidth=3,
            zorder=15,
        )

        final_x = xy[-1, 0]
        final_y = xy[-1, 1]

        plt.plot(
            final_x,
            final_y,
            "o",
            color="#d33e4c",
            alpha=1,
            markersize=10,
            zorder=15,
        )
        plt.xlabel("Map X")
        plt.ylabel("Map Y")
        plt.axis("off")
        plt.title(f"Number of candidates = {len(candidate_centerlines)}")
        plt.show()

    return obs_pred_lanes



def normalized_map(xy_seq, base_pt, angle):
    m = [1, 0, 0, 1, -base_pt[0], -base_pt[1]]
    ls = LineString(xy_seq)

    # Now apply rotation, taking care of edge cases
    ls_offset = affine_transform(ls, m)

    # Rotate the trajetory
    ls_rotate = rotate(ls_offset, angle, origin=(0, 0)).coords[:]

    # Normalized trajectory
    norm_xy = np.array(ls_rotate)

    return norm_xy

def swap_left_and_right(condition, left_centerline, right_centerline):
    """
    Swap points in left and right centerline according to condition.
    Args:
    condition: Numpy array of shape (N,) of type boolean. Where true, swap the values in the left and
                right centerlines.
    left_centerline: The left centerline, whose points should be swapped with the right centerline.
    right_centerline: The right centerline.
    Returns:
    left_centerline
    right_centerline
    """

    right_swap_indices = right_centerline[condition]
    left_swap_indices = left_centerline[condition]

    left_centerline[condition] = right_swap_indices
    right_centerline[condition] = left_swap_indices

    return left_centerline, right_centerline


def centerline_to_polygon(centerline, width_scaling_factor = 1.0, visualize = False):
    """
    Convert a lane centerline polyline into a rough polygon of the lane's area.
    On average, a lane is 3.8 meters in width. Thus, we allow 1.9 m on each side.
    We use this as the length of the hypotenuse of a right triangle, and compute the
    other two legs to find the scaled x and y displacement.
    Args:
    centerline: Numpy array of shape (N,2).
    width_scaling_factor: Multiplier that scales 3.8 meters to get the lane width.
    visualize: Save a figure showing the the output polygon.
    Returns:
    polygon: Numpy array of shape (2N+1,2), with duplicate first and last vertices.
    """
    # eliminate duplicates
    _, inds = np.unique(centerline, axis=0, return_index=True)
    # does not return indices in sorted order
    inds = np.sort(inds)
    centerline = centerline[inds]

    dx = np.gradient(centerline[:, 0])
    dy = np.gradient(centerline[:, 1])

    # compute the normal at each point
    slopes = dy / dx
    inv_slopes = -1.0 / slopes

    thetas = np.arctan(inv_slopes)
    x_disp = 3.6 * width_scaling_factor / 2.0 * np.cos(thetas)
    y_disp = 3.6 * width_scaling_factor / 2.0 * np.sin(thetas)

    displacement = np.hstack([x_disp[:, np.newaxis], y_disp[:, np.newaxis]])
    right_centerline = centerline + displacement
    left_centerline = centerline - displacement

    # right centerline position depends on sign of dx and dy
    subtract_cond1 = np.logical_and(dx > 0, dy < 0)
    subtract_cond2 = np.logical_and(dx > 0, dy > 0)
    subtract_cond = np.logical_or(subtract_cond1, subtract_cond2)
    left_centerline, right_centerline = swap_left_and_right(subtract_cond, left_centerline, right_centerline)

    # right centerline also depended on if we added or subtracted y
    neg_disp_cond = displacement[:, 1] > 0
    left_centerline, right_centerline = swap_left_and_right(neg_disp_cond, left_centerline, right_centerline)

    if visualize:
        plt.scatter(centerline[:, 0], centerline[:, 1], 20, marker=".", color="b")
        plt.scatter(right_centerline[:, 0], right_centerline[:, 1], 20, marker=".", color="r")
        plt.scatter(left_centerline[:, 0], left_centerline[:, 1], 20, marker=".", color="g")
        fname = datetime.datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S_%f")
        plt.savefig(f"polygon_unit_tests/{fname}.png")
        plt.close("all")

    # return the polygon
    return right_centerline, left_centerline

def normalized_traj(args, xy_seq, base_pt):

    # First apply translation
    m = [1, 0, 0, 1, -base_pt[0], -base_pt[1]]
    ls = LineString(xy_seq)

    # Now apply rotation, taking care of edge cases
    ls_offset = affine_transform(ls, m)
    end = ls_offset.coords[args.obs_len]
    if end[0] == 0 and end[1] == 0:
        angle = 0.0
    elif end[0] == 0:
        angle = -90.0 if end[1] > 0 else 90.0
    elif end[1] == 0:
        angle = 0.0 if end[0] > 0 else 180.0
    else:
        angle = math.degrees(math.atan(end[1] / end[0]))
        if (end[0] > 0 and end[1] > 0) or (end[0] > 0 and end[1] < 0):
            angle = -angle
        else:
            angle = 180.0 - angle

    # Rotate the trajetory
    ls_rotate = rotate(ls_offset, angle, origin=(0, 0)).coords[:]

    # Normalized trajectory
    norm_xy = np.array(ls_rotate)

    return norm_xy, angle

def load_data_from_origin(args, sequences, start_seq, end_seq, output_filename):

    one_features = []
    one_labels = []
    one_masks = []
    one_graph_masks = []
    one_params = []
    one_pids = []
    one_node_masks = []
    one_origin_features = []
    one_origin_input_trajs = []

    train_data_dir = args.data_dir + "train/data/"

    avm = ArgoverseMap()

    # sequences = os.listdir(train_data_dir)
    # num_sequences = self.feature_small_size # len(sequences)
    # print("start_seq:"+ str(start_seq) + ", end_seq:" + str(end_seq))

    for i in range(start_seq, end_seq):
        print("process {}/{}-{}".format(i, start_seq, end_seq))
        # feq_cnt = feq_cnt + 1
        # if feq_cnt >= self.feature_small_size:
        #     break
        seq = sequences[i]
        if not seq.endswith(".csv"):
            continue

        seq_path = f"{train_data_dir}/{seq}"
        seq_id = int(seq.split(".")[0])

        seq_df = pd.read_csv(seq_path)

        # print(argoverse_forecasting_data.seq_df[argoverse_forecasting_data.seq_df["OBJECT_TYPE"] == "AGENT"])

        agent_x = seq_df[seq_df["OBJECT_TYPE"] == "AGENT"]["X"]
        agent_y = seq_df[seq_df["OBJECT_TYPE"] == "AGENT"]["Y"]
        agent_traj = np.column_stack((agent_x, agent_y))
        agent_timestamp = seq_df[seq_df["OBJECT_TYPE"] == "AGENT"]["TIMESTAMP"]
        # agent_track_id = argoverse_forecasting_data.seq_df[argoverse_forecasting_data.seq_df["OBJECT_TYPE"] == "AGENT"]["TRACK_ID"]

        # print(argoverse_forecasting_data.agent_traj)
        # agent_traj = argoverse_forecasting_data.agent_traj
        city_name = seq_df["CITY_NAME"].values[0]

        # base_x = agent_traj[self.obs_len - 1][0]
        # base_y = agent_traj[self.obs_len - 1][1]

        base_pt = agent_traj[args.obs_len - 1]

        offset_agent_traj, angle = normalized_traj(args, agent_traj, base_pt) # agent_traj - base_pt

        # print(offset_agent_traj)

        # candidate_centerlines = avm.get_candidate_centerlines_for_traj(agent_traj, city_name, viz=False, max_search_radius=50.0)
        # print(candidate_centerlines[0])

        sub_graph = []
        sub_graph_mask = []

        # min_x = float('inf')
        # max_x = float('-inf')

        origin_center_lines = []

        # min_y = float('inf')
        # max_y = float('-inf')

        candidate_lane_seqs = []
        # candidate_lane_seqs = avm.get_lane_ids_in_xy_bbox(base_pt[0], base_pt[1], city_name, query_search_range_manhattan=20.0)
        # obs_pred_lanes, scores = self.sort_lanes_based_on_point_in_polygon_score(
        #                             [candidate_lane_seqs], agent_traj, city_name, avm)

        sub_pids = []

        # for lane_id in candidate_lane_seqs:
        candidate_lane_seqs = get_candidate_centerlines_for_trajectory(agent_traj[:args.obs_len], city_name, avm, viz=False)
        dup_check = set()
        for lane_seq in candidate_lane_seqs:
            for lane_id in lane_seq:
                if lane_id in dup_check:
                    continue

                dup_check.add(lane_id)

                # has_traffic_contrl = avm.lane_has_traffic_control_measure(lane_id, city_name)
                # turn = avm.get_lane_turn_direction(lane_id, city_name)
                # in_intersection = avm.lane_is_in_intersection(lane_id, city_name)
                origin_center_line = avm.get_lane_segment_centerline(lane_id, city_name)
                

                centerline_2d = np.delete(origin_center_line, -1, axis=-1)
                    
                origin_center_lines.append(origin_center_line)
                origin_center_lines.append(origin_center_line)
                    
                origin_center_line = interpolate_polyline(centerline_2d)

                # center_line = self.normalized_map(origin_center_line, base_pt, angle)

                # lane_features = []
                # for cl_indx in range(1, len(center_line)):
                #    lane_feature = []

                #    cl_0 = center_line[cl_indx - 1]
                #    cl_1 = center_line[cl_indx]

                #    # offset to the last observed point
                #    lane_feature.extend([cl_0[0], cl_0[1]])
                #    lane_feature.extend([cl_1[0], cl_1[1]])

                #    # object type: lane
                #    lane_feature.append(1)

                #    if has_traffic_contrl == True:
                #        lane_feature.append(1)
                #    else:
                #        lane_feature.append(-1)

                #   if turn == 'LEFT':
                #        lane_feature.append(1)
                #    elif turn == 'RIGHT':
                #        lane_feature.append(-1)
                #    else:
                #        lane_feature.append(0)

                #   if in_intersection == True:
                #        lane_feature.append(1)
                #    else:
                #        lane_feature.append(-1)

                #    lane_feature.append(lane_id)

                #    lane_features.append(lane_feature)

                # sub_pids.append([min(np.array(lane_features)[:, 0]), min(np.array(lane_features)[:, 1])])

                #if len(lane_features) <= 0:
                #    print("lane_features no data")
                #    continue

                # lane_features_mask = [True] * len(lane_features)

                #if len(lane_features) < self.max_features_in_elems:
                #    pad_mask = [False] * (self.max_features_in_elems - len(lane_features))
                #    lane_features_mask.extend(pad_mask)

                #    pad_feature = [[0] * self.max_feature_dim] * (self.max_features_in_elems - len(lane_features))
                #    lane_features.extend(pad_feature)
                #else:
                #    lane_features = lane_features[:self.max_features_in_elems]
                #    lane_features_mask = lane_features_mask[:self.max_features_in_elems]

                #sub_graph.append(lane_features)
                #sub_graph_mask.append(lane_features_mask)

                origin_left_bound, origin_right_bound = centerline_to_polygon(centerline_2d)

                origin_left_bound = interpolate_polyline(origin_left_bound)
                origin_right_bound = interpolate_polyline(origin_right_bound)
                left_bound = normalized_map(origin_left_bound, base_pt, angle)
                right_bound = normalized_map(origin_right_bound, base_pt, angle)
                create_bound_features(args, left_bound, "left", lane_id, city_name, avm, sub_graph, sub_graph_mask, sub_pids)
                create_bound_features(args, right_bound, "right", lane_id, city_name, avm, sub_graph, sub_graph_mask, sub_pids)

        # add agent traj features
        origin_center_lines.append(agent_traj[:args.obs_len])
        agent_features = []
        # obs_traj
        for agent_idx in range(1, args.obs_len):
            agent_feature = []
            agent_0 = offset_agent_traj[agent_idx - 1]
            agent_1 = offset_agent_traj[agent_idx]

            agent_feature.extend([agent_0[0], agent_0[1]])
            agent_feature.extend([agent_1[0], agent_1[1]])

            # object type: agent_traj
            agent_feature.append(2)

            # timestamp
            diff_timestamp = agent_timestamp.iloc[args.obs_len - 1] - agent_timestamp.iloc[agent_idx - 1]
            agent_feature.append(diff_timestamp)

            diff_timestamp = agent_timestamp.iloc[args.obs_len - 1] - agent_timestamp.iloc[agent_idx]
            agent_feature.append(diff_timestamp)

            # distance
            dis = np.linalg.norm(agent_1 - agent_0)
            agent_feature.append(dis)

            # heading
            # unit_vec = (agent_1 - agent_0)
            # agent_feature.append(unit_vec[0])
            # agent_feature.append(unit_vec[1])

            # id
            agent_feature.append(1111)

            agent_features.append(agent_feature)

        sub_pids.append([min(np.array(agent_features)[:, 0]), min(np.array(agent_features)[:, 1])])

        agent_features_mask = [True] * len(agent_features)

        if len(agent_features) < args.max_features_in_elems:
            pad_mask = [False] * (args.max_features_in_elems - len(agent_features))
            agent_features_mask.extend(pad_mask)

            pad_feature = [[0] * args.max_feature_dim] * (args.max_features_in_elems - len(agent_features))
            agent_features.extend(pad_feature)
        else:
            agent_features_mask = agent_features_mask[:args.max_features_in_elems]
            agent_features = agent_features[:args.max_features_in_elems]

        sub_graph.append(agent_features)
        sub_graph_mask.append(agent_features_mask)

        graph_mask = [True] * len(sub_graph)

        if len(sub_graph) < args.max_elems_in_sub_graph:
            pad_mask = [[False] * args.max_features_in_elems]  * (args.max_elems_in_sub_graph - len(sub_graph))
            sub_graph_mask.extend(pad_mask)
            # sub_graph.append(sub_graph[-1] * (max_elems_in_sub_graph - len(sub_graph)))
            # print("diff" + str(max_elems_in_sub_graph - len(sub_graph)))
            # print(sub_graph[-1])
            # print([[[0, 0]] * (self.max_elems_in_sub_graph - len(sub_graph))])
            sub_pids.extend([[0, 0]] * (args.max_elems_in_sub_graph - len(sub_graph)))
            origin_center_lines.extend([[0]] * (args.max_elems_in_sub_graph - len(sub_graph)))
            # print([0] * (self.max_elems_in_sub_graph - len(graph_mask)))
            graph_mask.extend([False] * (args.max_elems_in_sub_graph - len(sub_graph)))
            pad_feature = [[[0] * args.max_feature_dim] * args.max_features_in_elems]  * (args.max_elems_in_sub_graph - len(sub_graph))
            # print(np.array(sub_graph).shape)
            # print(np.array(pad_feature).shape)
            sub_graph.extend(pad_feature)
        else:
            sub_graph_mask = sub_graph_mask[:args.max_elems_in_sub_graph]

            graph_mask = graph_mask[:args.max_elems_in_sub_graph]

            sub_graph = sub_graph[:args.max_elems_in_sub_graph]

            sub_pids = sub_pids[:args.max_elems_in_sub_graph]

            origin_center_lines = origin_center_lines[:args.max_elems_in_sub_graph]

            # print("grpah mask len:" + str(len(graph_mask)))

            # print("sub pids:" + str(np.array(sub_pids).shape))

        agent_label = offset_agent_traj[(args.obs_len - 1) :]

        sub_graph = np.array(sub_graph)

        node_mask = np.array(graph_mask)

        lane_idxs = np.argwhere(node_mask == True)

        # print(lane_idxs)
        random_cnt = 5 # (int)(lane_idxs.shape[0] * 0.3)
        indexs = np.random.choice(np.arange(lane_idxs.shape[0]), size = random_cnt, replace = False)
        lane_random_idxs = lane_idxs[indexs]
        node_mask = np.full(node_mask.shape, False)
        # print(lane_random_idxs)
        node_mask[lane_random_idxs] = True

        min_x = np.min(sub_graph[:, :, [0, 2]])
        max_x = np.max(sub_graph[:, :, [0, 2]])

        min_y = np.min(sub_graph[:, :, [1, 3]])
        max_y = np.max(sub_graph[:, :, [1, 3]])

        min_4 = np.min(sub_graph[:, :, [4]])
        max_4 = np.max(sub_graph[:, :, [4]])

        min_5 = np.min(sub_graph[:, :, [5]])
        max_5 = np.max(sub_graph[:, :, [5]])

        min_6 = np.min(sub_graph[:, :, [6]])
        max_6 = np.max(sub_graph[:, :, [6]])

        min_7 = np.min(sub_graph[:, :, [7]])
        max_7 = np.max(sub_graph[:, :, [7]])

        min_8 = np.min(sub_graph[:, :, [8]])
        max_8 = np.max(sub_graph[:, :, [8]])

        # min_label_x = np.min(offset_agent_traj[:self.obs_len, [0]])
        # max_label_x = np.max(offset_agent_traj[:self.obs_len, [0]])
        # min_label_y = np.min(offset_agent_traj[:self.obs_len, [1]])
        # max_label_y = np.max(offset_agent_traj[:self.obs_len, [1]])

        # scale_label_x = max(abs(min_label_x), abs(max_label_x))
        # scale_label_y = max(abs(min_label_y), abs(max_label_y))

        # sub_graph[:, :, [0, 2]] = sub_graph[:, :, [0, 2]] / scale_label_x
        # sub_graph[:, :, [1, 3]] = sub_graph[:, :, [1, 3]] / scale_label_y

        # min_x = float('inf')
        # max_x = float('-inf')
        # min_y = float('inf')
        # max_y = float('-inf')

        # min_x = min(min_x, min_label_x)
        # max_x = max(max_x, max_label_x)
        # min_y = min(min_y, min_label_y)
        # max_y = max(max_y, max_label_y)

        # min_x = np.min(sub_graph[:, :, [0, 2]])
        # max_x = np.max(sub_graph[:, :, [0, 2]])

        # min_y = np.min(sub_graph[:, :, [1, 3]])
        # max_y = np.max(sub_graph[:, :, [1, 3]])

        # normalize
        x_scale =  max(abs(min_x), abs(max_x))
        y_scale =  max(abs(min_y), abs(max_y))
        scale_4 = max(abs(min_4), abs(max_4))
        scale_5 = max(abs(min_5), abs(max_5))
        scale_6 = max(abs(min_6), abs(max_6))
        scale_7 = max(abs(min_7), abs(max_7))
        scale_8 = max(abs(min_8), abs(max_8))

        x_scale =  max_x - min_x
        y_scale =  max_y - min_y
        scale_4 =  max_4 - min_4
        scale_8 =  max_8 - min_8


        sub_graph[:, :, [0, 2]] = sub_graph[:, :, [0, 2]] / x_scale
        sub_graph[:, :, [1, 3]] = sub_graph[:, :, [1, 3]] / y_scale

        sub_graph[:, :, [4]] = sub_graph[:, :, [4]] / scale_4
        sub_graph[:, :, [5]] = sub_graph[:, :, [5]] / scale_5
        sub_graph[:, :, [6]] = sub_graph[:, :, [6]] / scale_6
        sub_graph[:, :, [7]] = sub_graph[:, :, [7]] / scale_7
        sub_graph[:, :, [8]] = sub_graph[:, :, [8]] / scale_8

        # print("x_scale:" + str(x_scale) + ", y_scale:" + str(y_scale))

        # print("scale_label_x:" + str(scale_label_x) + ", x_scale:" + str(x_scale) + ", scale_label_y:" + str(scale_label_y) + ", y_scale:" + str(y_scale))

        # agent_label[:, [0]] = agent_label[:, [0]]
        # agent_label[:, [1]] = agent_label[:, [1]]

        # print(agent_label)

        # sys.exit(-1)

        # print("before diff:")
        # print(agent_label.shape)
        # print(agent_label)
        # agent_label[:, [0]] = agent_label[:, [0]] / scale_label_x
        # agent_label[:, [1]] = agent_label[:, [1]] / scale_label_y
        agent_label = np.diff(agent_label, axis=0)

        # agent_label_minx = min(agent_label[:, 0])
        # agent_label_maxx = max(agent_label[:, 0])

        # agent_label_miny = min(agent_label[:, 1])
        # agent_label_maxy = max(agent_label[:, 1])

        # agent_label_scale_x = max(abs(agent_label_maxx), abs(agent_label_minx))
        # agent_label_scale_y = max(abs(agent_label_maxy), abs(agent_label_miny))

        # x_scale = 1
        # y_scale = 1

        param = ([seq_id, city_name, base_pt[0], base_pt[1], 1.0 , 1.0, angle])

        one_features.append(sub_graph)
        # offset to the last observed point
        one_labels.append(agent_label)

        one_masks.append(sub_graph_mask)

        one_graph_masks.append(graph_mask)

        one_params.append(param)

        one_pids.append(sub_pids)

        one_node_masks.append(node_mask)

        one_origin_features.append(origin_center_lines)

        one_origin_input_trajs.append(agent_traj[:args.obs_len])


    # self.num_batchs = (int)(len(self.features) / self.args.batch_size)
    one_features = np.array(one_features)

    one_labels = np.array(one_labels).astype('float32')
    one_masks = np.array(one_masks).astype('bool')
    one_graph_masks = np.array(one_graph_masks).astype('bool')
    one_params = np.array(one_params)
    one_pids = np.array(one_pids)
    one_node_masks = np.array(one_node_masks)
    one_origin_features = np.array(one_origin_features)
    # self.origin_input_trajs = np.array(self.origin_input_trajs)

    with open(output_filename, 'wb') as fp:
        pickle.dump({"features" : one_features, "labels" : one_labels,
            "masks" : one_masks, "graph_masks" : one_graph_masks, "params" : one_params,
            "origin_input_trajs" : one_origin_input_trajs, "pids" : one_pids, "node_masks" : one_node_masks,
            "origin_features" : one_origin_features}, fp, protocol = 0)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--obs_len', type = int, default = 20, help = "Observed length of the trajectory")
    parser.add_argument('--pred_len', type = int, default = 30, help = "Prediction length of the trajectory")
    parser.add_argument('--mode', default = 'train', type = str, help = 'train/val/test')
    parser.add_argument('--data_dir', default = "/home/featurize/data/data/", type = str, help = 'training data path')
    parser.add_argument('--batch_size', type = int, default = 32, help = "batch size")
    parser.add_argument('--split_seq_size', type = int, default = 1000, help = "split seq size")
    parser.add_argument('--processor_num', type = int, default = 4, help = "process num")

    parser.add_argument('--max_elems_in_sub_graph', type = int, default = 100, help = "max_elems_in_sub_graph")
    parser.add_argument('--max_features_in_elems', type = int, default = 100, help = "max_features_in_elems")
    parser.add_argument('--max_feature_dim', type = int, default = 9, help = "max_feature_dim")
    
    # root_dir = '../../vectornet/data/forecasting_sample/data/'

    # /home/featurize/data/data/
    args = parser.parse_args()

    train_data_dir = args.data_dir + "train/data/"
    sequences = os.listdir(train_data_dir)

    num_sequences = 20000 # len(sequences) # self.feature_small_size
    total_task_num = int(num_sequences / args.split_seq_size)

    feature_data_dir = args.data_dir + "features/"

    if os.path.exists(feature_data_dir):
        shutil.rmtree(feature_data_dir)

    os.mkdir(feature_data_dir)

    # output_filename = feature_data_dir + ".pkl"
    # load_data_from_origin(args, sequences, 0, 100, output_filename)

    pool = multiprocessing.Pool(args.processor_num)
    for i in range(0, total_task_num):
        start_seq = i * args.split_seq_size
        end_seq = (i + 1) * args.split_seq_size
        output_filename = feature_data_dir + str(i) + ".pkl"
        # self.feature_file_list.append(output_filename)
        # print("start process:" + str(i))
        pool.apply_async(load_data_from_origin, (args, sequences, start_seq, end_seq, output_filename, ))

    print("pool close")
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()