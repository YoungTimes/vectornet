import re
import sys

from tensorflow.python.keras.losses import huber
from preprocess import ArgoverseData
from model import VectorNet
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from argoverse.map_representation.map_api import ArgoverseMap

from argoverse.evaluation.eval_forecasting import get_displacement_errors_and_miss_rate

from shapely.geometry import Point, Polygon, LineString, LinearRing
from shapely.affinity import affine_transform, rotate

import tensorflow as tf
import datetime
# TODO: learn about generator

# dataset_filename = "./train_dataset.pkl"
# f = open(dataset_filename, 'rb')
# data = pickle.load(f)
# f.close()

# dataset = tf.data.Dataset.from_tensor_slices((data["feature"], data["label"]))

def get_coef(logits):
    mux, muy, sx, sy, corr = tf.split(logits, 5, -1)

    sx = tf.exp(sx)
    sy = tf.exp(sy)

    corr = tf.tanh(corr)

    return (mux, muy, sx, sy, corr)


def guass_likelihood_loss(logits, labels):
    mux, muy, sx, sy, corr = get_coef(logits)

    # print(mux)

    # mux = tf.cumsum(mux, axis = 1)
    # muy = tf.cumsum(muy, axis = 1)
    # print("=======================")

    # print(mux)

    # sys.exit(-1)

    [x_data, y_data] = tf.split(labels, 2, -1)

    epsilon = 1e-20
    sx = tf.math.maximum(sx, epsilon)
    sy = tf.math.maximum(sy, epsilon)

    sxsy = tf.math.multiply(sx, sy)
    sxsy = tf.math.maximum(sxsy, epsilon)

    norm_x = tf.math.subtract(x_data, mux)
    norm_y = tf.math.subtract(y_data, muy)

    # # print(x_data)
    # print("mux")
    # print(mux)

    # # print(y_data)
    # print("muy")
    # print(muy)

    z = tf.math.square(tf.math.divide(norm_x, sx)) \
        + tf.math.square(tf.math.divide(norm_y, sy)) \
        - 2 * tf.math.divide(tf.math.multiply(corr, tf.math.multiply(norm_x, norm_y)), sxsy)

    neg_rho = 1 - tf.square(corr)
    neg_rho = tf.math.maximum(neg_rho, epsilon)

    result = tf.math.exp(tf.math.divide(-z, 2 * neg_rho))

    denom = 2 * np.pi * tf.multiply(sxsy, tf.math.sqrt(neg_rho))
    denom = tf.math.maximum(denom, epsilon)

    result = tf.math.divide(result, denom)

    # print(result)

    result = -tf.math.log(tf.math.maximum(result, epsilon))

    # print(result.shape)

                # mask = tf.math.logical_not(tf.math.equal(real, 0))
                # loss_ = loss_object(real, pred)

                # mask = tf.cast(mask, dtype=loss_.dtype)
                # loss_ *= mask

    return tf.reduce_sum(result) / x_data.shape[0]

def huber_loss(predictions, labels, delta = 1.0):
    return tf.keras.losses.Huber(delta=1.0, reduction=tf.keras.losses.Reduction.SUM)(labels, predictions) / predictions.shape[0]

#     return tf.losses.huber_loss(labels, predictions, delta = delta)

#     residual = tf.abs(predictions - labels)
#     condition = tf.less(residual, delta)
#     small_res = 0.5 * tf.square(residual)
#     large_res = delta * residual - 0.5 * tf.square(delta)

#     loss_value = tf.where(condition, small_res, large_res)

#     return tf.reduce_sum(loss_value) / labels.shape[0]

def train(args):
    dataset = ArgoverseData(args)
    dataset.load_data(force_reproduce = False)

    lr_scheduler =tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=5*dataset.num_batchs, decay_rate=0.7, staircase=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

    vectornet = VectorNet(args)

    print("num_epochs:" + str(args.num_epochs))
    print("num batchs:" + str(dataset.num_batchs))

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    
    
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience = 2, verbose = 0, restore_best_weights = False)
    callbacks = tf.keras.callbacks.CallbackList([early_stop_callback], model=vectornet)
    logs = {}

    callbacks.on_train_begin(logs = logs)
    for epoch in range(args.num_epochs):
        loss = 0
        callbacks.on_epoch_begin(epoch, logs = logs)
        for batch in range(dataset.num_batchs):
            x_batch, y_batch, mask_batch, graph_mask_batch, param_batch, origin_input_traj_batch, pid_batch, node_mask_batch, origin_features_batch = dataset.next_batch()

            # print("mask batch:")
            # print(mask_batch.shape)

            # print("x batch:")
            # print(x_batch.shape)


            with tf.GradientTape() as tape:
                logits, scores, node_cmp, node_true = vectornet([x_batch, mask_batch, graph_mask_batch, pid_batch, node_mask_batch], training = True)

                # print(logits)

                # logits = tf.cast(logits, dtype=tf.float32)

                guass_loss_val = guass_likelihood_loss(logits, y_batch)

                # node_true = tf.boolean_mask(x_batch, node_mask_batch)


                # mask = tf.math.logical_not(tf.math.equal(node_mask_batch, True))

                huber_loss_val = huber_loss(node_cmp, node_true)

                # print("guass loss:" + str(guass_loss_val) + ", huber loss:" + str(huber_loss_val))

                loss = guass_loss_val + huber_loss_val

                # print(loss)

                # print("{}/{}-{}/{}, loss:{}".format(epoch, args.num_epochs, batch, dataset.num_batchs, loss))

                grads = tape.gradient(loss, vectornet.trainable_variables)

                optimizer.apply_gradients(zip(grads, vectornet.trainable_variables))

                
                # print(optimizer._decayed_lr('float32').numpy())
                # print("learning_rate:" + str(optimizer.learning_rate))

        print("{}/{}, loss:{}".format(epoch, args.num_epochs, loss))

        train_loss(loss)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            # tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)


        obs_traj_idx = np.argwhere(x_batch[:, :, 0, [4]] == 2)



        # print(obs_traj_idx[:, :2])

        obs_traj_scores = tf.gather_nd(scores, obs_traj_idx[:, :2])

        scores_top_k = tf.math.top_k(obs_traj_scores, 3, sorted = False)
        # print("++++++++++++++=======================")
        # print(scores_top_k)
        scores_top_k_idx = scores_top_k.indices

#         print("================================+++")
#         print(scores_top_k_idx)

        origin_features = []
#         print("origin_features_batch shape:" + str(origin_features_batch.shape))
#         print(origin_features_batch)
#         print("=================")
        for origin_feature_idx in range(len(scores_top_k_idx)):
            origin_features.append(origin_features_batch[origin_feature_idx, scores_top_k_idx[origin_feature_idx]])

        # print(np.array(origin_features).shape)
        # loss_seq.append(loss)
#         early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 3)
#         early_stopping.set_model(vectornet)

        logs.update({"loss":loss})
        callbacks.on_epoch_end(epoch, logs = logs)

        if (epoch + 1) % args.metric_every == 0 or epoch + 1 == args.num_epochs or vectornet.stop_training:
            measure_metric(args, epoch, vectornet, x_batch, y_batch, mask_batch, graph_mask_batch, param_batch, origin_input_traj_batch, pid_batch, node_mask_batch, origin_features)

#         if vectornet.stop_training:
#             print("Callback_EarlyStopping signal received at epoch= %d/%d, Terminating training"%(epoch, args.num_epochs))
#             break

    callbacks.on_train_end(logs = logs)
    # get_displacement_errors_and_miss_rate()

    # forecasted_trajectories: Dict[int, List[np.ndarray]],
    # gt_trajectories: Dict[int, np.ndarray],
    # max_guesses: int,
    # horizon: int,
    # miss_threshold: float,
    # forecasted_probabilities: Optional[Dict[int, List[float]]] = None,
def measure_metric(args, epoch, model, features, labels, masks, graph_masks, params, origin_input_trajs, pids, node_masks, origin_features):
    logits, scores, node_cmp, node_true = model([features, masks, graph_masks, pids, node_masks], training = False)
    mux, muy, sx, sy, corr = get_coef(logits)

    # print(labels)

    forecasted_trajectories = {}
    gt_trajectories = {}
    for j in range(args.batch_size):
        pred_trajs = []
        for i in range(args.pred_len):
            mean = [mux[j][i][0], muy[j][i][0]]
            # cov = [[sx[j][i][0] * sx[j][i][0], corr[j][i][0] * sx[j][i][0] * sy[j][i][0]], 
            #     [corr[j][i][0] *  sx[j][i][0] * sy[j][i][0], sy[j][i][0] * sy[j][i][0]]]

            #mean = np.array(mean, dtype='float')
            #cov = np.array(cov, dtype='float')
            #next_values = np.random.multivariate_normal(mean, cov, 1)

            pred_trajs.append(mean)

        # print(np.array(pred_trajs).shape)
        # print(labels[j].shape)

        seq_id = params[j][0]

        pred_trajs = np.array(pred_trajs)
        # pred_trajs = pred_trajs.cumsum(axis = 0)

        # print("------------------")
        # print(pred_trajs)
        # print("------------------============")
        # print(labels[j])
        # print("------------------")

        # print("=========================")
        translation = [float(params[j][2]), float(params[j][3])]
        rotation = float(params[j][6])

        scale_x = float(params[j][4])
        scale_y = float(params[j][5])

        for k in range(1, len(pred_trajs)):
            pred_trajs[k] = pred_trajs[k - 1] + pred_trajs[k]

        # pred_trajs[:, [0]] = pred_trajs[:, [0]] * scale_x
        # pred_trajs[:, [1]] = pred_trajs[:, [1]] * scale_y

        pred_trajs = normalized_to_map_coordinates(pred_trajs, translation, rotation)

        # print(pred_trajs)

        for k in range(1, len(labels[j])):
            labels[j][k] = labels[j][k - 1] + labels[j][k]

        # labels[j][:, [0]] = labels[j][:, [0]] * scale_x
        # labels[j][:, [1]] = labels[j][:, [1]] * scale_y

        labels[j] = normalized_to_map_coordinates(labels[j], translation, rotation)

        forecasted_trajectories[seq_id] = np.array([pred_trajs])

        gt_trajectories[seq_id] = labels[j]

        # print(pred_trajs)
        # print("=======================")
        # print(labels[j])

        # sys.exit(-1)

    # if args.viz:
    # if args.viz:
    #     id_for_viz = None
    #     if args.viz_seq_id:
    #         with open(args.viz_seq_id, "rb") as f:
    #             id_for_viz = pkl.load(f)
    #     viz_predictions_helper(forecasted_trajectories, gt_trajectories,
    #                            features_df, id_for_viz)

    metric_results = get_displacement_errors_and_miss_rate(forecasted_trajectories, gt_trajectories, 1, args.pred_len, 2.0)

    print("------------------------------------------------")
    print(metric_results)
    print("------------------------------------------------")

    if epoch + 1 == args.num_epochs:
        viz_predictions_helper(forecasted_trajectories, gt_trajectories, params, origin_input_trajs, origin_features)

def normalized_to_map_coordinates(coords, translation, rotation):
    """Denormalize trajectory to bring it back to map frame.

    Args:
        coords (numpy array): Array of shape (num_tracks x seq_len x 2) containing normalized coordinates
        translation (list): Translation matrix used in normalizing trajectories
        rotation (list): Rotation angle used in normalizing trajectories 
    Returns:
        _ (numpy array: Array of shape (num_tracks x seq_len x 2) containing coordinates in map frame

    """
    # abs_coords = []
    # for i in range(coords.shape[0]):
    ls = LineString(coords)

    # Rotate
    ls_rotate = rotate(ls, -rotation, origin=(0, 0))

    # Translate
    M_inv = [1, 0, 0, 1, translation[0], translation[1]]

    ls_offset = affine_transform(ls_rotate, M_inv).coords[:]

    return np.array(ls_offset)
    #abs_coords.append(ls_offset)

    # return np.array(abs_coords)

def viz_predictions(
        input_: np.ndarray,
        output: np.ndarray,
        target: np.ndarray,
        centerlines: np.ndarray,
        city_names: np.ndarray,
        idx=None,
        show: bool = True,
) -> None:
    """Visualize predicted trjectories.

    Args:
        input_ (numpy array): Input Trajectory with shape (num_tracks x obs_len x 2)
        output (numpy array of list): Top-k predicted trajectories, each with shape (num_tracks x pred_len x 2)
        target (numpy array): Ground Truth Trajectory with shape (num_tracks x pred_len x 2)
        centerlines (numpy array of list of centerlines): Centerlines (Oracle/Top-k) for each trajectory
        city_names (numpy array): city names for each trajectory
        show (bool): if True, show

    """
    num_tracks = input_.shape[0]
    obs_len = input_.shape[1]
    pred_len = target.shape[1]

    plt.figure(0, figsize=(8, 7))
    avm = ArgoverseMap()
    for i in range(num_tracks):
        plt.plot(
            input_[i, :, 0],
            input_[i, :, 1],
            color="#ECA154",
            label="Observed",
            alpha=1,
            linewidth=3,
            zorder=15,
        )
        plt.plot(
            input_[i, -1, 0],
            input_[i, -1, 1],
            "o",
            color="#ECA154",
            label="Observed",
            alpha=1,
            linewidth=3,
            zorder=15,
            markersize=9,
        )
        plt.plot(
            target[i, :, 0],
            target[i, :, 1],
            color="#d33e4c",
            label="Target",
            alpha=1,
            linewidth=3,
            zorder=20,
        )
        plt.plot(
            target[i, -1, 0],
            target[i, -1, 1],
            "o",
            color="#d33e4c",
            label="Target",
            alpha=1,
            linewidth=3,
            zorder=20,
            markersize=9,
        )

        for j in range(len(centerlines[i])):
            plt.plot(
                centerlines[i][j][:, 0],
                centerlines[i][j][:, 1],
                "--",
                color="grey",
                alpha=1,
                linewidth=1,
                zorder=0,
            )

        for j in range(len(output[i])):
            plt.plot(
                output[i][j][:, 0],
                output[i][j][:, 1],
                color="#007672",
                label="Predicted",
                alpha=1,
                linewidth=3,
                zorder=15,
            )
            plt.plot(
                output[i][j][-1, 0],
                output[i][j][-1, 1],
                "o",
                color="#007672",
                label="Predicted",
                alpha=1,
                linewidth=3,
                zorder=15,
                markersize=9,
            )
            for k in range(pred_len):
                lane_ids = avm.get_lane_ids_in_xy_bbox(
                    output[i][j][k, 0],
                    output[i][j][k, 1],
                    city_names[i],
                    query_search_range_manhattan=5,
                )

        for j in range(obs_len):
            lane_ids = avm.get_lane_ids_in_xy_bbox(
                input_[i, j, 0],
                input_[i, j, 1],
                city_names[i],
                query_search_range_manhattan=5,
            )
            [avm.draw_lane(lane_id, city_names[i]) for lane_id in lane_ids]
        for j in range(pred_len):
            lane_ids = avm.get_lane_ids_in_xy_bbox(
                target[i, j, 0],
                target[i, j, 1],
                city_names[i],
                query_search_range_manhattan=5,
            )
            [avm.draw_lane(lane_id, city_names[i]) for lane_id in lane_ids]

        plt.axis("equal")
        plt.xticks([])
        plt.yticks([])
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        if show:
            plt.show()

def viz_predictions_helper(forecasted_trajectories, gt_trajectories, params, origin_input_trajs, origin_features):
    seq_ids = list(gt_trajectories.keys())
    for i in range(len(seq_ids)):
        seq_id = seq_ids[i]
        gt_trajectory = gt_trajectories[seq_id]
        output_trajectories = forecasted_trajectories[seq_id]
        candidate_centerlines = origin_features[i]
        input_trajectory = origin_input_trajs[i]

        # candidate_centerlines = np.array([])


# def viz_predictions_helper(
#         forecasted_trajectories: Dict[int, List[np.ndarray]],
#         gt_trajectories: Dict[int, np.ndarray],
#         features_df: pd.DataFrame,
#         viz_seq_id: Union[None, List[int]],
# ) -> None:

#     seq_ids = gt_trajectories.keys() if viz_seq_id is None else viz_seq_id
#     for seq_id in seq_ids:
#         gt_trajectory = gt_trajectories[seq_id]
#         curr_features_df = features_df[features_df["SEQUENCE"] == seq_id]
#         input_trajectory = (
#             curr_features_df["FEATURES"].values[0]
#             [:args.obs_len, [FEATURE_FORMAT["X"], FEATURE_FORMAT["Y"]]].astype(
#                 "float"))
#         output_trajectories = forecasted_trajectories[seq_id]
#         candidate_centerlines = curr_features_df[
#             "CANDIDATE_CENTERLINES"].values[0]
#         city_name = curr_features_df["FEATURES"].values[0][
#             0, FEATURE_FORMAT["CITY_NAME"]]

        gt_trajectory = np.expand_dims(gt_trajectory, 0)
        input_trajectory = np.expand_dims(input_trajectory, 0)
        output_trajectories = np.expand_dims(np.array(output_trajectories), 0)
        candidate_centerlines = np.expand_dims(np.array(candidate_centerlines), 0)
        city_name = np.array([params[i][1]])
        # print(city_name)

        viz_predictions(
            input_trajectory,
            output_trajectories,
            gt_trajectory,
            candidate_centerlines,
            city_name,
            show=True,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs', type = int, default = 50, help = 'Number of epochs')
    parser.add_argument('--obs_len', type = int, default = 20, help = "Observed length of the trajectory")
    parser.add_argument('--pred_len', type = int, default = 30, help = "Prediction length of the trajectory")
    parser.add_argument('--mode', default = 'train', type = str, help = 'train/val/test')
    parser.add_argument('--metric_every', default = 5, type = int, help = 'caculate metric every x epoch')
    parser.add_argument('--data_dir', default = "/home/liuyang/Documents/code/vectornet/data/", type = str, help = 'training data path')
    parser.add_argument('--batch_size', type = int, default = 32, help = "batch size")
    parser.add_argument('--split_seq_size', type = int, default = 1000, help = "split seq size")

    # root_dir = '../../vectornet/data/forecasting_sample/data/'

    # /home/featurize/data/data/
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()