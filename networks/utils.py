import logging
import os
import shutil
import sys
import scipy.sparse as sparse
from medpy.metric.binary import assd, dc
import numpy as np
import torch
from medpy.metric.binary import assd, dc


def save_checkpoint(state, is_best, checkpoint_dir, logger=None):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.

    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    def log_info(message):
        if logger is not None:
            logger.info(message)

    if not os.path.exists(checkpoint_dir):
        log_info(
            f"Checkpoint directory does not exists. Creating {checkpoint_dir}")
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    log_info(f"Saving last checkpoint to '{last_file_path}'")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        log_info(f"Saving best checkpoint to '{best_file_path}'")
        shutil.copyfile(last_file_path, best_file_path)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.

    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied

    Returns:
        state
    """
    # if not os.path.exists(checkpoint_path):
    #     raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")
    #
    # state = torch.load(checkpoint_path, map_location='cpu')
    # model.load_state_dict(state['model_state_dict'])
    #
    # if optimizer is not None:
    #     optimizer.load_state_dict(state['optimizer_state_dict'])
    #
    # return state

    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location='cpu')
    pretrained_dict = state['model_state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer_state_dict'])

    return state


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Logging to console
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


def find_maximum_patch_size(model, device):
    """Tries to find the biggest patch size that can be send to GPU for inference
    without throwing CUDA out of memory"""
    logger = get_logger('PatchFinder')
    in_channels = model.in_channels

    patch_shapes = [(64, 128, 128), (96, 128, 128),
                    (64, 160, 160), (96, 160, 160),
                    (64, 192, 192), (96, 192, 192)]

    for shape in patch_shapes:
        # generate random patch of a given size
        patch = np.random.randn(*shape).astype('float32')

        patch = torch \
            .from_numpy(patch) \
            .view((1, in_channels) + patch.shape) \
            .to(device)

        logger.info(f"Current patch size: {shape}")
        model(patch)


def unpad(probs, index, shape, pad_width=8):
    def _new_slices(slicing, max_size):
        if slicing.start == 0:
            p_start = 0
            i_start = 0
        else:
            p_start = pad_width
            i_start = slicing.start + pad_width

        if slicing.stop == max_size:
            p_stop = None
            i_stop = max_size
        else:
            p_stop = -pad_width
            i_stop = slicing.stop - pad_width

        return slice(p_start, p_stop), slice(i_start, i_stop)

    D, H, W = shape

    i_c, i_z, i_y, i_x = index
    p_c = slice(0, probs.shape[0])

    p_z, i_z = _new_slices(i_z, D)
    p_y, i_y = _new_slices(i_y, H)
    p_x, i_x = _new_slices(i_x, W)

    probs_index = (p_c, p_z, p_y, p_x)
    index = (i_c, i_z, i_y, i_x)
    return probs[probs_index], index


def create_feature_maps(init_channel_number, number_of_fmaps):
    return [init_channel_number * 2 ** k for k in range(number_of_fmaps)]


# Code taken from https://github.com/cremi/cremi_python
def adapted_rand(seg, gt, all_stats=False):
    """Compute Adapted Rand error as defined by the SNEMI3D contest [1]
    Formula is given as 1 - the maximal F-score of the Rand index
    (excluding the zero component of the original labels). Adapted
    from the SNEMI3D MATLAB script, hence the strange style.
    Parameters
    ----------
    seg : np.ndarray
        the segmentation to score, where each value is the label at that point
    gt : np.ndarray, same shape as seg
        the groundtruth to score against, where each value is a label
    all_stats : boolean, optional
        whether to also return precision and recall as a 3-tuple with rand_error
    Returns
    -------
    are : float
        The adapted Rand error; equal to $1 - \frac{2pr}{p + r}$,
        where $p$ and $r$ are the precision and recall described below.
    prec : float, optional
        The adapted Rand precision. (Only returned when `all_stats` is ``True``.)
    rec : float, optional
        The adapted Rand recall.  (Only returned when `all_stats` is ``True``.)
    References
    ----------
    [1]: http://brainiac2.mit.edu/SNEMI3D/evaluation
    """
    # just to prevent division by 0
    epsilon = 1e-6

    # segA is truth, segB is query
    segA = np.ravel(gt)
    segB = np.ravel(seg)
    n = segA.size

    n_labels_A = np.amax(segA) + 1
    n_labels_B = np.amax(segB) + 1

    ones_data = np.ones(n)

    p_ij = sparse.csr_matrix((ones_data, (segA[:], segB[:])), shape=(n_labels_A, n_labels_B))

    a = p_ij[1:n_labels_A, :]
    b = p_ij[1:n_labels_A, 1:n_labels_B]
    c = p_ij[1:n_labels_A, 0].todense()
    d = b.multiply(b)

    a_i = np.array(a.sum(1))
    b_i = np.array(b.sum(0))

    sumA = np.sum(a_i * a_i)
    sumB = np.sum(b_i * b_i) + (np.sum(c) / n)
    sumAB = np.sum(d) + (np.sum(c) / n)

    precision = sumAB / max(sumB, epsilon)
    recall = sumAB / max(sumA, epsilon)

    fScore = 2.0 * precision * recall / max(precision + recall, epsilon)
    are = 1.0 - fScore

    if all_stats:
        return are, precision, recall
    else:
        return are

def dice_per_class(prediction, target, eps=1e-10):
    '''

    :param prediction: numpy array
    :param target: numpy array
    :param eps:
    :return:
    '''
    prediction = prediction.astype(np.float)
    target = target.astype(np.float)
    intersect = np.sum(prediction * target)
    return (2. * intersect / (np.sum(prediction) + np.sum(target) + eps))

def intersect_per_class(prediction, target, eps=1e-10):
    '''

    :param prediction: numpy array
    :param target: numpy array
    :param eps:
    :return:
    '''
    prediction = prediction.astype(np.float)
    target = target.astype(np.float)
    intersect = np.sum(prediction * target)
    return intersect, np.sum(prediction), np.sum(target)

def dice_all_class(prediction, target, class_num=20, eps=1e-10):
    '''

    :param prediction: numpy array with shape of [D, H, W], [H, W], or [H, W, D]
    :param target: numpy array with shape of [D, H, W], [H, W], or [H, W, D]
    :param class_num:
    :param eps:
    :return:
    '''
    dices = []
    for i in range(1, class_num):
        if i not in target and i not in prediction:
            continue
        target_per_class = np.where(target == i, 1, 0)
        prediction_per_class = np.where(prediction == i, 1, 0)
        dice = dice_per_class(prediction_per_class, target_per_class)
        dices.append(dice)
    return np.mean(dices)

def iou_per_class(prediction, target, eps=1e-10):
    prediction = prediction.astype(np.float)
    target = target.astype(np.float)
    intersect = np.sum(prediction * target)
    union = np.sum((prediction + target) > 0)
    return (intersect / (union + eps))

def miou_all_class(prediction, target, class_num=20, eps=1e-10):
    ious = []
    for i in range(1, class_num):
        if i not in target and i not in prediction:
            continue
        target_per_class = np.where(target == i, 1, 0)
        prediction_per_class = np.where(prediction == i, 1, 0)
        iou = iou_per_class(prediction_per_class, target_per_class)
        ious.append(iou)
    return np.mean(ious)


def dices_each_class(prediction, target, class_num=20, eps=1e-10, empty_value=-1.0):
    '''

    :param prediction: numpy array with shape of [D, H, W], [H, W], or [H, W, D]
    :param target: numpy array with shape of [D, H, W], [H, W], or [H, W, D]
    :param class_num:
    :param eps:
    :return:
    '''
    dices = empty_value * np.ones((class_num), dtype=np.float32)
    for i in range(0, class_num):
        if i not in target and i not in prediction:
            continue
        target_per_class = np.where(target == i, 1, 0)
        prediction_per_class = np.where(prediction == i, 1, 0)
        # dice = dice_per_class(prediction_per_class, target_per_class)
        dice = dc(prediction_per_class, target_per_class)
        dices[i] = dice
    return dices

def dice_whole_class(prediction, target, class_num=20, eps=1e-10):
    '''

    :param prediction: numpy array with shape of [D, H, W], [H, W], or [H, W, D]
    :param target: numpy array with shape of [D, H, W], [H, W], or [H, W, D]
    :param class_num:
    :param eps:
    :return:
    '''
    intersect_sum = 0
    prediction_sum = 0
    target_sum = 0
    for i in range(1, class_num):
        if i not in target:
            continue
        target_per_class = np.where(target == i, 1, 0)
        prediction_per_class = np.where(prediction == i, 1, 0)
        result = intersect_per_class(prediction_per_class, target_per_class)
        intersect_sum += result[0]
        prediction_sum += result[1]
        target_sum += result[2]
    return (2. * intersect_sum / (prediction_sum + target_sum + eps))

def assds_each_class(prediction, target, class_num=20, eps=1e-10, voxel_size=(1,1,1), empty_value=-1.0, connectivity=1):
    assds = empty_value * np.ones((class_num), dtype=np.float32)
    for i in range(0, class_num):
        if i not in target:
            continue
        if i not in prediction:
            print('label %d is zero' % i)
            continue
        target_per_class = np.where(target == i, 1, 0)
        prediction_per_class = np.where(prediction == i, 1, 0)
        ad = assd(prediction_per_class, target_per_class, voxelspacing=voxel_size, connectivity=connectivity)
        assds[i] = ad
    return assds

def evaluation_metrics_each_class(prediction, target, class_num=20, eps=1e-10, empty_value=-1.0):
    dscs = empty_value * np.ones((class_num), dtype=np.float32)
    precisions = empty_value * np.ones((class_num), dtype=np.float32)
    recalls = empty_value * np.ones((class_num), dtype=np.float32)
    for i in range(0, class_num):
        if i not in target and i not in prediction:
            continue
        target_per_class = np.where(target == i, 1, 0).astype(np.float32)
        prediction_per_class = np.where(prediction == i, 1, 0).astype(np.float32)

        tp = np.sum(prediction_per_class * target_per_class)
        fp = np.sum(prediction_per_class) - tp
        fn = np.sum(target_per_class) - tp

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        dsc = 2 * tp / (2 * tp + fp + fn + eps)

        dscs[i] = dsc
        precisions[i] = precision
        recalls[i] = recall
    return dscs, precisions, recalls

def evaluation_accuracy(prediction, target, class_num=20):
    voxel_num = np.size(target)
    voxel_num = np.float32(voxel_num)
    tp = 0.0
    for i in range(0, class_num):
        if i not in target and i not in prediction:
            continue
        target_per_class = np.where(target == i, 1, 0).astype(np.float32)
        prediction_per_class = np.where(prediction == i, 1, 0).astype(np.float32)

        tp += np.sum(prediction_per_class * target_per_class)

    accuracy = tp / voxel_num


    return accuracy



def np_onehot(label, num_classes):
    return np.eye(num_classes)[label.astype(np.int32)]
