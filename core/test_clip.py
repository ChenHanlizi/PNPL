import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import sklearn.metrics as sk
from utils import label_transform
from core import evaluation
from transformers import CLIPTokenizer
from transformers import CLIPModel
import time
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.filters import threshold_triangle
import statsmodels.api as sm
from scipy.stats import genpareto
import numpy as np
import cv2

def test_clip(net, criterion, testloader, outloader, epoch=None, **options):

    
    
    correct, total = 0, 0

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels = [], [], []
    known_number = {}
    correct_number = {}
    all_results = {}
    with torch.no_grad():
        for data, labels in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            
            with torch.set_grad_enabled(False):
                logits,_ = net(data)
                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()
                for i in range(len(labels.data)):
                    if labels.data[i].item() not in known_number.keys():
                        known_number[labels.data[i].item()] = 0
                        correct_number[labels.data[i].item()] = 0
                        all_results[labels.data[i].item()] = {}
                    if predictions[i].item() not in all_results[labels.data[i].item()].keys():
                        all_results[labels.data[i].item()][predictions[i].item()] = 0
                    all_results[labels.data[i].item()][predictions[i].item()] += 1
                    known_number[labels.data[i].item()] += 1
                    if predictions[i] == labels.data[i]:
                        correct_number[labels.data[i].item()] += 1
                                        
                _pred_k.append(logits.data.cpu().numpy())
                _labels.append(labels.data.cpu().numpy())

        for batch_idx, (data, labels) in enumerate(outloader):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                logits,_ = net(data)
                ood_score = logits.data.cpu().numpy()

    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))


    _pred_k = np.concatenate(_pred_k, 0)
    # _pred_u = np.concatenate(_pred_u, 0)
    # _labels = np.concatenate(_labels, 0)
    
    # # Out-of-Distribution detction evaluation
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_k, axis=1)
    results = evaluation.metric_ood(x1, x2)['Bas']
    

    results['ACC'] = acc

    return results
def returnCAM(feature_conv, weight_softmax, class_idx):
    b, c, h, w = feature_conv.shape  
    output_cam = []
    for idx in class_idx: 
        cam = weight_softmax[idx].dot(feature_conv.reshape((c, h * w)))  
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  # 归一化
        cam_img = np.uint8(255 * cam_img)  # 转为 uint8
        output_cam.append(cam_img)
    return output_cam
def test_nega_clip(net, criterion, testloader, outloader, epoch=None, **options):
    correct, total = 0, 0
    _pred_k, _pred_u, _labels = [], [], []
    logits_posi_id, logits_nega_id, logits_posi_ood, logits_nega_ood = [], [], [], []

    with torch.no_grad():
        if torch.cuda.device_count() > 1:
            prompts = net.module.prompt_learner()
            tokenized_prompts = net.module.tokenized_prompts
            text_features = net.module.text_encoder(prompts, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            prompts = net.prompt_learner()
            tokenized_prompts = net.tokenized_prompts
            text_features = net.text_encoder(prompts, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        torch.cuda.empty_cache()
        # breakpoint()
        all_predictions = []
        all_labels = []

        dataset = options['dataset']

        tqdm_object = tqdm(testloader, total=len(testloader))
        for batch_idx, (data, labels) in enumerate(tqdm_object):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()

            dataset = options['dataset']

            if torch.cuda.device_count() > 1:
                logits, _ = net.module.forward_test(data, text_features)
                logits /= net.module.logit_scale.exp()
            else:
                logits, _ = net.forward_test(data, text_features)
                logits /= net.logit_scale.exp()
            predictions, ood_score, logits_posi, logits_negas, _ = get_ood_score(logits, options)
            _pred_k.append(ood_score)
            # Append predictions and labels to lists
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.data.cpu().numpy())
    
            correct += (predictions == labels.data).sum()
            total += labels.size(0)
            _labels.append(labels.data.cpu().numpy())
            logits_posi_id.append(logits_posi.data.cpu().numpy())
            logits_nega_id.append(logits_negas.data.cpu().numpy())
        # Concatenate predictions and labels across batches
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        pre=all_predictions
        gt=all_labels
        different_indices = np.where(all_predictions != all_labels)[0]
        # print(f"Different indices between predictions and labels: {different_indices}")

        acc = float(correct) * 100. / float(total)
        print('Acc: {:.5f}'.format(acc))
        ood_predictions1=[]
        ood_labels1=[]
        tqdm_object = tqdm(outloader, total=len(outloader))
        unknown_confidence = []
        for batch_idx, (data, labels) in enumerate(tqdm_object):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()

            with torch.set_grad_enabled(False):
                dataset =  options['dataset']

                if torch.cuda.device_count() > 1:
                    logits, _ = net.module.forward_test(data, text_features)
                    logits /= net.module.logit_scale.exp()
                else:
                    logits, _ = net.forward_test(data, text_features)
                    logits /= net.logit_scale.exp()
                predictions1, ood_score, logits_posi, logits_negas, confidence = get_ood_score(logits, options)
                _pred_u.append(ood_score)
            ood_predictions1.append(predictions1.cpu().numpy())
            ood_labels1.append(labels.data.cpu().numpy())
                # unknown_confidence.append(confidence)
            logits_posi_ood.append(logits_posi.data.cpu().numpy())
            logits_nega_ood.append(logits_negas.data.cpu().numpy())
        ood_predictions1 = np.concatenate(ood_predictions1)
        ood_labels1 = np.concatenate(ood_labels1) 
        final_pre = np.concatenate((pre,ood_predictions1))
        final_gt = np.concatenate((gt, ood_labels1))
        np.savetxt('final_gt.txt', final_gt.astype(int), fmt='%d')
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    # print(_pred_k)
    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels = np.concatenate(_labels, 0)
    # Out-of-Distribution detction evaluation
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    # Output file paths
    output_file = "ood分数分两行.txt"

    # Convert arrays to strings
    x1_str = ' '.join(map(str, x1))
    x2_str = ' '.join(map(str, x2))

    # Write x1 and x2 strings to the output file
    with open(output_file, 'w') as file:
        file.write(x1_str + '\n')
        file.write(x2_str + '\n')

    results = evaluation.metric_ood(x1, x2)['Bas']
    
    # save _pred_k, -pred_u
    score_dic = {}
    score_dic['pred_k'] = _pred_k
    score_dic['pred_u'] = _pred_u
    score_dic['logits_posi_id'] = np.concatenate(logits_posi_id, 0)
    score_dic['logits_nega_id'] = np.concatenate(logits_nega_id, 0)
    score_dic['logits_posi_ood'] = np.concatenate(logits_posi_ood, 0)
    score_dic['logits_nega_ood'] = np.concatenate(logits_nega_ood, 0)
    # OSCR
    _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    auroc, aupr, fpr95, roc_threshold, otsu_threshold, kmeans, evt_threshold, new_evt_threshold = compute_fpr(x1, x2)
    file_path = 'ood分数分两行.txt' 
    input_a, input_b = read_inputs_from_file(file_path)
    count_a_lt_t7, count_b_gt_t7, len_a7, len_b7 ,indices_a_gt_t7, unknownfalseindices7 = count_elements_from_numpy_str(input_a, input_b, t=kmeans)
    pre_known_as_unknown = np.zeros(len_a7)
    pre_unknown_as_unknown = np.zeros(len_b7)
    pre_known_as_unknown[indices_a_gt_t7] = -2
    pre_unknown_as_unknown[unknownfalseindices7] = -2
    pre_as_unknown = np.concatenate((pre_known_as_unknown, pre_unknown_as_unknown))
    final_pre[pre_as_unknown == -2] = -2
    # 输出最终的 final_pre
    final_final_pre = final_pre
    # print(final_final_pre)
    np.savetxt('final_final_pre.txt', final_final_pre.astype(int), fmt='%d')
    # print(indices_a_gt_t)
    merged_indices7 = np.union1d(indices_a_gt_t7, different_indices)
    length_of_merged_indices7 = len(merged_indices7)

    oa7 = float(len_a7 - length_of_merged_indices7 + count_b_gt_t7)/float(len_a7 + len_b7)

    # oa = compute_oa(_pred_k, _pred_u, _labels)
    results['OA'] = oa7 * 100
    results['ACC'] = acc
    results['OSCR'] = _oscr_socre * 100.
    results['FPR95'] = fpr95 * 100.
    results['AUPR'] = aupr * 100.
    return results


def count_elements_from_numpy_str(input_a, input_b, t=0):
    # Convert input strings to NumPy arrays
    a = np.fromstring(input_a, dtype=float, sep=' ')
    b = np.fromstring(input_b, dtype=float, sep=' ')
    
    count_a_gt_t = np.sum(a > t)
    count_b_lt_t = np.sum(b < t)
    len_a = len(a)
    len_b = len(b)
    
    indices_a_gt_t = np.where(a < t)[0]
    
    indices_b_gt_t = np.where(b < t)[0]
    
    return count_a_gt_t, count_b_lt_t, len_a, len_b, indices_a_gt_t, indices_b_gt_t

def read_inputs_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        input_a = lines[0].strip()
        input_b = lines[1].strip()
    return input_a, input_b

def read_floats_from_file(file_path):
    with open(file_path, 'r') as file:
        floats = [float(line.strip()) for line in file]
    return floats
# 找出最佳阈值
def find_best_threshold(data):
    # 将列表转换为NumPy数组
    data_array = np.array(data)
    # 使用Otsu's方法
    threshold = threshold_otsu(data_array)
    threshold1 = threshold_otsu_median(data_array)
    return threshold


# 步骤 1：绘制均值残差图
def plot_mean_residual_life(data, num_thresholds=50):
    thresholds = np.linspace(np.min(data), np.max(data), num_thresholds)
    mean_residuals = [np.mean(data[data > t] - t) for t in thresholds]

    return thresholds, mean_residuals
# 步骤 2：选择初步阈值并拟合GPD
def fit_gpd(data, threshold):
    exceedances = data[data > threshold] - threshold
    params = genpareto.fit(exceedances)
    return params, exceedances

def EVT_threshold(data, num_thresholds=50):
    thresholds, mean_residuals = plot_mean_residual_life(data, num_thresholds)

    best_threshold = None
    best_score = float('inf')

    for threshold in thresholds:
        if len(data[data > threshold]) < 30: 
            continue

        params, exceedances = fit_gpd(data, threshold)

        neg_log_likelihood = -genpareto.logpdf(exceedances, *params).sum()

        if neg_log_likelihood < best_score:
            best_score = neg_log_likelihood
            best_threshold = threshold

    return best_threshold



def plot_mean_residual_life1(data, num_thresholds=200, min_threshold_ratio=0, max_threshold_ratio=1):
    min_value = np.min(data)
    max_value = np.max(data)
    thresholds = np.linspace(min_value + (max_value - min_value) * min_threshold_ratio, 
                             min_value + (max_value - min_value) * max_threshold_ratio, 
                             num_thresholds)
    mean_residuals = [np.mean(data[data > t] - t) for t in thresholds]

    return thresholds, mean_residuals
def aic_score(exceedances, params):
    k = len(params)
    log_likelihood = -genpareto.logpdf(exceedances, *params).sum()
    return 2 * k - 2 * log_likelihood

def bic_score(exceedances, params):
    n = len(exceedances)
    k = len(params)
    log_likelihood = -genpareto.logpdf(exceedances, *params).sum()
    return np.log(n) * k - 2 * log_likelihood

def automatic_threshold_selection1(data, num_thresholds=200, min_exceedances=20, criterion='aic'):
    thresholds, mean_residuals = plot_mean_residual_life1(data, num_thresholds)

    best_threshold = None
    best_score = float('inf')

    for threshold in thresholds:
        exceedances_count = len(data[data > threshold])
        if exceedances_count < min_exceedances:
            continue

        params, exceedances = fit_gpd(data, threshold)
        if exceedances is None:
            continue

        if criterion == 'aic':
            score = aic_score(exceedances, params)
        elif criterion == 'bic':
            score = bic_score(exceedances, params)
        else:
            score = -genpareto.logpdf(exceedances, *params).sum()

        if score < best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold


def threshold_otsu_median(data_array):
    """
    Compute Otsu threshold using median instead of mean.
    Parameters:
        data_array : ndarray
            Input data array.
    Returns:
        threshold : float
            Computed threshold.
    """
    if len(data_array.shape) > 1:
        raise ValueError("Input data_array should be 1-dimensional")

    hist, bin_edges = np.histogram(data_array, bins=256, range=(data_array.min(), data_array.max()))
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    total_weight = weight1[-1]
    probability1 = weight1 / total_weight
    probability2 = weight2 / total_weight

    all_median1 = []
    all_median2 = []

    for t in bin_mids:
        if np.sum(data_array <= t) == 0 or np.sum(data_array > t) == 0:
            all_median1.append(0)
            all_median2.append(0)
        else:
            all_median1.append(np.median(data_array[data_array <= t]))
            all_median2.append(np.median(data_array[data_array > t]))

    all_median1 = np.array(all_median1)
    all_median2 = np.array(all_median2)

    variance12 = probability1 * probability2 * (all_median1 - all_median2) ** 2

    idx = np.argmax(variance12)
    threshold = bin_mids[idx]

    return threshold

def read_data(file_path):
    with open(file_path, 'r') as file:
        data = [float(line.strip()) for line in file]
    return np.array(data)

from sklearn.cluster import KMeans
# K均值聚类
# def kmeans_threshold(data, n_clusters=2, ratio=0.75):
def kmeans_threshold(data, n_clusters=3):
    data = data.reshape(-1, 1)  # 将数据转换为列向量
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    labels = kmeans.labels_
    centers = sorted(kmeans.cluster_centers_.flatten())
    
    # 计算每个聚簇的数量
    cluster_counts = np.bincount(labels)
    
    # 计算阈值并返回
    threshold = (centers[1] + centers[0]) / 2
    return threshold

def compute_fpr(pred_k, pred_u):
        x1 = pred_k
        x2 = pred_u
        pos = np.array(x1[:]).reshape((-1, 1))
        neg = np.array(x2[:]).reshape((-1, 1))
        examples = np.squeeze(np.vstack((pos, neg)))
        labels = np.zeros(len(examples), dtype=np.int32)
        labels[:len(pos)] += 1

        # print('labels length:', labels.shape)
        
        labels_file = '标签值.txt'
        examples_file = 'ood分数.txt'
        # 将labels输出到labels.txt
        with open(labels_file, 'w') as file:
            for label in labels:
                file.write(f"{label}\n")
        
        # 将examples输出到examples.txt
        with open(examples_file, 'w') as file:
            for example in examples:
                file.write(f"{example}\n")
        
        file_path1 = 'ood分数.txt'
        data1 = read_data(file_path1)
        threshold_kmeans = kmeans_threshold(data1)

        input_file = "ood分数.txt"  # 请替换为实际的txt文件路径
        data = read_floats_from_file(input_file)
        data_array = np.array(data)

        evt_threshold = EVT_threshold(data_array)
        best_threshold = find_best_threshold(data)
        new_evt_threshold = automatic_threshold_selection1(data_array, num_thresholds=200, min_exceedances=300, criterion='aic')

        auroc = sk.roc_auc_score(labels, examples)
        aupr = sk.average_precision_score(labels, examples)
        fpr95 = fpr_and_fdr_at_recall(labels, examples)

        # Compute ROC curve
        fpr, tpr, thresholds = sk.roc_curve(labels, examples, pos_label=1)
        # 计算距离左上角最近的点的索引
        idx = np.argmax(tpr - fpr)
        best_threshold1 = thresholds[idx]
        # print('约登点对应阈值：', best_threshold1)

        
        return auroc, aupr, fpr95, best_threshold1, best_threshold, threshold_kmeans, evt_threshold, new_evt_threshold
        


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

def get_ood_score(logits, options):
    n_nega_ctx = options['NEGA_CTX']
    softmax_logits = F.softmax(logits, dim=1)
    softmax_logits = softmax_logits.view(-1, int(softmax_logits.shape[1]/(1+n_nega_ctx)), 1+n_nega_ctx)
    logits = logits.view(-1, int(logits.shape[1]/(1+n_nega_ctx)), 1+n_nega_ctx)
    softmax_logits_posi = softmax_logits[:, :, 0]
    softmax_logits_negas = softmax_logits[:, :, 1:]
    logits_posi = softmax_logits[:, :, 0]
    logits_negas = softmax_logits[:, :, 1:]
    predictions = softmax_logits_posi.data.max(1)[1]
    confidence_scores = softmax_logits_posi.data.cpu().numpy()
    ood_score = softmax_logits_posi.data.cpu().numpy()
    if options['open_score'] == 'msp':
        ood_score = softmax_logits_posi.data.cpu().numpy()
    return predictions, ood_score, logits_posi, logits_negas, confidence_scores
    