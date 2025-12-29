import math
import torch
import numpy as np
from skimage.morphology import skeletonize
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch_topological.nn import CubicalComplex

getPersistentInfo = CubicalComplex(dim=2)

def cal_acc(pred, target):
    diff = torch.abs(pred - target)
    acc = torch.mean(1.0-diff)

    return acc.item()

def specificity(pred, target):
    TN = torch.sum((pred==0) & (target==0)).float()
    FP = torch.sum((pred==1) & (target==0)).float()

    if TN + FP == 0:
        return float('nan')
    else:
        result = TN / (TN + FP)
        if isinstance(result, torch.Tensor):
            result = result.item()
        return result

def sensitivity(pred, target):
    TP = torch.sum((pred==1) & (target==1)).float()
    FN = torch.sum((pred==0) & (target==1)).float()

    if TP + FN == 0:
        return float('nan')
    else:
        result = TP / (TP + FN)
        if isinstance(result, torch.Tensor):
            result = result.item()
        return result

def precision(pred, target):
    TP = torch.sum((pred == 1) & (target == 1)).float()
    FP = torch.sum((pred == 1) & (target == 0)).float()

    if TP + FP == 0:
        return float('nan')
    else:
        result = TP / (TP + FP)
        if isinstance(result, torch.Tensor):
            result = result.item()
        return result

def F1_SCORE(pred, target):
    TP = torch.sum((pred == 1) & (target == 1)).float()
    FP = torch.sum((pred == 1) & (target == 0)).float()
    FN = torch.sum((pred == 0) & (target == 1)).float()

    if TP + FP + FN == 0:
        return float('nan')
    else:
        result = (2 * TP) / (2 * TP + FP + FN)
        if isinstance(result, torch.Tensor):
            result = result.item()
        return result

def MCC_SCORE(pred, target):
    TP = torch.sum((pred == 1) & (target == 1)).float()
    TN = torch.sum((pred == 0) & (target == 0)).float()
    FP = torch.sum((pred == 1) & (target == 0)).float()
    FN = torch.sum((pred == 0) & (target == 1)).float()

    denominator = ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5
    if denominator == 0:
        return float('nan')
    else:
        result = (TP*TN-FP*FN) / denominator
        if isinstance(result, torch.Tensor):
            result = result.item()
        return result

def AUC_SCORE(likelihood, target):
    likelihood = likelihood.flatten().cpu().numpy()
    target = target.flatten().cpu().numpy()

    try:
        auc_score = roc_auc_score(target, likelihood)
    except:
        auc_score = 0
    return auc_score

def dice_score(pred_mask, true_mask):
    intersection = torch.sum(true_mask * pred_mask)
    union = torch.sum(true_mask) + torch.sum(pred_mask)
    dice = (2. * intersection + 1e-6) / (union + 1e-6) # adding a small epsilon to avoid division by zero
    return dice.item()

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)

def clDice_ins(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    # elif len(v_p.shape)==3:
    #     tprec = cl_score(v_p,skeletonize_3d(v_l))
    #     tsens = cl_score(v_l,skeletonize_3d(v_p))

    if tprec == 0 and tsens == 0:
        return 0.0
    elif math.isnan(tsens) and math.isnan(tprec):
        return 1.0
    elif not math.isnan(tsens) and not math.isnan(tprec):
        return 2 * tprec * tsens / (tprec + tsens)
    else:
        return 0.0

def clDice_score(x, y):
    assert x.size() == y.size()
    N, C, H, W = x.size()
    assert C == 1

    x = x.squeeze(1)
    y = y.squeeze(1)

    x = x.cpu().numpy()
    y = y.cpu().numpy()

    result = []

    for i in range(N):
        score = clDice_ins(x[i], y[i])
        if score == np.nan:
            score = 1.0
        result.append(score)

    return result[0]

def getBettiNumber(pred, target):
    N_o, C_o, H_o, W_o = target.size()
    assert N_o == 1

    pred_o = F.interpolate(pred, size=(H_o, W_o), mode='bilinear')

    # binarize
    pred_o[pred_o < 0.5] = 0
    pred_o[pred_o >= 0.5] = 1

    # pad to square
    if H_o != W_o:
        margin = abs(H_o - W_o)
        pad1, pad2 = margin // 2, margin - margin // 2

        if H_o > W_o:
            paddings = (pad1, pad2, 0, 0)
        else:
            paddings = (0, 0, pad1, pad2)

        if pred_o is not None:
            pred_o = F.pad(pred_o, paddings, "constant", 0.0)
        if target is not None:
            target = F.pad(target, paddings, "constant", 0.0)

    # Invert color and clamp
    pred_o = 1 - pred_o
    target = 1 - target
    pred_o = torch.clamp(pred_o, 0, 1)
    target = torch.clamp(target, 0, 1)

    p_pred = getPersistentInfo(pred_o)
    p_tgt = getPersistentInfo(target)

    Betti_p_0 = len(p_pred[0][0][0].diagram)
    Betti_p_1 = len(p_pred[0][0][1].diagram)
    Betti_t_0 = len(p_tgt[0][0][0].diagram)
    Betti_t_1 = len(p_tgt[0][0][1].diagram)

    return Betti_p_0, Betti_p_1, Betti_t_0, Betti_t_1