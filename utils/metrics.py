import numpy as np
from typing import Dict, List, Optional, Union
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
    brier_score_loss,
    roc_curve,
    cohen_kappa_score,
)


def calculate_fast_metrics(
    y_true: Union[List[int], np.ndarray],
    y_probs: Union[List[float], np.ndarray],
    fixed_thresh: Optional[float] = None,
) -> Dict[str, float]:
    """
    快速计算二分类指标，不带置信区间，适合训练中每个 epoch 打印。
    如果 fixed_thresh 为 None，则基于 Youden Index 动态寻找最佳阈值。

    Args:
        y_true (Union[List[int], np.ndarray]): 真实标签列表或数组。
        y_probs (Union[List[float], np.ndarray]): 模型预测的正类概率。
        fixed_thresh (Optional[float]): 固定的分类阈值，若为 None 则动态计算。

    Returns:
        Dict[str, float]: 包含 AUC, PR-AUC, Sens, Spec, Thresh 的字典。
    """
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    if len(np.unique(y_true)) < 2:
        return {
            "AUC": np.nan,
            "PR-AUC": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
            "ACC": np.nan,
            "Micro-ACC": np.nan,
            "NPV": np.nan,
            "Kappa": np.nan,
            "Sens": np.nan,
            "Spec": np.nan,
            "Thresh": 0.5,
        }

    try:
        auc_val = float(roc_auc_score(y_true, y_probs))
    except ValueError:
        auc_val = np.nan

    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = float(auc(recall, precision))

    if fixed_thresh is None:
        fpr, tpr, thresh = roc_curve(y_true, y_probs)
        finite_mask = np.isfinite(thresh)
        if finite_mask.sum() > 0:
            clean_thresh = thresh[finite_mask]
            clean_tpr = tpr[finite_mask]
            clean_fpr = fpr[finite_mask]
            best_idx = np.argmax(clean_tpr - clean_fpr)
            best_thresh = clean_thresh[best_idx]
        else:
            best_thresh = 0.5
    else:
        best_thresh = fixed_thresh

    y_pred = (y_probs >= best_thresh).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = float(tp / (tp + fn + 1e-8))
    spec = float(tn / (tn + fp + 1e-8))
    precision_val = float(tp / (tp + fp + 1e-8))
    recall_val = sens
    f1_val = float(
        2.0 * precision_val * recall_val / (precision_val + recall_val + 1e-8)
    )
    acc_val = float((tp + tn) / (tp + tn + fp + fn + 1e-8))
    npv_val = float(tn / (tn + fn + 1e-8))
    kappa_val = float(cohen_kappa_score(y_true, y_pred))

    return {
        "AUC": auc_val,
        "PR-AUC": pr_auc,
        "Precision": precision_val,
        "Recall": recall_val,
        "F1": f1_val,
        "ACC": acc_val,
        "Micro-ACC": acc_val,
        "NPV": npv_val,
        "Kappa": kappa_val,
        "Sens": sens,
        "Spec": spec,
        "Thresh": float(best_thresh),
    }


def binary_metrics_from_probs(
    y_true: Union[List[int], np.ndarray],
    y_prob: Union[List[float], np.ndarray],
    threshold: float,
) -> Dict[str, float]:
    """
    计算详尽的二分类评估指标，适合测试与外部验证阶段使用。

    Args:
        y_true: 真实标签列表或数组。
        y_prob: 模型预测的正类概率。
        threshold: 固定的分类判断阈值。

    Returns:
        Dict[str, float]: 包含 auc, pr_auc, accuracy, sensitivity, specificity, ppv, npv, brier_score 的字典。
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    try:
        auc_val = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc_val = 0.5

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(recall, precision))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sens = float(tp / (tp + fn + 1e-8))
    spec = float(tn / (tn + fp + 1e-8))
    acc = float((tp + tn) / len(y_true))
    ppv = float(tp / (tp + fp + 1e-8))
    npv = float(tn / (tn + fn + 1e-8))
    f1 = float(2.0 * ppv * sens / (ppv + sens + 1e-8))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "auc": auc_val,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "micro_accuracy": acc,
        "precision": ppv,
        "recall": sens,
        "f1": f1,
        "kappa": kappa,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "brier_score": brier,
    }


def bootstrap_auc_ci(
    y_true: Union[List[int], np.ndarray],
    y_prob: Union[List[float], np.ndarray],
    seed: int = 42,
    n_boot: int = 1000,
) -> List[Optional[float]]:
    """
    计算 AUC 的 Bootstrap 95% 置信区间

    Args:
        y_true: 真实标签列表或数组。
        y_prob: 模型预测的正类概率。
        seed: 随机种子，保证可复现性。
        n_boot: 采样次数，默认 1000。

    Returns:
        List[Optional[float]]: 包含置信区间下限和上限的列表，若无法计算则为 [None, None]。
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if not scores:
        return [None, None]
    return [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))]


def bootstrap_metric_cis(y_true, y_prob, threshold, seed=42, n_boot=1000):
    """
    计算各项二分类指标的 Bootstrap 95% 置信区间
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rng = np.random.default_rng(seed)
    metric_keys = [
        "auc",
        "pr_auc",
        "accuracy",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "brier_score",
    ]
    values = {key: [] for key in metric_keys}
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample_y = y_true[idx]
        sample_p = y_prob[idx]
        if len(np.unique(sample_y)) < 2:
            continue
        metrics = binary_metrics_from_probs(sample_y, sample_p, threshold)
        for key in metric_keys:
            values[key].append(metrics[key])
    ci = {}
    for key in metric_keys:
        if values[key]:
            ci[key] = [
                float(np.percentile(values[key], 2.5)),
                float(np.percentile(values[key], 97.5)),
            ]
        else:
            ci[key] = [None, None]
    return ci


def format_metrics(metrics):
    return {k: f"{v:.3f}" for k, v in metrics.items()}
