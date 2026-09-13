import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from utils.calibration import expected_calibration_error


class ClinicalEvaluator:
    """Generate threshold-compliant clinical evaluation metrics and plots."""

    def __init__(
        self,
        y_true,
        y_probs,
        save_dir=None,
        seed=2026,
        fixed_threshold=None,
        allow_posthoc=False,
    ):
        self.y_true = np.array(y_true)
        self.y_probs = np.array(y_probs)
        self.save_dir = save_dir
        self.seed = seed

        if fixed_threshold is not None:
            self.best_thresh = fixed_threshold
            self.threshold_source = "fixed (from CV)"
        elif allow_posthoc:
            # Post-hoc thresholding is only allowed for explicitly exploratory analysis.
            fpr, tpr, thresholds = roc_curve(self.y_true, self.y_probs)
            valid_mask = np.isfinite(thresholds)
            if valid_mask.sum() > 0:
                youden_idx = np.argmax(tpr[valid_mask] - fpr[valid_mask])
                self.best_thresh = thresholds[valid_mask][youden_idx]
            else:
                self.best_thresh = 0.5
            self.threshold_source = "post-hoc (on test set)"
        else:
            raise ValueError(
                "A 'fixed_threshold' must be provided for compliant evaluation. "
                "To override for exploratory analysis, explicitly set 'allow_posthoc=True'."
            )

    def _bootstrap_metric(self, metric_func, n_bootstraps=1000):
        """Estimate a metric distribution with bootstrap resampling."""
        values = []
        rng = np.random.RandomState(self.seed)
        for _ in range(n_bootstraps):
            indices = rng.choice(len(self.y_true), len(self.y_true), replace=True)
            if len(np.unique(self.y_true[indices])) < 2:
                continue
            values.append(metric_func(self.y_true[indices], self.y_probs[indices]))
        if not values:
            return 0.0, 0.0, 0.0
        return np.mean(values), np.percentile(values, 2.5), np.percentile(values, 97.5)

    def _calc_pr_auc(self, y_t, y_p):
        precision, recall, _ = precision_recall_curve(y_t, y_p)
        return auc(recall, precision)

    def _calc_sens(self, y_t, y_p):
        y_pred = (y_p >= self.best_thresh).astype(int)
        _, _, fn, tp = confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()
        return tp / (tp + fn + 1e-8)

    def _calc_spec(self, y_t, y_p):
        y_pred = (y_p >= self.best_thresh).astype(int)
        tn, fp, _, _ = confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()
        return tn / (tn + fp + 1e-8)

    def _calc_ppv(self, y_t, y_p):
        y_pred = (y_p >= self.best_thresh).astype(int)
        _, fp, _, tp = confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()
        return tp / (tp + fp + 1e-8)

    def _calc_npv(self, y_t, y_p):
        y_pred = (y_p >= self.best_thresh).astype(int)
        tn, _, fn, _ = confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()
        return tn / (tn + fn + 1e-8)

    def _calc_brier(self, y_t, y_p):
        return brier_score_loss(y_t, y_p)

    def _calc_ece(self, n_bins=10):
        return expected_calibration_error(self.y_true, self.y_probs, n_bins=n_bins)

    def generate_report(self):
        """Build the final clinical report payload."""
        auc_m, auc_l, auc_u = self._bootstrap_metric(roc_auc_score)
        prauc_m, prauc_l, prauc_u = self._bootstrap_metric(self._calc_pr_auc)
        sens_m, sens_l, sens_u = self._bootstrap_metric(self._calc_sens)
        spec_m, spec_l, spec_u = self._bootstrap_metric(self._calc_spec)
        ppv_m, ppv_l, ppv_u = self._bootstrap_metric(self._calc_ppv)
        npv_m, npv_l, npv_u = self._bootstrap_metric(self._calc_npv)
        brier_m, brier_l, brier_u = self._bootstrap_metric(self._calc_brier)

        report = {
            "ROC-AUC": f"{auc_m:.3f} (95% CI: {auc_l:.3f}-{auc_u:.3f})",
            "PR-AUC": f"{prauc_m:.3f} (95% CI: {prauc_l:.3f}-{prauc_u:.3f})",
            "Sensitivity": f"{sens_m:.3f} (95% CI: {sens_l:.3f}-{sens_u:.3f})",
            "Specificity": f"{spec_m:.3f} (95% CI: {spec_l:.3f}-{spec_u:.3f})",
            "PPV": f"{ppv_m:.3f} (95% CI: {ppv_l:.3f}-{ppv_u:.3f})",
            "NPV": f"{npv_m:.3f} (95% CI: {npv_l:.3f}-{npv_u:.3f})",
            "Brier score": f"{brier_m:.3f} (95% CI: {brier_l:.3f}-{brier_u:.3f})",
            "ECE": f"{self._calc_ece():.3f}",
            "Threshold": f"{self.best_thresh:.3f} ({self.threshold_source})",
        }

        if self.save_dir:
            self._plot_calibration_curve()
            self._plot_dca_curve()

        return report

    def _plot_calibration_curve(self):
        prob_true, prob_pred = calibration_curve(self.y_true, self.y_probs, n_bins=10)
        plt.figure(figsize=(6, 6))
        plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label="RAPSS-Net")
        plt.plot(
            [0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated"
        )
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, "Calibration_Curve.png"), dpi=300)
        plt.close()

    def _plot_dca_curve(self):
        thresholds = np.linspace(0.01, 0.99, 100)
        net_benefits = []
        net_benefit_all = (np.sum(self.y_true) / len(self.y_true)) - (
            np.sum(1 - self.y_true) / len(self.y_true)
        ) * (thresholds / (1 - thresholds))

        for thresh in thresholds:
            y_pred_thresh = (self.y_probs >= thresh).astype(int)
            tp = np.sum((y_pred_thresh == 1) & (self.y_true == 1))
            fp = np.sum((y_pred_thresh == 1) & (self.y_true == 0))
            n = len(self.y_true)
            nb = (tp / n) - (fp / n) * (thresh / (1 - thresh))
            net_benefits.append(nb)

        plt.figure(figsize=(6, 6))
        plt.plot(
            thresholds, net_benefits, label="RAPSS-Net", color="red", linewidth=2
        )
        plt.plot(
            thresholds, net_benefit_all, label="Treat All", color="gray", linestyle="--"
        )
        plt.plot(
            thresholds,
            [0] * len(thresholds),
            label="Treat None",
            color="black",
            linestyle=":",
        )
        plt.ylim([-0.1, 0.6])
        plt.xlim([0, 1.0])
        plt.xlabel("Threshold Probability")
        plt.ylabel("Net Benefit")
        plt.title("Decision Curve Analysis (DCA)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, "DCA_Curve.png"), dpi=300)
        plt.close()
