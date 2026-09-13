import os
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from scipy.ndimage import binary_dilation
from scipy.stats import ttest_ind
from tqdm import tqdm

from models.builder import build_model
from datasets.mri_dataset import MRIDataset
from utils.calibration import expected_calibration_error, reliability_bin_stats
from utils.config_manager import ConfigManager
from utils.explain_selection import load_cv_threshold, select_explain_records
from utils.metadata import load_metadata_records


def plot_reliability_diagram(
    y_true, y_prob, out_path, n_bins=10, title="Reliability Diagram"
):
    empirical_rates, confidence_means, counts = reliability_bin_stats(
        y_true, y_prob, n_bins=n_bins
    )
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.scatter(
        confidence_means,
        empirical_rates,
        s=np.array(counts, dtype=float) * 8.0 + 10.0,
        alpha=0.8,
    )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Empirical Positive Rate")
    plt.title(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def integrated_gradients_3d(
    model,
    images,
    kinetics=None,
    lesion_mask=None,
    clinical=None,
    target_class=1,
    steps=16,
):
    model.eval()
    baseline = torch.zeros_like(images)
    scaled_inputs = [
        baseline + (float(i) / steps) * (images - baseline) for i in range(1, steps + 1)
    ]
    total_grads = torch.zeros_like(images)
    for x in scaled_inputs:
        x = x.clone().detach().requires_grad_(True)
        outputs = model(
            x,
            kinetics=kinetics,
            lesion_mask=lesion_mask,
            clinical=clinical,
        )
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        score = logits[:, target_class].sum()
        grads = torch.autograd.grad(score, x, retain_graph=False, create_graph=False)[0]
        total_grads += grads.detach()
    avg_grads = total_grads / float(steps)
    attributions = (images - baseline) * avg_grads
    return attributions


def plot_gate_attention_maps(img_phase, core_map, peri_map, out_path, title_prefix="Gate"):
    img_np = img_phase.detach().cpu().numpy()
    core_np = core_map.detach().cpu().numpy()
    peri_np = peri_map.detach().cpu().numpy()
    D = img_np.shape[0]
    slice_indices = [D // 4, D // 2, 3 * D // 4]
    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    for i, z in enumerate(slice_indices):
        axes[i, 0].imshow(img_np[z], cmap="gray")
        axes[i, 0].set_title(f"Original Slice {z}")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(img_np[z], cmap="gray")
        axes[i, 1].imshow(core_np[z], cmap="Reds", alpha=0.6)
        axes[i, 1].set_title(f"{title_prefix} Core")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(img_np[z], cmap="gray")
        axes[i, 2].imshow(peri_np[z], cmap="Blues", alpha=0.6)
        axes[i, 2].set_title(f"{title_prefix} Peri")
        axes[i, 2].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _resolve_experiment_base_dir(base_log_dir, seed, experiment_name=None):
    if experiment_name:
        candidate = os.path.join(base_log_dir, experiment_name)
        if os.path.isdir(candidate):
            return candidate
        raise FileNotFoundError(f"Requested experiment directory not found: {candidate}")
    prefix = "InternalCV_"
    suffix = f"_seed_{seed}"
    candidates = []
    if os.path.isdir(base_log_dir):
        for name in os.listdir(base_log_dir):
            full_path = os.path.join(base_log_dir, name)
            if (
                os.path.isdir(full_path)
                and name.startswith(prefix)
                and name.endswith(suffix)
            ):
                candidates.append(full_path)
    if not candidates:
        raise FileNotFoundError(
            f"No InternalCV experiment with seed {seed} was found in '{base_log_dir}'."
        )
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _select_representative_cases(test_data_list, labels, probs, threshold=0.5):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    preds = (probs >= threshold).astype(int)
    case_specs = [
        ("tp", "true_positive", (labels == 1) & (preds == 1)),
        ("tn", "true_negative", (labels == 0) & (preds == 0)),
        ("fp", "false_positive", (labels == 0) & (preds == 1)),
        ("fn", "false_negative", (labels == 1) & (preds == 0)),
    ]
    selected = []
    for short_name, display_name, mask in case_specs:
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        category_probs = probs[indices]
        target_prob = float(np.median(category_probs))
        chosen_local = int(np.argmin(np.abs(category_probs - target_prob)))
        chosen_idx = int(indices[chosen_local])
        selected.append(
            {
                "case_type": short_name,
                "case_label": display_name,
                "dataset_index": chosen_idx,
                "id": str(test_data_list[chosen_idx]["id"]),
                "label": int(labels[chosen_idx]),
                "pred": int(preds[chosen_idx]),
                "prob": float(probs[chosen_idx]),
                "selection_rule": "median_probability_within_case_type",
            }
        )
    if selected:
        return selected
    if len(test_data_list) == 0:
        return []
    fallback_idx = int(np.argmin(np.abs(probs - np.median(probs))))
    return [
        {
            "case_type": "representative",
            "case_label": "representative",
            "dataset_index": fallback_idx,
            "id": str(test_data_list[fallback_idx]["id"]),
            "label": int(labels[fallback_idx]),
            "pred": int(preds[fallback_idx]),
            "prob": float(probs[fallback_idx]),
            "selection_rule": "global_median_probability_fallback",
        }
    ]


def run_explainability():
    """
    主函数：执行完整的可解释性分析流程。
    """
    parser = argparse.ArgumentParser(description="RAPSS-Net/TriDual3D Explainability Engine")
    parser.add_argument(
        "--center", type=str, required=True, help="The external center to be analyzed."
    )
    parser.add_argument(
        "--fold", type=int, required=True, help="Specify which fold's weights to use."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/server/random/train_config_RAPSS_Net.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Specific internal CV experiment directory name.",
    )
    args = parser.parse_args()

    config_mgr = ConfigManager(args.config)
    config = config_mgr.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = config["project"].get("seed", 42)
    try:
        base_dir = _resolve_experiment_base_dir(
            config["paths"]["log_dir"], seed, args.experiment_name
        )
    except FileNotFoundError as e:
        print(f"{e}")
        return
    safe_center_name = str(args.center).replace(os.sep, "_").replace("/", "_")
    out_dir = os.path.join(
        base_dir, f"explainability_{safe_center_name}_fold_{args.fold}"
    )
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"Starting RAPSS-Net/TriDual3D Explainability Engine | Target: {args.center} | Fold: {args.fold} ..."
    )

    all_data = load_metadata_records(
        config["paths"]["metadata_path"], include_ihc=False, require_unique_ids=False
    )
    internal_centers = {
        str(center) for center in config.get("train", {}).get("internal_centers", [])
    }
    strict_fold_match = str(args.center) in internal_centers
    try:
        test_data_list, selection_mode = select_explain_records(
            all_data,
            args.center,
            base_dir,
            args.fold,
            strict_fold_match=strict_fold_match,
        )
    except ValueError as e:
        print(f"{e}")
        return
    if not test_data_list:
        print(f"No data found for center {args.center}. Exiting.")
        return
    explain_threshold, threshold_source = load_cv_threshold(base_dir)
    print(
        f"Explainability selection mode: {selection_mode} | samples={len(test_data_list)}"
    )
    print(
        f"Explainability case threshold: {explain_threshold:.4f} ({threshold_source})"
    )

    clinical_keys = [str(key) for key in config.get("data", {}).get("clinical_keys", [])]
    clinical_stats = config.get("data", {}).get("clinical_stats", {}) or {}
    if (
        bool(config.get("model", {}).get("use_clinical", False))
        and clinical_keys
        and not clinical_stats
    ):
        raise ValueError(
            "Explainability with clinical inputs requires data.clinical_stats "
            "from the training split."
        )

    dataset = MRIDataset(
        test_data_list,
        config["paths"]["data_dir"],
        model_config=config.get("model", {}),
        is_train=False,
        use_mask=True,
        strict_file_check=True,
        clinical_keys=clinical_keys,
        clinical_stats=clinical_stats,
        clinical_missing_indicator=config.get("data", {}).get(
            "clinical_missing_indicator", True
        ),
    )

    model = build_model(config, device)

    ckpt_path = os.path.join(
        base_dir, f"fold_{args.fold}", f"best_model_fold_{args.fold}.pth"
    )
    if not os.path.exists(ckpt_path):
        print(f"Weight file not found: {ckpt_path}. Please run training first.")
        return

    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(
        checkpoint.get("ema_model_state_dict", checkpoint["model_state_dict"])
    )
    model.eval()

    all_fused_feats, all_labels, all_probs, core_norms, peri_norms = [], [], [], [], []
    valid_indices = list(range(len(dataset)))

    print("⏳ Extracting deep features from imaging data...")
    for i in tqdm(valid_indices, desc="Extracting Features"):
        sample = dataset[i]
        images = sample["image"].unsqueeze(0).to(device)
        kinetics = sample["kinetics"].unsqueeze(0).to(device)
        clinical = sample.get("clinical")
        clinical = (
            clinical.unsqueeze(0).to(device)
            if clinical is not None and clinical.nelement() > 0
            else None
        )
        labels = sample["label"]

        with torch.no_grad():
            outputs = model(
                images,
                kinetics=kinetics,
                lesion_mask=sample["mask"].unsqueeze(0).to(device),
                clinical=clinical,
            )
            out_pcr = outputs[0] if isinstance(outputs, tuple) else outputs
            f_core = outputs[2] if isinstance(outputs, tuple) and len(outputs) > 2 else None
            f_peri = outputs[3] if isinstance(outputs, tuple) and len(outputs) > 3 else None
            global_feat = outputs[4] if isinstance(outputs, tuple) and len(outputs) > 4 else None
            prob = F.softmax(out_pcr, dim=1)[:, 1].item()
            if global_feat is not None:
                img_f = global_feat
            elif getattr(model, "use_peritumor", False) and f_core is not None and f_peri is not None:
                img_f = torch.cat([f_core, f_peri], dim=-1)
            else:
                img_f = f_core

            all_fused_feats.append(img_f.cpu().numpy().flatten())
            all_labels.append(labels.item())
            all_probs.append(prob)
            core_norms.append(torch.norm(f_core).item() if f_core is not None else 0.0)
            peri_norms.append(torch.norm(f_peri).item() if f_peri is not None else 0.0)

            del images, outputs, out_pcr, f_core, f_peri, global_feat, prob, img_f

    # 在循环后调用 empty_cache 以释放缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_fused_feats, all_labels = np.array(all_fused_feats), np.array(all_labels)

    # --- 1. t-SNE Visualization ---
    print("🎨 [1/4] Plotting t-SNE latent space...")
    if len(all_labels) > 5:
        tsne = TSNE(
            n_components=2,
            random_state=seed,
            perplexity=max(1, min(30, len(all_labels) - 2)),
        )
        proj = tsne.fit_transform(all_fused_feats)
        plt.figure(figsize=(9, 7))
        sns.kdeplot(
            x=proj[all_labels == 0, 0],
            y=proj[all_labels == 0, 1],
            cmap="Blues",
            fill=True,
            alpha=0.3,
        )
        sns.kdeplot(
            x=proj[all_labels == 1, 0],
            y=proj[all_labels == 1, 1],
            cmap="Reds",
            fill=True,
            alpha=0.3,
        )
        sns.scatterplot(
            x=proj[:, 0],
            y=proj[:, 1],
            hue=["pCR" if lbl == 1 else "Non-pCR" for lbl in all_labels],
            palette={"pCR": "#d62728", "Non-pCR": "#1f77b4"},
            s=80,
            edgecolor="white",
            alpha=0.8,
        )
        plt.title("Imaging-Exclusive Latent Space", fontweight="bold")
        plt.savefig(os.path.join(out_dir, "KDE_tSNE_Manifold.png"), dpi=300)
        plt.close()

    # --- 2. Core/Peri Activation Analysis ---
    print("🎨 [2/4] Analyzing core vs. peri-tumoral activation...")
    if getattr(model, "use_peritumor", False):
        ratios = np.array(peri_norms) / (
            np.array(core_norms) + np.array(peri_norms) + 1e-8
        )
        pcr_ratios, non_pcr_ratios = ratios[all_labels == 1], ratios[all_labels == 0]
        if len(pcr_ratios) > 1 and len(non_pcr_ratios) > 1:
            t_stat, p_value = ttest_ind(pcr_ratios, non_pcr_ratios, equal_var=False)
            p_title = f"(Welch's t-test: p = {p_value:.4e})"
        else:
            p_title = "(Not enough samples for t-test)"
        df_r = pd.DataFrame(
            {
                "Ratio": ratios,
                "Label": ["pCR" if lbl == 1 else "Non-pCR" for lbl in all_labels],
            }
        )
        plt.figure(figsize=(7, 6))
        sns.violinplot(
            x="Label",
            y="Ratio",
            data=df_r,
            palette={"pCR": "#ff9999", "Non-pCR": "#99ccff"},
        )
        plt.title(f"Core/Peri Activation Balance\n{p_title}", fontweight="bold")
        plt.ylabel("Relative Activation Ratio (Peri-Stream)")
        plt.savefig(os.path.join(out_dir, "Core_Peri_Activation_Ratio.png"), dpi=300)
        plt.close()

    print("🎨 [3/4] Selecting representative cases with a deterministic rule...")
    selected_cases = _select_representative_cases(
        test_data_list, all_labels, all_probs, threshold=explain_threshold
    )
    pd.DataFrame(selected_cases).to_csv(
        os.path.join(out_dir, "selected_explainability_cases.csv"),
        index=False,
        encoding="utf-8",
    )

    # --- 4. Calibration, gate maps & Counterfactual Evidence ---
    print("🎨 [4/4] Calibration, gate maps and counterfactual evidence analysis...")
    if len(all_labels) > 5:
        ece = expected_calibration_error(all_labels, all_probs, n_bins=10)
        plot_reliability_diagram(
            all_labels,
            all_probs,
            out_path=os.path.join(out_dir, "Reliability_Diagram.png"),
            n_bins=10,
            title=f"Reliability Diagram (ECE={ece:.4f})",
        )
        df_cal = pd.DataFrame(
            {"label": all_labels.astype(int), "prob": np.array(all_probs, dtype=float)}
        )
        df_cal.to_csv(
            os.path.join(out_dir, "calibration_table.csv"),
            index=False,
            encoding="utf-8",
        )

        for case_info in selected_cases:
            vis_idx = int(case_info["dataset_index"])
            sample = dataset[vis_idx]
            img = sample["image"]
            kinetics = sample["kinetics"]
            clinical = sample.get("clinical")
            mask_gt = sample["mask"]
            label_gt = sample["label"]
            images = img.unsqueeze(0).to(device)
            kinetics_batch = kinetics.unsqueeze(0).to(device)
            clinical_batch = (
                clinical.unsqueeze(0).to(device)
                if clinical is not None and clinical.nelement() > 0
                else None
            )
            case_dir = os.path.join(out_dir, case_info["case_type"])
            os.makedirs(case_dir, exist_ok=True)

            print("🔬 Running quantitative biological validation for Peri-Gate...")
            mask_t = mask_gt.float()
            if mask_t.dim() == 4 and mask_t.size(0) == 1:
                mask_t = mask_t.squeeze(0)
            if mask_t.dim() != 3:
                mask_t = mask_t.view(img.shape[-3], img.shape[-2], img.shape[-1])

            mask_binary = (mask_t > 0.5).cpu().numpy()
            dilated_mask = binary_dilation(mask_binary, iterations=3)
            true_peri_mask = (dilated_mask ^ mask_binary).astype(np.float32)
            true_peri_tensor = torch.from_numpy(true_peri_mask).to(device)
            mask_t = (mask_t > 0.5).to(device=device)

            learned_peri_gate = None

            def hook_fn(module, input, output):
                nonlocal learned_peri_gate
                if isinstance(output, tuple) and len(output) == 2:
                    learned_peri_gate = output[1].detach()
                elif isinstance(output, torch.Tensor):
                    learned_peri_gate = torch.sigmoid(output.detach())

            hook_handle = None
            for name, module in model.named_modules():
                lname = name.lower()
                if "peri_gate" in lname and "phase" not in lname:
                    hook_handle = module.register_forward_hook(hook_fn)
                    break
                if "phase_dual_gate" in lname or "dual_gate" in lname:
                    hook_handle = module.register_forward_hook(hook_fn)
                    break

            with torch.no_grad():
                o0 = model(
                    images,
                    kinetics=kinetics_batch,
                    lesion_mask=mask_gt.unsqueeze(0).to(device),
                    clinical=clinical_batch,
                    return_attention_maps=True,
                )
                if (
                    isinstance(o0, tuple)
                    and len(o0) > 5
                    and isinstance(o0[5], dict)
                    and "peri" in o0[5]
                ):
                    learned_peri_gate = o0[5]["peri"].detach()

            if hook_handle:
                hook_handle.remove()

            if learned_peri_gate is not None:
                if learned_peri_gate.dim() < 3:
                    learned_peri_gate = None
            if learned_peri_gate is not None:
                batch_size = int(images.shape[0])
                phase_count = int(images.shape[1]) if images.dim() > 1 else 1
                if learned_peri_gate.dim() == 5:
                    if learned_peri_gate.shape[0] == batch_size * phase_count:
                        learned_peri_gate = learned_peri_gate.view(
                            batch_size, phase_count, *learned_peri_gate.shape[1:]
                        )[:, -1]
                    learned_peri_gate = learned_peri_gate.squeeze(0).squeeze(0)
                if learned_peri_gate.shape[-3:] != true_peri_tensor.shape:
                    learned_peri_gate = F.interpolate(
                        learned_peri_gate.unsqueeze(0).unsqueeze(0),
                        size=true_peri_tensor.shape,
                        mode="trilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)

                gate_norm = learned_peri_gate.squeeze()
                gate_norm = (gate_norm - gate_norm.min()) / (gate_norm.max() - gate_norm.min() + 1e-8)

                threshold = torch.quantile(gate_norm.view(-1), 0.8)
                binary_learned_peri = (gate_norm > threshold).float()

                intersection = (binary_learned_peri * true_peri_tensor).sum()
                union = binary_learned_peri.sum() + true_peri_tensor.sum()

                dice = (2. * intersection) / (union + 1e-8)
                iou = intersection / (union - intersection + 1e-8)

                print(f"✅ Biological Validation | Learned Peri-Gate vs True Dilation (3mm): Dice = {dice.item():.4f}, IoU = {iou.item():.4f}")

                with open(os.path.join(case_dir, "peri_gate_validation.txt"), "w") as f:
                    f.write("Biological Validation of Peri-Gate Mechanism\n")
                    f.write(f"Sample ID: {test_data_list[vis_idx]['id']}\n")
                    f.write(f"Case Type: {case_info['case_label']}\n")
                    f.write("Dilation Iterations: 3 (approx. 3mm)\n")
                    f.write(f"Dice Coefficient: {dice.item():.4f}\n")
                    f.write(f"Intersection over Union (IoU): {iou.item():.4f}\n")
                    f.write(f"\nConclusion: Quantitative analysis shows that the spontaneously learned peri_gate has a {dice.item()*100:.1f}% overlap with the radiologically defined 3mm tumor microenvironment.\n")

            peri = torch.from_numpy(
                binary_dilation(mask_t.detach().cpu().numpy(), iterations=3)
            ).to(device=device)
            peri = (peri.float() - mask_t.float()).clamp_(0.0, 1.0)
            core = mask_t.float()
            core_bc = core.unsqueeze(0).unsqueeze(0)
            peri_bc = peri.unsqueeze(0).unsqueeze(0)
            core_only = images * core_bc
            peri_only = images * peri_bc
            with torch.no_grad():
                lesion_mask_batch = mask_gt.unsqueeze(0).to(device)
                o0 = model(
                    images,
                    kinetics=kinetics_batch,
                    lesion_mask=lesion_mask_batch,
                    clinical=clinical_batch,
                )
                o1 = model(
                    core_only,
                    kinetics=kinetics_batch,
                    lesion_mask=lesion_mask_batch,
                    clinical=clinical_batch,
                )
                o2 = model(
                    peri_only,
                    kinetics=kinetics_batch,
                    lesion_mask=lesion_mask_batch,
                    clinical=clinical_batch,
                )
                p0 = (
                    float(F.softmax(o0[0], dim=1)[:, 1].item())
                    if isinstance(o0, tuple)
                    else float(F.softmax(o0, dim=1)[:, 1].item())
                )
                p1 = (
                    float(F.softmax(o1[0], dim=1)[:, 1].item())
                    if isinstance(o1, tuple)
                    else float(F.softmax(o1, dim=1)[:, 1].item())
                )
                p2 = (
                    float(F.softmax(o2[0], dim=1)[:, 1].item())
                    if isinstance(o2, tuple)
                    else float(F.softmax(o2, dim=1)[:, 1].item())
                )
            df_cf = pd.DataFrame(
                [
                    {
                        "id": str(test_data_list[vis_idx]["id"]),
                        "case_type": case_info["case_label"],
                        "label": int(
                            label_gt.item()
                            if isinstance(label_gt, torch.Tensor)
                            else label_gt
                        ),
                        "prob_original": p0,
                        "prob_core_only": p1,
                        "prob_peri_only": p2,
                        "delta_core_only": p1 - p0,
                        "delta_peri_only": p2 - p0,
                    }
                ]
            )
            df_cf.to_csv(
                os.path.join(case_dir, "counterfactual_core_peri.csv"),
                index=False,
                encoding="utf-8",
            )
            plt.figure(figsize=(7, 5))
            sns.barplot(
                x=["Original", "Core only", "Peri only"],
                y=[p0, p1, p2],
                palette=["#444444", "#d62728", "#1f77b4"],
            )
            plt.ylim(0.0, 1.0)
            plt.title(
                f"Counterfactual Evidence Decomposition ({case_info['case_label']})",
                fontweight="bold",
            )
            plt.ylabel("Predicted pCR Probability")
            plt.tight_layout()
            plt.savefig(
                os.path.join(case_dir, "Counterfactual_Evidence_Bar.png"), dpi=300
            )
            plt.close()

            try:
                with torch.no_grad():
                    o_gate = model(
                        images,
                        kinetics=kinetics_batch,
                        lesion_mask=mask_gt.unsqueeze(0).to(device),
                        clinical=clinical_batch,
                        return_attention_maps=True,
                    )
                    gate_maps = (
                        o_gate[-1] if isinstance(o_gate, tuple) and len(o_gate) >= 6 else None
                    )
                if gate_maps is not None:
                    core_gate_map = gate_maps["core"][0, 0]
                    peri_gate_map = gate_maps["peri"][0, 0]
                    late_phase_img = img[-1]
                    plot_gate_attention_maps(
                        late_phase_img,
                        core_gate_map,
                        peri_gate_map,
                        os.path.join(case_dir, "Core_Peri_Gate_Attention_LatePhase.png"),
                        title_prefix="Late Phase",
                    )
            except Exception:
                pass

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                attributions = integrated_gradients_3d(
                    model,
                    images,
                    kinetics=kinetics_batch,
                    lesion_mask=mask_gt.unsqueeze(0).to(device),
                    clinical=clinical_batch,
                    target_class=1,
                    steps=12,
                )
                attr_map = attributions.abs().sum(dim=1).squeeze(0)
                attr_map = (attr_map - attr_map.min()) / (
                    attr_map.max() - attr_map.min() + 1e-8
                )
                orig = img[0].cpu().numpy()
                D, H, W = orig.shape
                attr_np = attr_map.detach().cpu().numpy()
                fig, axes = plt.subplots(3, 2, figsize=(12, 16))
                slice_indices = [D // 4, D // 2, 3 * D // 4]
                for i, z in enumerate(slice_indices):
                    axes[i, 0].imshow(orig[z], cmap="gray")
                    axes[i, 0].set_title(f"Slice {z} (Original)")
                    axes[i, 0].axis("off")
                    axes[i, 1].imshow(orig[z], cmap="gray")
                    axes[i, 1].imshow(attr_np[z], cmap="inferno", alpha=0.6)
                    axes[i, 1].set_title("Integrated Gradients (|attr|)")
                    axes[i, 1].axis("off")
                plt.suptitle(
                    f"Input Attribution: Patient {test_data_list[vis_idx]['id']} | p={p0:.3f}",
                    fontsize=16,
                    y=0.98,
                )
                plt.tight_layout()
                plt.savefig(
                    os.path.join(case_dir, "Integrated_Gradients_Attribution.png"),
                    dpi=300,
                )
                plt.close()
            except Exception:
                pass

    print(f"✅ Explainability analysis complete! Results saved in {out_dir}")


if __name__ == "__main__":
    run_explainability()
