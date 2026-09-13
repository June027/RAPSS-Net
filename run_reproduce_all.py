import argparse
import json
import os
import subprocess
import sys

from utils.config_manager import ConfigManager


def run_cmd(cmd, step_name, cwd):
    print("\n" + "=" * 80)
    print(f"[STAGE] {step_name}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)

    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, cwd=cwd)
    process.wait()

    if process.returncode != 0:
        print(f"\n[FATAL ERROR] {step_name} failed with exit code {process.returncode}.")
        sys.exit(process.returncode)

    print(f"\n[SUCCESS] {step_name} completed successfully.")


def _resolve_path(project_dir, path):
    return path if os.path.isabs(path) else os.path.join(project_dir, path)


def _resolve_latest_internal_cv_dir(base_log_dir, seed, experiment_name=None):
    if experiment_name:
        candidate = os.path.join(base_log_dir, experiment_name)
        if os.path.isdir(candidate) and os.path.exists(
            os.path.join(candidate, "cv_results.json")
        ):
            return candidate
        raise FileNotFoundError(
            f"Internal CV experiment was not found or incomplete: {candidate}"
        )

    suffix = f"_seed_{seed}"
    candidates = []
    if os.path.isdir(base_log_dir):
        for name in os.listdir(base_log_dir):
            full_path = os.path.join(base_log_dir, name)
            if (
                os.path.isdir(full_path)
                and name.startswith(("InternalCV_", "InternalCVFull_"))
                and name.endswith(suffix)
                and os.path.exists(os.path.join(full_path, "cv_results.json"))
            ):
                candidates.append(full_path)
    if not candidates:
        raise FileNotFoundError(
            f"No InternalCV experiment with seed {seed} was found in '{base_log_dir}'."
        )
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _resolve_internal_centers(train_config, cli_centers):
    if cli_centers:
        return [str(center) for center in cli_centers]
    centers = train_config.get("train", {}).get("internal_centers", None)
    if centers:
        return [str(center) for center in centers]
    return []


def _resolve_external_centers(train_config, cli_centers):
    if cli_centers is not None:
        return [str(center) for center in cli_centers if str(center).strip()]
    centers = train_config.get("train", {}).get("external_centers", None)
    if centers:
        return [str(center) for center in centers if str(center).strip()]
    return []


def _resolve_internal_test_size(train_config, cli_test_size):
    if cli_test_size is not None:
        return float(cli_test_size)
    return float(train_config.get("train", {}).get("internal_test_size", 0.2))


def _build_internal_cv_experiment_name(train_config, internal_centers, internal_test_size):
    seed = train_config["project"].get("seed", 2026)
    center_label = "+".join(internal_centers) if internal_centers else "All"
    prefix = "InternalCV" if float(internal_test_size) > 0 else "InternalCVFull"
    return f"{prefix}_({center_label})_seed_{seed}"


def _resolve_explain_folds(base_log_dir, seed, explain_folds, experiment_name):
    if explain_folds:
        return [str(fold) for fold in explain_folds]
    internal_cv_dir = _resolve_latest_internal_cv_dir(
        base_log_dir, seed, experiment_name=experiment_name
    )
    cv_results_path = os.path.join(internal_cv_dir, "cv_results.json")
    with open(cv_results_path, "r", encoding="utf-8") as f:
        cv_info = json.load(f)
    folds = [
        str(fold.get("fold"))
        for fold in cv_info.get("folds", [])
        if fold.get("fold") is not None
    ]
    if not folds:
        raise ValueError(f"No fold records were found in '{cv_results_path}'.")
    return folds


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Server pipeline: preprocessing, random split, internal 5-fold CV, "
            "external validation, external figures, and explainability."
        )
    )
    parser.add_argument(
        "--prep_config",
        type=str,
        default="configs/server/prep_config_public_server.yaml",
        help="Path to the preprocessing config.",
    )
    parser.add_argument(
        "--random_config",
        type=str,
        default="configs/server/random/train_config_RAPSS_Net.yaml",
        help="Path to the server random-split training config.",
    )
    parser.add_argument(
        "--cv_config",
        "--train_config",
        dest="cv_config",
        type=str,
        default="configs/server/cv/train_config_RAPSS_Net.yaml",
        help="Path to the server CV training config. --train_config is kept as an alias.",
    )
    parser.add_argument(
        "--skip_prep", action="store_true", help="Skip preprocessing."
    )
    parser.add_argument(
        "--skip_random_split", action="store_true", help="Skip random-split training."
    )
    parser.add_argument(
        "--skip_cv",
        "--skip_train",
        dest="skip_cv",
        action="store_true",
        help="Skip internal 5-fold CV training. --skip_train is kept as an alias.",
    )
    parser.add_argument(
        "--skip_external", action="store_true", help="Skip external validation."
    )
    parser.add_argument(
        "--skip_external_figures",
        action="store_true",
        help="Skip external validation report figures.",
    )
    parser.add_argument(
        "--skip_explain", action="store_true", help="Skip explainability export."
    )
    parser.add_argument(
        "--external_centers",
        nargs="*",
        default=None,
        help=(
            "External centers for held-out validation. If omitted, "
            "train.external_centers from the CV config is used. Pass with no values to skip."
        ),
    )
    parser.add_argument(
        "--internal_centers",
        nargs="+",
        default=None,
        help="Internal centers to pool for random split and 5-fold CV.",
    )
    parser.add_argument(
        "--internal_test_size",
        type=float,
        default=None,
        help="Internal hold-out ratio before CV. Set to 0 for full internal CV.",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=None,
        help="Validation ratio for the random-split run.",
    )
    parser.add_argument(
        "--explain_folds",
        nargs="+",
        default=None,
        help="Specific CV folds to use for explainability export. Defaults to all folds.",
    )
    args = parser.parse_args()

    py_exec = sys.executable
    project_dir = os.path.dirname(os.path.abspath(__file__))
    cv_config_path = _resolve_path(project_dir, args.cv_config)
    cv_config = ConfigManager(cv_config_path).config
    base_log_dir = cv_config["paths"].get("log_dir", "logs")
    target_seed = cv_config["project"].get("seed", 2026)
    internal_centers = _resolve_internal_centers(cv_config, args.internal_centers)
    external_centers = _resolve_external_centers(cv_config, args.external_centers)
    internal_test_size = _resolve_internal_test_size(cv_config, args.internal_test_size)
    experiment_name = _build_internal_cv_experiment_name(
        cv_config, internal_centers, internal_test_size
    )

    if not args.skip_prep:
        cmd_prep = [py_exec, "pre/prepare_all_to_npy_mask.py", "-c", args.prep_config]
        run_cmd(cmd_prep, "1. Data Preparation", project_dir)
    else:
        print("\nSkipping data preparation.")

    if not args.skip_random_split:
        cmd_random = [py_exec, "run_random_split.py", "-c", args.random_config]
        if args.internal_centers:
            cmd_random.extend(["--internal_centers", *args.internal_centers])
        if args.val_size is not None:
            cmd_random.extend(["--val_size", str(args.val_size)])
        run_cmd(cmd_random, "2. Random-Split Training", project_dir)
    else:
        print("\nSkipping random-split training.")

    if not args.skip_cv:
        cmd_cv = [py_exec, "run_internal_cv.py", "-c", args.cv_config]
        if args.internal_centers:
            cmd_cv.extend(["--internal_centers", *args.internal_centers])
        if args.internal_test_size is not None:
            cmd_cv.extend(["--internal_test_size", str(args.internal_test_size)])
        run_cmd(cmd_cv, "3. Internal Split + 5-Fold Cross-Validation", project_dir)
    else:
        print("\nSkipping internal 5-fold cross-validation.")

    need_cv_artifacts = (
        (external_centers and not args.skip_external)
        or (external_centers and not args.skip_external_figures)
        or not args.skip_explain
    )
    internal_cv_dir = None
    if need_cv_artifacts:
        internal_cv_dir = _resolve_latest_internal_cv_dir(
            base_log_dir, target_seed, experiment_name=experiment_name
        )

    if args.skip_external:
        print("\nSkipping external validation.")
    elif not external_centers:
        print("\nSkipping external validation because no external centers were provided.")
    else:
        for center in external_centers:
            cmd_ext = [
                py_exec,
                "run_external_validation.py",
                "-c",
                args.cv_config,
                "--test_center",
                center,
                "--experiment_name",
                experiment_name,
            ]
            run_cmd(cmd_ext, f"4. External Validation on {center}", project_dir)

    if args.skip_external_figures:
        print("\nSkipping external validation report figures.")
    elif not external_centers:
        print("\nSkipping external validation report figures because no external centers were provided.")
    else:
        for center in external_centers:
            input_dir = os.path.join(internal_cv_dir, f"ExternalTest_{center}")
            cmd_fig = [
                py_exec,
                "plot_external_validation_report.py",
                "--input_dir",
                input_dir,
                "--seed",
                str(target_seed),
            ]
            run_cmd(cmd_fig, f"5. External Report Figures for {center}", project_dir)

    if args.skip_explain:
        print("\nSkipping explainability.")
    else:
        explain_centers = external_centers if external_centers else internal_centers
        if not explain_centers:
            print("\nSkipping explainability because no target centers were resolved.")
        else:
            explain_folds = _resolve_explain_folds(
                base_log_dir, target_seed, args.explain_folds, experiment_name
            )
            for center in explain_centers:
                for fold in explain_folds:
                    cmd_exp = [
                        py_exec,
                        "explain.py",
                        "-c",
                        args.cv_config,
                        "--center",
                        center,
                        "--fold",
                        str(fold),
                        "--experiment_name",
                        experiment_name,
                    ]
                    run_cmd(
                        cmd_exp,
                        f"6. Explainability for {center} (fold {fold})",
                        project_dir,
                    )

    print("\nAll requested pipeline stages completed successfully.")


if __name__ == "__main__":
    main()
