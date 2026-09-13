# Experiment configurations

All checked-in paths are repository-relative placeholders. Update the cohort names and local paths for data you are authorized to use.

```text
prep_config_public_server.yaml
random/train_config_Proposed_3D_Base.yaml
random/train_config_RAPSS_Net.yaml
cv/train_config_Proposed_3D_Base.yaml
cv/train_config_RAPSS_Net.yaml
```

The Base configuration removes the region-aware Mamba components for comparison. The RAPSS-Net configuration enables kinetics input, an explicit peritumoral ring, dynamic tri-axial scanning, cross-region interaction, and multiscale aggregation.

```bash
python pre/prepare_all_to_npy_mask.py -c configs/server/prep_config_public_server.yaml
python run_random_split.py -c configs/server/random/train_config_RAPSS_Net.yaml
python run_internal_cv.py -c configs/server/cv/train_config_RAPSS_Net.yaml
```
