def build_model(config, device):
    """
    Build the paper-facing lesion-centric 3D classifier.
    """
    model_type = config["model"].get("type", "tridual_3d").lower()
    if model_type != "tridual_3d":
        raise ValueError(
            f"Unsupported model type: '{model_type}'. Only 'tridual_3d' is available."
        )

    from .lesion_centric_3d_net import TriDual3D

    model_config = config.get("model", {})
    clinical_keys = config.get("data", {}).get("clinical_keys", [])
    use_clinical = bool(model_config.get("use_clinical", False))
    clinical_input_dim = int(model_config.get("clinical_input_dim", 0))
    if use_clinical and clinical_input_dim <= 0:
        clinical_input_dim = len(clinical_keys) * (
            2 if config.get("data", {}).get("clinical_missing_indicator", True) else 1
        )

    model = TriDual3D(
        in_channels=model_config.get("in_channels", 2),
        base_dim=model_config.get("base_dim", 32),
        spatial_mamba_layers=model_config.get("mamba_layers", 1),
        use_peritumor=model_config.get("use_peritumor", True),
        use_deep_supervision=model_config.get("use_deep_supervision", True),
        d_state=model_config.get("d_state", 16),
        d_conv=model_config.get("d_conv", 4),
        expand=model_config.get("expand", 2),
        use_dynamic_scan_router=model_config.get("use_dynamic_scan_router", True),
        drop_path_rate=model_config.get("drop_path_rate", 0.1),
        layer_scale_init_value=model_config.get("layer_scale_init_value", 1.0e-4),
        use_kinetics_channel=model_config.get("use_kinetics_channel", False),
        kinetics_fusion_scale=model_config.get("kinetics_fusion_scale", 1.0),
        use_feature_recalibration=model_config.get("use_feature_recalibration", False),
        feature_recalibration_reduction=model_config.get(
            "feature_recalibration_reduction", 4
        ),
        feature_recalibration_dropout=model_config.get(
            "feature_recalibration_dropout", 0.0
        ),
        use_explicit_peritumor_ring=model_config.get(
            "use_explicit_peritumor_ring", False
        ),
        peritumor_ring_kernel_size=model_config.get("peritumor_ring_kernel_size", 5),
        use_clinical=use_clinical,
        clinical_input_dim=clinical_input_dim,
        clinical_hidden_dim=model_config.get("clinical_hidden_dim", 32),
        clinical_dropout=model_config.get("clinical_dropout", 0.1),
        use_temporal_order_aux=model_config.get("use_temporal_order_aux", False),
        temporal_order_hidden_dim=model_config.get("temporal_order_hidden_dim", None),
        use_multiscale_feature_aggregation=model_config.get(
            "use_multiscale_feature_aggregation", False
        ),
        multiscale_dropout=model_config.get("multiscale_dropout", 0.0),
    )
    return model.to(device)
