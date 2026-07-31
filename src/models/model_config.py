"""Model configuration helpers shared by training and model loading."""

from omegaconf import DictConfig, OmegaConf


def _resolve_expected_image_channels(cfg: DictConfig) -> int:
    """Resolve expected model image channels from the active data contract."""
    loader_mode = OmegaConf.select(cfg, "data_mode.loader_mode")
    num_modalities = OmegaConf.select(cfg, "dataset.num_modalities")
    if num_modalities is None:
        modalities = OmegaConf.select(cfg, "dataset.modalities")
        if modalities is None:
            raise ValueError(
                "Missing dataset channel contract: expected dataset.num_modalities "
                "or dataset.modalities."
            )
        num_modalities = len(modalities)
    num_modalities = int(num_modalities)

    if loader_mode == "nnunet_slices_2d":
        per_side_context_slices = int(
            OmegaConf.select(cfg, "data_mode.per_side_context_slices", default=0) or 0
        )
        num_effective_slices = (2 * per_side_context_slices) + 1
        return num_modalities * num_effective_slices

    return num_modalities


def sync_model_image_channels_with_data_contract(cfg: DictConfig) -> int:
    """Auto-sync model.image_channels from the active data contract."""
    expected_channels = _resolve_expected_image_channels(cfg)
    current_channels = OmegaConf.select(cfg, "model.image_channels")
    previous = None if current_channels is None else int(current_channels)

    if previous != expected_channels:
        OmegaConf.set_struct(cfg, False)
        OmegaConf.update(
            cfg,
            "model.image_channels",
            int(expected_channels),
            merge=False,
        )
        OmegaConf.set_struct(cfg, True)
        if previous is None:
            print(
                "[Data Contract] Set model.image_channels from data contract: "
                f"{expected_channels}."
            )
        else:
            print(
                "[Data Contract] Updated model.image_channels to match data contract: "
                f"{previous} -> {expected_channels}."
            )

    return int(expected_channels)


def validate_model_channel_contract(cfg: DictConfig) -> None:
    """Fail fast when model input channels disagree with data contract."""
    expected_channels = _resolve_expected_image_channels(cfg)
    configured_channels = int(OmegaConf.select(cfg, "model.image_channels"))
    if configured_channels != expected_channels:
        loader_mode = OmegaConf.select(cfg, "data_mode.loader_mode")
        per_side_context_slices = int(
            OmegaConf.select(cfg, "data_mode.per_side_context_slices", default=0) or 0
        )
        raise ValueError(
            "Model image channel contract mismatch. "
            f"Expected model.image_channels={expected_channels} from data contract "
            f"(loader_mode={loader_mode}, "
            f"per_side_context_slices={per_side_context_slices}), "
            f"got model.image_channels={configured_channels}."
        )
