"""Descriptor utilities for masks and segmentation outputs."""

from ore_detection.descriptors.intergrowth import (
    summarize_intergrowth_prediction,
    summarize_prediction_artifacts,
    write_descriptor_csv,
)
from ore_detection.descriptors.intergrowth_classification import (
    IntergrowthClassifierConfig,
    classify_intergrowth_mask,
    extract_intergrowth_features,
    save_intergrowth_artifacts,
)
from ore_detection.descriptors.erosion_ratio import (
    ErosionRatioConfig,
    classify_erosion_ratio_intergrowth,
    erosion_ratio_score_map,
)

__all__ = [
    "ErosionRatioConfig",
    "IntergrowthClassifierConfig",
    "classify_erosion_ratio_intergrowth",
    "classify_intergrowth_mask",
    "erosion_ratio_score_map",
    "extract_intergrowth_features",
    "save_intergrowth_artifacts",
    "summarize_intergrowth_prediction",
    "summarize_prediction_artifacts",
    "write_descriptor_csv",
]
