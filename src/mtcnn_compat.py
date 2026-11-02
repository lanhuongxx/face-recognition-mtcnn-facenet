
import os
import importlib.resources

import joblib


def safe_load_weights(weights_name):
    """Load MTCNN weights without LZ4 file-close warnings."""
    paths = [
        os.path.abspath(weights_name),
        importlib.resources.files("mtcnn.assets.weights") / weights_name,
    ]

    for path in paths:
        if os.path.exists(path):
            with open(path, "rb") as file:
                return joblib.load(file)

    raise FileNotFoundError(
        f"Weights file '{weights_name}' not found"
    )


def apply_mtcnn_patch():
    """Patch MTCNN's weight-loading references."""
    from mtcnn.utils import tensorflow as mtcnn_tf
    import mtcnn.stages.stage_pnet as stage_pnet
    import mtcnn.stages.stage_rnet as stage_rnet
    import mtcnn.stages.stage_onet as stage_onet

    mtcnn_tf.load_weights = safe_load_weights
    stage_pnet.load_weights = safe_load_weights
    stage_rnet.load_weights = safe_load_weights
    stage_onet.load_weights = safe_load_weights