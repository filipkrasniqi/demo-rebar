import os.path
import sys

import tensorflow as tf

from model.layers.decode_predictions import DecodePredictions
from model.retinanet import RetinaNet
from model.utils.inference import inference_model, show_sample, infer
from model.utils.utils import get_backbone, get_dataset_path

if __name__ == "__main__":
    args = sys.argv
    assert len(args) > 1, "Requires file in input"
    fpath = args[1]
    assert os.path.exists(fpath), "File not found"

    num_classes = 1
    resnet50_backbone = get_backbone()
    base_dir = get_dataset_path()

    retina_net = RetinaNet(num_classes, resnet50_backbone)

    model = inference_model(
        model=retina_net,
        decoder=DecodePredictions(
            confidence_threshold=.35,
            max_detections=1000,
            max_detections_per_class=1000,
            nms_iou_threshold=.6,
        ),
        weights_dir="../weights/retinanet-v1",
        weight_epoch=3,
    )

    sample = {
        "input":
            tf.image.decode_png(
                tf.io.read_file(fpath),
                channels=3,
                dtype=tf.uint8,
            ),
    }

    show_sample(
        model=model,
        sample=sample,
    )

    _, n, _ = infer(model=model, sample=sample)

    print(f"Number of detected objects: {n}")
