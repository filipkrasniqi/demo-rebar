import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from typing import Tuple, Dict, List

from .data import load_dataset


def get_test_dataset(
    base_dir: str,
    test_tf_record: str = "test_v3.tfrecord",
):
    dataset_path = os.path.join(base_dir, "dataset")
    test_ds_path = os.path.join(dataset_path, test_tf_record)

    return load_dataset(test_ds_path)


def resize_and_pad_image(
    image,
    min_side=800.0,
    max_side=1333.0,
    jitter=[640, 1024],
    stride=128.0,
):
    """Resizes and pads image while preserving aspect ratio.

    1. Resizes images so that the shorter side is equal to `min_side`
    2. If the longer side is greater than `max_side`, then resize the image
      with longer side equal to `max_side`
    3. Pad with zeros on right and bottom to make the image shape divisible by
    `stride`

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      min_side: The shorter side of the image is resized to this value, if
        `jitter` is set to None.
      max_side: If the longer side of the image exceeds this value after
        resizing, the image is resized such that the longer side now equals to
        this value.
      jitter: A list of floats containing minimum and maximum size for scale
        jittering. If available, the shorter side of the image will be
        resized to a random value in this range.
      stride: The stride of the smallest feature map in the feature pyramid.
        Can be calculated using `image_size / feature_map_size`.

    Returns:
      image: Resized and padded image.
      image_shape: Shape of the image before padding.
      ratio: The scaling factor used to resize the image
    """
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    if jitter is not None:
        min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)
    ratio = min_side / tf.reduce_min(image_shape)
    if ratio * tf.reduce_max(image_shape) > max_side:
        ratio = max_side / tf.reduce_max(image_shape)
    image_shape = ratio * image_shape
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
    padded_image_shape = tf.cast(tf.math.ceil(image_shape / stride) * stride,
                                 dtype=tf.int32)
    image = tf.image.pad_to_bounding_box(image, 0, 0, padded_image_shape[0],
                                         padded_image_shape[1])
    return image, image_shape, ratio


def inference_model(
    model: tf.keras.Model,
    decoder,
    weights_dir: str = "../training/retinanet-v1",
    weight_epoch: int = 3,
) -> tf.keras.Model:

    model.load_weights(
        os.path.join(
            weights_dir,
            f"weights_epoch_{weight_epoch}",
        ))

    img = tf.keras.Input(shape=[None, None, 3], name="image")
    predictions = model(img, training=False)
    detections = decoder(img, predictions)

    return tf.keras.Model(inputs=img, outputs=detections)


def prepare_image_for_inference(image: tf.Tensor) -> Tuple[tf.Tensor, float]:
    image, _, ratio = resize_and_pad_image(image, jitter=None)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio


def infer(
    model: tf.keras.Model,
    sample: Dict,
) -> Tuple[np.ndarray, int, float]:
    image = tf.cast(sample['input'], dtype=tf.float32)
    input_image, ratio = prepare_image_for_inference(image)
    detections = model.predict(input_image, verbose=0)
    num_detections = detections.valid_detections[0]

    return detections, num_detections, ratio


def visualize_detections(
        image,
        boxes,
        classes,
        scores,
        figsize=(7, 7),
        linewidth=1,
        color=[0, 0, 1],
        show_confidence=False,
):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle([x1, y1],
                              w,
                              h,
                              fill=False,
                              edgecolor=color,
                              linewidth=linewidth)
        ax.add_patch(patch)

        if show_confidence:
            ax.text(
                x1,
                y1,
                text,
                bbox={
                    "facecolor": color,
                    "alpha": 0.4
                },
                clip_box=ax.clipbox,
                clip_on=True,
            )
    plt.show()
    return ax


def show_sample(
    model: tf.keras.Model,
    sample: Dict,
):
    image = tf.cast(sample['input'], dtype=tf.float32)

    detections, num_detections, ratio = infer(
        model=model,
        sample=sample,
    )

    visualize_detections(
        image,
        detections.nmsed_boxes[0][:num_detections] / ratio,
        ["Obj" for i in range(num_detections)],
        detections.nmsed_scores[0][:num_detections],
    )


def compute_num_objects(
    model: tf.keras.Model,
    ds,
    n_to_take: int = 1,
) -> List[int]:
    return [
        n for _, n, _ in
        [infer(model=model, sample=sample) for sample in ds.take(n_to_take)]
    ]
