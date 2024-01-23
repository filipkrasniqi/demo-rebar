import os
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from functools import partial
from tensorflow import keras
from tqdm import tqdm
from typing import Tuple

from model.utils.inference import resize_and_pad_image

load_dotenv()


def get_dataset_path() -> str:
    return os.environ["DATASET_PATH"]


def convert_to_corners(boxes: tf.Tensor) -> tf.Tensor:
    """Changes the box format to corner coordinates

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [
            boxes[..., :2] - boxes[..., 2:] / 2.0,
            boxes[..., :2] + boxes[..., 2:] / 2.0
        ],
        axis=-1,
    )


def swap_xy(boxes: tf.Tensor) -> tf.Tensor:
    """Swaps order the of x and y coordinates of the boxes.

    Arguments:
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes.

    Returns:
      swapped boxes with shape same as that of boxes.
    """
    return tf.stack(
        [boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]],
        axis=-1,
    )


def convert_to_xywh(boxes: tf.Tensor) -> tf.Tensor:
    """Changes the box format to center, width and height.

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[xmin, ymin, xmax, ymax]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0,
         boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )


def compute_iou(
    boxes1: tf.Tensor,
    boxes2: tf.Tensor,
) -> tf.Tensor:
    """Computes pairwise IOU matrix for given two sets of boxes

    Arguments:
      boxes1: A tensor with shape `(N, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
        boxes2: A tensor with shape `(M, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.

    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
    """
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8)
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)


def random_flip_horizontal(
    image: tf.Tensor,
    boxes: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Flips image and boxes horizontally with 50% chance

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes,
        having normalized coordinates.

    Returns:
      Randomly flipped image and boxes
    """
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]],
            axis=-1)
    return image, boxes


def preprocess_data_tuple(
    sample: Tuple,
    swap: bool = False,
    multiply_bbox: bool = False,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Applies preprocessing step to a single sample

    Arguments:
      sample: A dict representing a single training sample.

    Returns:
      image: Resized and padded image with random horizontal flipping applied.
      bbox: Bounding boxes with the shape `(num_objects, 4)` where each box is
        of the format `[x, y, width, height]`.
      class_id: An tensor representing the class id of the objects, having
        shape `(num_objects,)`.
    """
    image, bbox, class_id = sample
    if swap:
        bbox = swap_xy(bbox)
    class_id = tf.cast(class_id, dtype=tf.int32)

    image, bbox = random_flip_horizontal(image, bbox)
    image, image_shape, _ = resize_and_pad_image(image)

    if multiply_bbox:
        bbox = tf.stack(
            [
                bbox[:, 0] * image_shape[1],
                bbox[:, 1] * image_shape[0],
                bbox[:, 2] * image_shape[1],
                bbox[:, 3] * image_shape[0],
            ],
            axis=-1,
        )
    else:
        bbox = tf.stack(
            [
                bbox[:, 0],
                bbox[:, 1],
                bbox[:, 2],
                bbox[:, 3],
            ],
            axis=-1,
        )
    bbox = convert_to_xywh(bbox)
    return image, bbox, class_id


def preprocess_data(sample) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Applies preprocessing step to a single sample

    Arguments:
      sample: A dict representing a single training sample.

    Returns:
      image: Resized and padded image with random horizontal flipping applied.
      bbox: Bounding boxes with the shape `(num_objects, 4)` where each box is
        of the format `[x, y, width, height]`.
      class_id: An tensor representing the class id of the objects, having
        shape `(num_objects,)`.
    """
    return preprocess_data_tuple(
        (sample["image"], sample["objects"]["bbox"],
         sample["objects"]["label"]),
        swap=True,
        multiply_bbox=True,
    )


def get_backbone() -> keras.Model:
    """Builds ResNet50 with pre-trained imagenet weights"""
    backbone = keras.applications.ResNet50(include_top=False,
                                           input_shape=[None, None, 3])
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output for layer_name in
        ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]
    return keras.Model(inputs=[backbone.inputs],
                       outputs=[c3_output, c4_output, c5_output])


def build_head(output_filters, bias_init):
    """Builds the class/box predictions head.

    Arguments:
      output_filters: Number of convolution filters in the final layer.
      bias_init: Bias Initializer for the final convolution layer.

    Returns:
      A keras sequential model representing either the classification
        or the box regression head depending on `output_filters`.
    """
    head = keras.Sequential([keras.Input(shape=[None, None, 256])])
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    for _ in range(4):
        head.add(
            keras.layers.Conv2D(256,
                                3,
                                padding="same",
                                kernel_initializer=kernel_init))
        head.add(keras.layers.ReLU())
    head.add(
        keras.layers.Conv2D(
            output_filters,
            3,
            1,
            padding="same",
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
        ))
    return head


def image_feature(x) -> tf.train.Feature:
    """Returns a bytes_list from a string / byte."""
    value = tf.io.encode_png(x)
    value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bbox_features(
    df: pd.DataFrame,
    keys: str,
) -> tf.train.Feature:
    """Returns a dataframe, selected by keys, flattened.
    Args:
        df: pd.DataFrame containing the original features
        keys: columns identifying the keys

    Returns:
        tf.train.Feature with flattened values from dataframe

    """
    vals = df[keys].to_numpy().flatten()
    return tf.train.Feature(float_list=tf.train.FloatList(value=vals))


def labels_feature(num_bboxes: int) -> tf.train.Feature:
    """Initializes all classes to 0, as they all belong to same class."""
    return tf.train.Feature(int64_list=tf.train.Int64List(
        value=[0 for _ in range(num_bboxes)]))


def create_tf_record(
    df: pd.DataFrame,
    file_path: str,
) -> tf.train.Example:
    """Builds single tf record given a df containing all bboxes."""
    feature = {
        "input":
            image_feature(
                tf.image.decode_png(
                    tf.io.read_file(file_path),
                    channels=3,
                    dtype=tf.uint8,
                )),
        "labels":
            labels_feature(num_bboxes=len(df)),
    }
    keys = ['bbox-1', 'bbox-0', 'bbox-3', 'bbox-2']
    feature['bbox'] = bbox_features(df, keys=keys)

    return tf.train.Example(features=tf.train.Features(feature=feature))


def build_tfrecord(
    df: pd.DataFrame,
    images_path: str,
    output_path: str,
    overwrite: bool = False,
):
    """Builds TF record from a dataframe containing all bboxes and filenames."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if overwrite or not os.path.exists(output_path):
        with tf.io.TFRecordWriter(output_path) as writer:
            df_g = df.groupby('filename')

            for filename, df_current in tqdm(
                    df_g,
                    total=len(df_g),
                    desc=f"Converting to tf records...",
            ):
                writer.write(
                    create_tf_record(
                        df=df_current,
                        file_path=os.path.join(images_path, filename),
                    ).SerializeToString())
    else:
        print("Reading from cache")


def adapt(
    data_structure,
    multiply_bbox: bool = True,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Adapts tf record in dict form for RetinaNet."""
    batch_images = data_structure['input']
    gt_boxes = tf.reshape(data_structure['bbox'], [-1, 4])
    cls_ids = data_structure['labels']

    return preprocess_data_tuple(
        (batch_images, gt_boxes, cls_ids),
        swap=True,
        multiply_bbox=multiply_bbox,
    )


def adapt_dataset(
    ds,
    ds_len,
    batch_size,
    label_encoder,
    multiply_bbox,
    seed: int = 1234,
) -> tf.data.Dataset:
    """Builds dataset pipeline."""
    autotune = tf.data.AUTOTUNE
    ds = ds.apply(tf.data.experimental.assert_cardinality(ds_len))
    ds = ds.cache()

    ds = ds.shuffle(
        3 * batch_size,
        seed=seed,
    )

    ds = ds.map(
        partial(adapt, multiply_bbox=multiply_bbox),
        num_parallel_calls=autotune,
    )

    ds = ds.padded_batch(
        batch_size=batch_size,
        padding_values=(0.0, 1e-8, -1),
        drop_remainder=True,
    )

    ds = ds.map(
        label_encoder.encode_batch,
        num_parallel_calls=autotune,
    )

    ds = ds.prefetch(autotune)
    return ds
