import tensorflow as tf


def parse_tfrecord(record):
    feature_description = {
        "input":
            tf.io.FixedLenFeature([], tf.string),
        "labels":
            tf.io.FixedLenSequenceFeature(
                [],
                allow_missing=True,
                dtype=tf.int64,
            ),
        "bbox":
            tf.io.FixedLenSequenceFeature(
                [],
                allow_missing=True,
                dtype=tf.float32,
            ),
    }

    record = tf.io.parse_single_example(record, feature_description)
    record["input"] = tf.image.decode_png(
        record["input"],
        channels=3,
        dtype=tf.uint8,
    )
    return record


def load_dataset(tfrecord_path: str) -> str:
    """Loads tf record as dataset given the path.
    Args:
        tfrecord_path: path to tf record

    Returns:

    """
    ds = tf.data.TFRecordDataset(
        [tfrecord_path],
        num_parallel_reads=tf.data.AUTOTUNE,
    )

    ds = ds.map(
        parse_tfrecord,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds
