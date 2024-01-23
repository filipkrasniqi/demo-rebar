import os
import pandas as pd
import tensorflow as tf

from model.loss.loss import RetinaNetLoss
from model.retinanet import RetinaNet
from model.utils.data import load_dataset
from model.utils.label_encoder import LabelEncoder
from model.utils.utils import get_backbone, adapt_dataset, get_dataset_path

if __name__ == "__main__":
    multiply_bbox = True
    base_dir = get_dataset_path()
    images_path = os.path.join(base_dir, 'images')
    annotations_path = os.path.join(base_dir, "annotations")
    dataset_path = os.path.join(base_dir, "dataset")
    os.makedirs(dataset_path, exist_ok=True)

    train_ds_path = os.path.join(dataset_path, "train_v4.tfrecord")
    val_ds_path = os.path.join(dataset_path, "validation_v4.tfrecord")

    path_train_csv = os.path.join(annotations_path, "train_ds.csv")
    path_validation_csv = os.path.join(annotations_path, "val_ds.csv")

    df_train, df_val = pd.read_csv(path_train_csv), pd.read_csv(
        path_validation_csv)

    train_size = len(df_train['filename'].unique())
    validation_size = len(df_val['filename'].unique())

    batch_size = 4

    label_encoder = LabelEncoder()
    train_dataset = load_dataset(train_ds_path)
    val_dataset = load_dataset(val_ds_path)

    train_dataset = adapt_dataset(
        ds=train_dataset,
        ds_len=train_size,
        batch_size=batch_size,
        label_encoder=label_encoder,
        multiply_bbox=multiply_bbox,
    )
    val_dataset = adapt_dataset(
        ds=val_dataset,
        ds_len=validation_size,
        batch_size=batch_size,
        label_encoder=label_encoder,
        multiply_bbox=multiply_bbox,
    )

    print("DATASET loaded!")

    model_dir = "retinanet-v3/"

    num_classes = 1

    learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
    learning_rate_boundaries = [125, 250, 500, 240000, 360000]
    learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=learning_rate_boundaries, values=learning_rates)

    resnet50_backbone = get_backbone()
    loss_fn = RetinaNetLoss(num_classes)
    model = RetinaNet(num_classes, resnet50_backbone)

    optimizer = tf.keras.optimizers.legacy.SGD(
        learning_rate=learning_rate_fn,
        momentum=0.9,
    )
    model.compile(
        loss=loss_fn,
        optimizer=optimizer,
    )

    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
            monitor="loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )
    ]

    train_steps_per_epoch = train_size // batch_size

    train_steps = 4 * 10000
    epochs = train_steps // train_steps_per_epoch

    print("Training...")

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1,
    )
