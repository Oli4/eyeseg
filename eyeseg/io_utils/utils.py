from itertools import cycle
from pathlib import Path
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from scipy import ndimage

import eyepy as ep
from eyeseg.io_utils.metrics import get_mae, get_layer_mae, get_curv, get_layer_curv

DATA_PATH = Path("/home/data")


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_metrics(layer_mapping):
    mae = [
        get_mae(),
    ] + [get_layer_mae(i, name) for i, name in layer_mapping.items()]
    curv_mae = [
        get_curv(),
    ] + [get_layer_curv(i, name) for i, name in layer_mapping.items()]

    metrics = {"layer_output": mae + curv_mae}
    return metrics


def preprocess_split(volume_paths, savepath, split, excluded=None):
    if not excluded:
        excluded = []

    writers = []
    for i in range(10):
        writers.append(tf.io.TFRecordWriter(str(savepath / f"{split}_{i}.tfrecords")))
    writers_cycle = cycle(writers)
    writer = next(writers_cycle)

    for p in tqdm(volume_paths):
        # Load volume
        data = ep.Oct.from_duke_mat(p)
        # Compute center of annotation
        bm_annotation = (~np.isnan(data.analyse["BM"])).astype(int)
        height_center, width_center = [
            int(c) for c in ndimage.measurements.center_of_mass(bm_annotation)
        ]

        # Extract subvolume
        for bscan in data[height_center - 25 : height_center + 25]:

            if [p.name.rstrip(".mat"), bscan.name] in excluded:
                print(
                    f"Excluded {p.name.rstrip('.mat'), bscan.name} due to low quality."
                )
                continue
            try:
                image = bscan.scan[:, width_center - 256 : width_center + 256].astype(
                    np.uint8
                )
                bm = bscan.analyse["BM"][
                    width_center - 256 : width_center + 256
                ].astype(np.float32)
                rpe = bscan.analyse["RPE"][
                    width_center - 256 : width_center + 256
                ].astype(np.float32)
                ilm = bscan.analyse["ILM"][
                    width_center - 256 : width_center + 256
                ].astype(np.float32)

                # Save image and labels
                feature = {
                    "bm": _bytes_feature(
                        tf.io.serialize_tensor(tf.convert_to_tensor(bm))
                    ),
                    "ibrpe": _bytes_feature(
                        tf.io.serialize_tensor(tf.convert_to_tensor(rpe))
                    ),
                    "ilm": _bytes_feature(
                        tf.io.serialize_tensor(tf.convert_to_tensor(ilm))
                    ),
                    "volume": _bytes_feature(p.name.rstrip(".mat").encode("utf8")),
                    "bscan": _bytes_feature(bscan.name.encode("utf8")),
                    "image": _bytes_feature(
                        tf.io.serialize_tensor(tf.convert_to_tensor(image))
                    ),
                }

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
                writer = next(writers_cycle)
            except Exception as e:
                print(
                    f"Excluded {p.name.rstrip('.mat'), bscan.name} due to Exception: {e}"
                )

    for w in writers:
        w.close()
