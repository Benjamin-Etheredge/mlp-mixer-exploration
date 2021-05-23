import pandas as pd
import tensorflow as tf
import os

from tensorflow._api.v2 import data


# TODO is jpeg and write log of how many and what is filtered

def load_meta(path):
    meta = pd.read_csv(path)
    ids = [id-1 for id in meta.id.values]
    wnid_to_id = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            values=ids, keys=meta.wnid.values), 
        default_value=-1)
    return wnid_to_id


def get_wnid(path):
    file_name = tf.strings.split(path, sep=os.sep)[-1]
    split_name = tf.strings.split(file_name, sep='_')
    wnid = split_name[0]
    return wnid


# TODO random mutations
# TODO test setting shape
def img_decode(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, 'float32')
    img = tf.image.resize(img, (224, 224))
    # TODO check if shape is greater than a certain amount, if so random crop,
    # TODO else just resize
    # TODO make sure mutations don't go above/below 0/1
    return img


# TODO test running al lin parallel rather than stacking
def loader(path, meta_path):
    files_ds = tf.data.Dataset.list_files(os.sep.join((path, "*")))

    wnid_ds = files_ds.map(get_wnid, tf.data.AUTOTUNE)
    wnid_to_id_hash = load_meta(meta_path)
    ids_ds = wnid_ds.map(lambda x: wnid_to_id_hash.lookup(x), tf.data.AUTOTUNE) #.cache()

    ds = tf.data.Dataset.zip((files_ds, ids_ds)) #.shuffle(len(files_ds), reshuffle_each_iteration=True)
    ds = ds.cache()
    ds = ds.shuffle(len(files_ds), reshuffle_each_iteration=True)
    ds = ds.map(lambda x, y: (tf.io.read_file(x), y), tf.data.AUTOTUNE)
    ds = ds.filter(lambda x, _: tf.io.is_jpeg(x))

    ds = ds.map(lambda x, y: (img_decode(x), y), tf.data.AUTOTUNE)

    return ds
