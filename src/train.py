import tensorflow as tf
import tensorflow_datasets as tfds
import click
import dataloader
import mlp_mixer_keras
import dvclive
from dvclive.keras import DvcLiveCallback
import mlflow
mlflow.tensorflow.autolog()

# TODO test shuffle from tfds
def prep_img(img, batch_size):
    img = tf.image.convert_image_dtype(img, 'float32') # TODO set to policy
    img = tf.image.resize(img, (224+28, 224+28))
    img = tf.image.random_crop(img, [224, 224, 3])
    #img = tf.image.random_crop(img, [batch_size, 224, 224, 3])
    img = tf.image.random_flip_left_right(img)
    return img

 
def prep_ds(in_ds, batch_size):
    ds = in_ds
    # NOTE adding non deterministic map speed it up a decent bit (~84% util -> ~94% util) maybe. Batch size may have done that
    ds = ds.map(lambda x, y: (prep_img(x, batch_size), y), tf.data.AUTOTUNE, deterministic=False)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


@click.command()
#@click.argument('data-path')
#@click.argument('meta-path')
@click.option('--batch-size', default=256)
def main(batch_size):
    # TODO probably messes up softmax final layers
    # TODO update library to cast softmax layer with float32
    dvclive.init('metrics')
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    ds, val_ds = tfds.load(
        'imagenet2012', 
        split=['train', 'validation'], 
        shuffle_files=True,
        as_supervised=True,
        #batch_size=batch_size,  # NOTE this should allows for faster vectorized mappings
        #download_and_prepare_kwargs={
            #'download_config': config
        #}
    )
    ds = prep_ds(ds, batch_size)
    val_ds = prep_ds(val_ds, batch_size)

    # TODO override get_config
    model = mlp_mixer_keras.MlpMixerModel(
            input_shape=(224, 224, 3),
            num_classes=1000,
            num_blocks=8, 
            patch_size=32,
            hidden_dim=512, 
            tokens_mlp_dim=256,
            channels_mlp_dim=2048,
            use_softmax=True)


    model.compile(
        #optimizer=tf.keras.optimizers.Adam(beta1=0.9, beta2=0.999),
        metrics=['acc'],
        loss=tf.keras.losses.SparseCategoricalCrossentropy())
    model.summary()
    model.fit(
        ds, 
        epochs=20, 
        validation_data=val_ds,
        callbacks=[DvcLiveCallback()])

if __name__ == "__main__":
    main()
