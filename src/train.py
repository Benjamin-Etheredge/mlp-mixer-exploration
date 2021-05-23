import tensorflow as tf
import tensorflow_datasets as tfds
import click
import dataloader
import mlp_mixer_keras
import dvclive
from dvclive.keras import DvcLiveCallback


@click.command()
#@click.argument('data-path')
#@click.argument('meta-path')
#@click.option('--batch-size', default=256)
def main():
    # TODO probably messes up softmax final layers
    # TODO update library to cast softmax layer with float32
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    ds = tfds.load('imagenet2012', split='train', 
        shuffle_files=True,
        as_supervised=True,
        #batch_size=64, 
        #download_and_prepare_kwargs={
            #'download_config': config
        #}
    )
    # TODO test shuffle from tfds
    def foo(img):
        img = tf.image.convert_image_dtype(img, 'float16') # TODO set to policy
        img = tf.image.resize(img, (224, 224))
        # TODO random crop
        img = tf.image.random_flip_left_right(img)
        return img
    
    ds = ds.map(lambda x, y: (foo(x), y), tf.data.AUTOTUNE)
    ds = ds.batch(64)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    #print('len', len(ds))
    #print(next(iter(ds)))
    #ds = ds.map(lambda x, y: (tf.io.read_file(x), y), tf.data.AUTOTUNE)
    #ds = ds.filter(lambda x, _: tf.io.is_jpeg(x))
    #ds = tfds.load('imagenet2012', split='train', shuffle_files=True)
    model = mlp_mixer_keras.MlpMixerModel(
            input_shape=(224, 224, 3),
            num_classes=1000,
            num_blocks=16, 
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
    model.fit(ds, epochs=10, callbacks=[DvcLiveCallback()])

if __name__ == "__main__":
    main()
