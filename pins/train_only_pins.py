import argparse

from keras.callbacks import ReduceLROnPlateau

from model import *
from data import *


def main():
    arg = argparse.ArgumentParser()
    arg.add_argument('--batch_size', type=int, default=2)
    arg = arg.parse_args()
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')

    colorMode = "rgb"
    myGene = trainGenerator(arg.batch_size, 'data', 'image', 'pin_only_masks', data_gen_args,
                            image_color_mode=colorMode,
                            target_size=(256, 256),
                            save_to_dir=None)

    model = unet(input_size=(256, 256, 3))
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.load_weights('../checkpoints/membrane/color/unet_membrane_17_0.042_0.982.hdf5')

    checkpointsDir = 'checkpoints/rgb'
    os.makedirs(checkpointsDir, exist_ok=True)
    model_checkpoint = ModelCheckpoint(f'{checkpointsDir}/unet_pins_{{epoch}}_{{loss:.3f}}_{{acc:.3f}}.hdf5',
                                       monitor='loss',
                                       verbose=1, save_best_only=True)

    model.fit_generator(myGene, steps_per_epoch=2000, epochs=20, callbacks=[model_checkpoint])


main()
