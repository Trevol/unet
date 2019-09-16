import os


def train():
    from model import unet
    from data import trainGenerator
    from keras.callbacks import ModelCheckpoint
    from keras.optimizers import Adam
    from keras.callbacks import ReduceLROnPlateau

    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    targetSize = (512, 512)
    myGene = trainGenerator(4, 'data', 'colorImage', 'label', data_gen_args, image_color_mode='rgb',
                            target_size=targetSize, flag_multi_class=True, num_class=6, save_to_dir=None)

    model = unet(input_size=targetSize + (3,))
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    chckPtsDir = 'checkpoints/multiclass'
    os.makedirs(chckPtsDir, exist_ok=True)
    chckPtsPath = os.path.join(chckPtsDir, 'unet_multiclass_{epoch}_{loss:.3f}_{acc:.3f}.hdf5')
    model_checkpoint = ModelCheckpoint(chckPtsPath, monitor='loss', verbose=1, save_best_only=True)
    # reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=4, min_lr=0.001)
    model.fit_generator(myGene, steps_per_epoch=20, epochs=3, callbacks=[model_checkpoint])


def testFlow():
    from data import trainGenerator
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    targetSize = (16, 16)
    myGene = trainGenerator(1, 'data', 'image', 'multi_class_masks', data_gen_args, image_color_mode='rgb',
                            target_size=targetSize, flag_multi_class=True, num_class=6, save_to_dir=None)
    for _ in range(4):
        im, mask = next(myGene)
        im

def main():
    testFlow()
    # train()


main()
