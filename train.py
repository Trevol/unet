import skimage.io
import numpy as np
import os




def generateColorImages(grayImagesDir, colorImagesDir):
    os.makedirs(colorImagesDir, exist_ok=True)
    for fileName in sorted(os.listdir(grayImagesDir)):
        if not fileName.endswith('.png') or '_predict.' in fileName:
            continue
        grayFilePath = os.path.join(grayImagesDir, fileName)
        grayImg = skimage.io.imread(grayFilePath, as_gray=True)
        colorImagePath = os.path.join(colorImagesDir, fileName)
        colorImg = np.dstack([grayImg] * 3)
        skimage.io.imsave(colorImagePath, colorImg)
        print(colorImagePath, colorImg.shape, colorImg.dtype)


def trainGrayscale():
    from model import unet
    from data import trainGenerator
    from keras.callbacks import ModelCheckpoint
    from keras.optimizers import Adam

    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    myGene = trainGenerator(4, 'data/membrane/train', 'image', 'label', data_gen_args, save_to_dir=None)

    model = unet()
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    model_checkpoint = ModelCheckpoint('unet_membrane_{epoch}_{loss:.3f}_{acc:.3f}.hdf5', monitor='loss', verbose=1,
                                       save_best_only=True)
    model.fit_generator(myGene, steps_per_epoch=500, epochs=5, callbacks=[model_checkpoint])

    # imgs_train,imgs_mask_train = geneTrainNpy("data/membrane/train/aug/","data/membrane/train/aug/")
    # model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=10, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

    # testGene = testGenerator("data/membrane/test")
    # model = unet()
    # model.load_weights("unet_membrane.hdf5")
    # results = model.predict_generator(testGene,30,verbose=1)
    # saveResult("data/membrane/test",results)


def trainColor():
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
    targetSize = (256, 256)
    myGene = trainGenerator(4, 'data/membrane/train', 'colorImage', 'label', data_gen_args, image_color_mode='rgb',
                            target_size=targetSize, save_to_dir=None)

    model = unet(input_size=targetSize + (3,))
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    chckPtsDir = 'checkpoints'
    os.makedirs(chckPtsDir, exist_ok=True)
    chckPtsPath = os.path.join(chckPtsDir, 'unet_membrane_{epoch}_{loss:.3f}_{acc:.3f}.hdf5')
    model_checkpoint = ModelCheckpoint(chckPtsPath, monitor='loss', verbose=1, save_best_only=True)
    # reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=4, min_lr=0.001)
    model.fit_generator(myGene, steps_per_epoch=2000, epochs=17, callbacks=[model_checkpoint])

    # imgs_train,imgs_mask_train = geneTrainNpy("data/membrane/train/aug/","data/membrane/train/aug/")
    # model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=10, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

    # testGene = testGenerator("data/membrane/test")
    # model = unet()
    # model.load_weights("unet_membrane.hdf5")
    # results = model.predict_generator(testGene,30,verbose=1)
    # saveResult("data/membrane/test",results)


def main():
    generateColorImages('data/membrane/test', 'data/membrane/testColor')
    # trainColor()
    # trainGrayscale()


main()
