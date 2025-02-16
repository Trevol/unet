from model import *
from data import *


def main():
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    data_gen_args = dict()  # try without augmentations

    colorMode = "grayscale"
    myGene = trainGenerator(2, 'data', 'image', 'label', data_gen_args, image_color_mode=colorMode,
                            target_size=(512, 512),
                            save_to_dir=None)

    model = unet(input_size=(512, 512, 1))
    model.load_weights('../unet_membrane_5_0.123_0.946.hdf5')

    os.makedirs('checkpoints', exist_ok=True)
    model_checkpoint = ModelCheckpoint(f'checkpoints/grayscale_shapes/unet_{colorMode}_shapes_{{epoch}}_{{loss:.4f}}_{{acc:.3f}}.hdf5',
                                       monitor='loss',
                                       verbose=1, save_best_only=True)
    model.fit_generator(myGene, steps_per_epoch=2000, epochs=5, callbacks=[model_checkpoint])

    # imgs_train,imgs_mask_train = geneTrainNpy("data/membrane/train/aug/","data/membrane/train/aug/")
    # model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=10, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

    # testGene = testGenerator("data/membrane/test")
    # model = unet()
    # model.load_weights("unet_pins.hdf5")
    # results = model.predict_generator(testGene, 30, verbose=1)
    # saveResult("data/membrane/test", results)


main()
