import os
import skimage.io as io
import numpy as np


class Dataset:
    @staticmethod
    def prepare():
        binaryLabelsDir = 'data/membrane/train/label'
        multiclassLabelsDir = 'data/membrane/train/multiclass_label'
        os.makedirs(multiclassLabelsDir, exist_ok=True)
        for fileName in os.listdir(binaryLabelsDir):
            if not fileName.endswith('.png'):
                continue
            binaryLabelImg = io.imread(os.path.join(binaryLabelsDir, fileName), as_gray=True)
            assert binaryLabelImg.dtype == np.uint8
            # TODO: 0 -> 2 255-> 3 and save to
            multi = binaryLabelImg
            multi[binaryLabelImg == 0] = 2
            multi[binaryLabelImg == 255] = 3
            io.imsave(os.path.join(multiclassLabelsDir, fileName), multi)


class Trainer:
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    train_path = 'data/membrane/train'
    image_folder = 'image'
    mask_folder = 'multiclass_label'

    targetSize = (512, 512)
    batch_size = 1
    seed = 1
    num_class = 6

    def getTrainGenerator(self):
        from keras.preprocessing.image import ImageDataGenerator

        image_datagen = ImageDataGenerator(**self.data_gen_args)
        mask_datagen = ImageDataGenerator(**self.data_gen_args)
        image_generator = image_datagen.flow_from_directory(
            self.train_path,
            classes=[self.image_folder],
            class_mode=None,
            color_mode='grayscale',
            target_size=self.targetSize,
            batch_size=self.batch_size,
            save_to_dir=None,
            save_prefix=None,
            seed=self.seed)
        mask_generator = mask_datagen.flow_from_directory(
            self.train_path,
            classes=[self.mask_folder],
            class_mode=None,
            color_mode='grayscale',
            target_size=self.targetSize,
            batch_size=self.batch_size,
            save_to_dir=None,
            save_prefix=None,
            seed=self.seed)
        train_generator = zip(image_generator, mask_generator)
        for (img, mask) in train_generator:
            img, mask = self.adjustData(img, mask)
            yield (img, mask)

    def adjustData(self, img, mask):
        img = img / 255
        mask = mask / (self.num_class - 1) # from 0.0 to 1.0
        return img, mask

    def train(self):
        from model import unet
        from data import trainGenerator
        from keras.callbacks import ModelCheckpoint
        from keras.optimizers import Adam
        from keras.callbacks import ReduceLROnPlateau

        # myGene = trainGenerator(4, 'data', 'colorImage', 'label', self.data_gen_args, image_color_mode='rgb',
        #                         target_size=self.targetSize, flag_multi_class=True, num_class=6, save_to_dir=None)
        myGene = self.getTrainGenerator()

        model = unet(input_size=self.targetSize + (1,))
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        chckPtsDir = 'checkpoints/membrane/multiclass'
        os.makedirs(chckPtsDir, exist_ok=True)
        chckPtsPath = os.path.join(chckPtsDir, 'unet_multiclass_{epoch}_{loss:.3f}_{acc:.3f}.hdf5')
        model_checkpoint = ModelCheckpoint(chckPtsPath, monitor='loss', verbose=1, save_best_only=True)
        # reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=4, min_lr=0.001)
        model.fit_generator(myGene, steps_per_epoch=20, epochs=3, callbacks=[model_checkpoint])


def main():
    Trainer().train()


main()
