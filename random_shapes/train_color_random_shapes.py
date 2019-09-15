import os
from keras.callbacks import ReduceLROnPlateau

from model import *
from data import *


class CheckPointManager:
    def __init__(self, checkpointsRoot, checkpointBaseName):
        self.checkpointsRoot = checkpointsRoot
        self.checkpointBaseName = checkpointBaseName or 'chkpt'

    def nextTrainingSession(self, **checkpointKwargs):
        checkpointFileName = self.checkpointBaseName + '_{epoch}_{loss:.4f}_{acc:.3f}.hdf5'
        # prepare storage
        sessionDir = self.__sessionDirectory()
        sessionCheckpointsDir = os.path.join(self.checkpointsRoot, sessionDir)
        os.makedirs(sessionCheckpointsDir, exist_ok=True)

        filePath = os.path.join(self.checkpointsRoot, checkpointFileName)

        model_checkpoint = ModelCheckpoint(filePath, **checkpointKwargs)
        return model_checkpoint


def main():
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')

    colorMode = "rgb"
    myGene = trainGenerator(4, 'data', 'image', 'label', data_gen_args, image_color_mode=colorMode,
                            target_size=(256, 256),
                            save_to_dir=None)

    model = unet(input_size=(256, 256, 3))
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # checkpointManager = CheckPointManager('checkpoints/color_shapes', 'shapes_{epoch}_{loss:.4f}_{acc:.3f}.hdf5')
    # modelCheckpoint = checkpointManager.nextTrainingSession()

    os.makedirs('checkpoints/color_shapes', exist_ok=True)
    model_checkpoint = ModelCheckpoint(
        f'checkpoints/color_shapes/shapes_{{epoch}}_{{loss:.4f}}_{{acc:.3f}}.hdf5',
        monitor='loss',
        verbose=1, save_best_only=True)

    model.load_weights('../checkpoints/membrane/color/unet_membrane_17_0.042_0.982.hdf5')
    # model.load_weights(checkpointManager.bestCheckpoint(session=None))

    model.fit_generator(myGene, steps_per_epoch=2000, epochs=20, callbacks=[model_checkpoint])


main()
