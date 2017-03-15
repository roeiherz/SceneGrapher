import cPickle
import os

from keras.callbacks import ModelCheckpoint, ProgbarLogger, TensorBoard, LearningRateScheduler
from keras.layers import Convolution2D, GlobalAveragePooling2D, Dense, functools
from keras.optimizers import SGD

from Trax.Algo.Deep.Core.Architectures.SimpleCNN import NUMBER_OF_EPOCHS, SAMPLES_PER_EPOCH, LogEpochResults, schedual
from Trax.Algo.Deep.Core.Architectures.Zoo import ModelZoo
from Trax.Algo.Deep.Core.Datasets.Generators.BasicGenerator import WholeProbeToyGenerator
from Trax.Algo.Deep.Core.GlobalResources import TRAINING_OPTIMIZATION_SECTION
from Trax.Data.Benchmarks.Core import Benchmark
from Trax.Utils.Conf.Configuration import Config
from Trax.Utils.Files.FilesServices import create_folder
from Trax.Utils.Logging.Logger import Log


__author__ = 'itsikl'

if __name__ == '__main__':
    Config().init()
    Log().init('Whole image detector')

    benchmark_id = 315
    benchmark = Benchmark(benchmark_id)
    benchmark.download_probes()
    probes = benchmark.get_probes()
    probes_ids = benchmark.get_probes_ids()
    images = []


    output_folder = '/home/itsikl/engineCache/Deep/results'
    data_dest = '/home/itsikl/engineCache/Deep/datasets/ccru/315_newdataset/ccru_315_newdataset.p'
    probes_data = cPickle.load(open(data_dest, 'rb+'))
    model = ModelZoo().vgg19(convolution_only=True)
    model.add(Convolution2D(2048, 1, 1, activation='relu'))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(2048, activation='relu'))

    create_folder(output_folder)
    file_path = os.path.join(output_folder, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
    training_optimization_params = Config.confDict[TRAINING_OPTIMIZATION_SECTION]

    # Optimizer
    sgd = SGD(lr=training_optimization_params['base_learning_rate'], momentum=0.9, nesterov=True)

    model.add(Dense(2, activation='softmax'))

    _schedual = functools.partial(schedual, learning_rate=training_optimization_params['base_learning_rate'],
                                  epoch_list=[20, 30, 40, 50, 60, 90])

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy', 'fmeasure'])
    model.fit_generator(generator=WholeProbeToyGenerator(probes_data, probes, 1),
                        samples_per_epoch=SAMPLES_PER_EPOCH,
                        nb_epoch=NUMBER_OF_EPOCHS,
                        callbacks=[ModelCheckpoint(filepath=file_path,
                                                   save_best_only=True),
                                   ProgbarLogger(),
                                   TensorBoard(log_dir=output_folder),
                                   LogEpochResults(),
                                   LearningRateScheduler(_schedual)],
                        validation_data=WholeProbeToyGenerator(probes_data, probes, 1),
                        nb_val_samples=len(probes))

    print 'end'
