import keras
import numpy as np

class ImageBatchLoader(keras.utils.Sequence):

    def __init__(self, image_ids, labels, training_parameters, imagePreprocessor):
        '''
        Writing a child implementation of keras.utils.Sequence will help us
        manage our batches of data.
        Each sequence must implement __len__ and __getitem__
        This structure guarantees that the network will only train once on each
        sample per epoch which is not the case with generators.

        We can use this class to instantiate training and validation generators
        that we can pass into our keras model like:

        training_generator = ImageBatchLoader(...)
        validation_generator = ImageBatchLoader(...)
        model.set_generators(training_generator, validation_generator)
        '''
        self.params = training_parameters
        self.image_ids = image_ids
        self.labels = labels

        # Helper classes
        self._imagePreprocessor = imagePreprocessor

        # Training parameters
        self.batch_size = self.params.batch_size
        self.dimensions = (self.params.row_dimension, self.params.col_dimension)
        self.n_channels = self.params.n_channels
        self.shuffle = self.params.shuffle

        # Run on_epoch_end in _init_ to init our first image batch
        self.on_epoch_end()

    def on_epoch_end(self):
        '''
        Tensorflow will run this method at the end of each epoch
        So this is where we will modify our batch.
        '''
        self.indexes = np.arrange(len(self.image_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        # Denotes the number of batchs per epoch
        return int(np.floor(len(self.image_ids) / self.batch_size))

    def __getitem__(self, index):
        # Get this batches indexes
        indexes = self.indexes[index * self.batch_size:(index+1) * self.batch_size]

        # Get cooresponding image Ids
        batch_image_ids = [self.image_ids[i] for i in indexes]

        # Generate one batch of data
        X, y = self.__generator(batch_image_ids)
        return X, y

    def __generator(self, batch_image_ids):

        def get_target_classes(id):
            # .loc will lookup the row where the passed in statement is true
            targets = self.labels.loc[self.labels.Id == id]
            targets = targets.drop(
                ["Id", "Target", "number_of_targets"], axis=1).values

            # returns a multi-hot encoded vector for all targets
            return targets

        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.num_classes), dtype=int)

        for index, id in enumerate(batch_image_ids):
            image = self._imagePreprocessor.load_image(id)
            image = self._imagePreprocessor.preprocess(image)

            X[index] = image
            y[index] = get_target_classes(id)

        return X, y
