import numpy as np


class ModelParameters:

    def __init__(self,
                 training_data_path,
                 num_classes=28,
                 num_epochs=1,
                 batch_size=200,
                 image_rows=512,
                 image_cols=512,
                 row_scale_factor=4,
                 col_scale_factor=4,
                 n_channels=1,
                 shuffle=False):
        self.training_data_path = training_data_path
        self.num_classes = num_classes
        # what does n_epochs mean? it seems we pass this into the "epochs"
        # parameter on the "fit_generator" method on our keras model
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.row_dimension = np.int(image_rows / row_scale_factor)
        self.col_dimension = np.int(image_cols / col_scale_factor)
        self.n_channels = n_channels
        self.shuffle = shuffle
