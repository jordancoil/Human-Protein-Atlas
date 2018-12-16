import unittest
import pandas as pd

from helpers.ModelParameters import ModelParameters
from helpers.TrainingDataLabelPreprocessor import TrainingDataLabelPreprocessor
from helpers.ImageBatchLoader import ImageBatchLoader


class TrainingDataLabelPreprocessorTestCase(unittest.TestCase):

    def setUp(self):
        data = {"Id": pd.Series(["image_id_1", "image_id_1"], index=['1', '2']),
                "Target": pd.Series(["16 0 2", "7"], index=['1', '2'])}
        training_data = pd.DataFrame(data,)
        self.preprocessor = TrainingDataLabelPreprocessor(training_data)


if __name__ == '__main__':
    unittest.main()
