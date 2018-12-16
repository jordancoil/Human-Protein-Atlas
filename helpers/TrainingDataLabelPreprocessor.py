import pandas as pd


class TrainingDataLabelPreprocesser:

    def __init__(self, csv_path):
        self.training_data = pd.read_csv('train/train.csv')

        # Create a dictionary assigning labels to each
        # of the 28 cell functions/locations
        self.label_names = {
            0:  "Nucleoplasm",
            1:  "Nuclear membrane",
            2:  "Nucleoli",
            3:  "Nucleoli fibrillar center",
            4:  "Nuclear speckles",
            5:  "Nuclear bodies",
            6:  "Endoplasmic reticulum",
            7:  "Golgi apparatus",
            8:  "Peroxisomes",
            9:  "Endosomes",
            10:  "Lysosomes",
            11:  "Intermediate filaments",
            12:  "Actin filaments",
            13:  "Focal adhesion sites",
            14:  "Microtubules",
            15:  "Microtubule ends",
            16:  "Cytokinetic bridge",
            17:  "Mitotic spindle",
            18:  "Microtubule organizing center",
            19:  "Centrosome",
            20:  "Lipid droplets",
            21:  "Plasma membrane",
            22:  "Cell junctions",
            23:  "Mitochondria",
            24:  "Aggresome",
            25:  "Cytosol",
            26:  "Cytoplasmic bodies",
            27:  "Rods & rings"
        }

        self.label_names_reversed_keys = dict((value, key) for key, value in label_names.items())

        # run preprocessing on init
        self.multi_hot_encode()
        self.apply_number_of_targets_col()

    def fill_targets(row):
        # Multi hot encode the training data and correct responses
        row.Target = np.array(row.Target.split(" ")).astype(np.int)
        for num in row.Target:
            name = self.label_names[int(num)]
            row.loc[name] = 1
        return row

    def multi_hot_encode():
        for key in self.label_names.keys():
            self.training_data[self.label_names[key]] = 0

        self.training_data = self.training_data.apply(fill_targets, axis=1)

    def apply_number_of_targets_col():
        self.training_data["number_of_targets"] = train_labels.drop(
            ["Id", "Target"],axis=1).sum(axis=1)
