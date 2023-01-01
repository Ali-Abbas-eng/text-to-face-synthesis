import gc
import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import make_text, train_test_val_split
from utils import print_colored_text as c_print
from datasets.dataset_dict import DatasetDict
from utils import block_print, enable_print, get_mapping


class TextDataHandler:
    """
    this class is meant to load the text dataset from local files (presumably CSV file), which can be fed to the
    TextEncoding model (almost directly)
    """

    def __init__(self,
                 cleaned_data_path: str = os.path.join('Data', 'cleaned_data.csv'),
                 train_data_file: str = os.path.join('Data', 'train_text_data.csv'),
                 test_data_file: str = os.path.join('Data', 'test_text_data.csv'),
                 val_data_file: str = os.path.join('Data', 'val_text_data.csv'),
                 num_descriptions: int = 10,
                 is_it_just_a_test: bool = False,
                 data_creation: bool = False,
                 verbose: bool = False):
        """
        initializer pass the arguments which specify where the cleaned data is stored and to which path the
        text-specific data will be saved
        :param cleaned_data_path: str, the path to the cleaned data file.
        :param train_data_file: str, the path to which the train data csv file will be saved
        :param test_data_file: str, the path to which the test data csv file will be saved
        :param val_data_file: str, the path to which the validation data csv file will be saved
        :param data_creation: bool, in case this is the first time to call the class set to true to generate the files
        :param verbose: bool, set to true to print indicating statements at each step
        :param num_descriptions: int, number of description sentences to generate given the labels
        :param is_it_just_a_test: bool, set to true to load just 1% of the dataset
        """
        # set class' attributes
        self.data_df = pd.read_csv(cleaned_data_path)  # read the cleaned data file.
        self.train_data_path = train_data_file  # store the train data path as an attribute
        self.test_data_path = test_data_file  # store the test data path as an attribute
        self.val_data_path = val_data_file  # store the validation data path as an attribute
        self.verbose = verbose  # store the verbose argument as an attribute for later use
        self.num_descriptions = num_descriptions
        self.total_progress = 0
        self.is_it_just_a_test = is_it_just_a_test  # set the "is_it_just_a_test" property

        # if this execution was meant to be a test of the input pipeline and/or the NLP model performance
        if self.is_it_just_a_test:
            self.train_data_path = self.train_data_path[:-4] + '_trial.csv'
            self.test_data_path = self.test_data_path[:-4] + '_trial.csv'
            self.val_data_path = self.val_data_path[:-4] + '_trial.csv'
            self.data_df = self.data_df.iloc[:self.data_df.shape[0] // 1000, :]

        # drop the 'image_id' column since it's of no use with the NLP task
        self.data_df.drop('image_id', axis=1, inplace=True)

        # in case the user wishes to create/re-create the data
        if data_creation:
            # create the description column which is the column name + " " for each column where the value is 1
            self.data_df['description'] = ["" for _ in range(self.data_df.shape[0])]
            self.placeholder = pd.DataFrame(np.empty(shape=[self.data_df.shape[0] * self.num_descriptions,
                                                            self.data_df.shape[1]]),
                                            columns=self.data_df.columns)

            self._create_text_data()  # call the function which generates the files

        self.labels, self.id2label, self.label2id = get_mapping(cleaned_data_path)

        block_print()
        # initialize a DatasetDict object which holds the three splits of the dataset
        self.dataset: DatasetDict = DatasetDict.from_csv({
            'train': self.train_data_path,
            'test': self.test_data_path,
            'val': self.val_data_path},
            verbose=self.verbose)
        enable_print()

        if self.verbose:
            # take one row of the dataset to view
            example = self.dataset['train'][0]
            c_print(f'Dataset Example: {example}', 'green')

    def _create_text_data(self) -> None:
        """
        this is a helper function that will create the text-specific data (utilizing the function make_text from utils)
        will generate a text description of the image, hence the labels which indicates the presence or absence of
        a facial feature will also be the labels of the input text
        :return: None.
        """

        self.create_new_dataset()
        # drop the rows with no descriptions
        self.data_df = pd.DataFrame(self.placeholder, columns=self.data_df.columns)
        del self.placeholder
        gc.collect()
        print(self.data_df.head())
        if self.verbose:
            c_print(f'The first 10 rows of the text specific dataset are:\n{self.data_df.head(10)}', 'green')

        # call the function which is responsible for the final step of the data preparing
        self._initiate_and_save()

    def _initiate_and_save(self) -> None:
        """
        this is a helper function that will split the dataset into three splits (train, test, and validation splits),
        and save them to the hard drive, once this function is executed successfully you no longer need to set the
        'data_creation' argument to True
        :return: None
        """

        # call the utils.train_test_val_split which will return a dictionary containing
        # (train and test splits with (optionally) a validation split
        data_dictionary = train_test_val_split(self.data_df)

        # save the value of the train key (i.e. the training data set) to a single csv file
        data_dictionary['train'].to_csv(self.train_data_path,
                                        index_label=False,
                                        index=False)

        if self.verbose:
            c_print('Training data was successfully saved to the disk', 'green')
        # save the value of the test key (i.e. the test data set) to a single csv file
        data_dictionary['test'].to_csv(self.test_data_path,
                                       index_label=False,
                                       index=False)

        if self.verbose:
            c_print('Testing data was successfully saved to the disk', 'green')
        # save the value of the val key (i.e. the validation data set) to a single csv file
        data_dictionary['val'].to_csv(self.val_data_path,
                                      index_label=False,
                                      index=False)

        if self.verbose:
            c_print('Validation data was successfully saved to the disk', 'green')

    def create_new_dataset(self):
        """
        generates 'num_description' descriptions for each row in the dataframe
        :return: None
        """
        # ignore the deprecation warning (just one time please)
        with warnings.catch_warnings():
            # set the ignore filter
            warnings.simplefilter('ignore')

            # loop over each row of the DataFrame
            for _, original_row in tqdm(self.data_df.iterrows(), total=self.data_df.shape[0]):
                # generate the new permutation of texts
                texts = make_text(original_row[:40], num_descriptions=self.num_descriptions)
                # generate the new rows
                new_rows = [[*original_row[:40], text] for text in texts]

                # loop over the new rows
                for new_row in new_rows:
                    # set the new row in its corresponding place in the new DataFrame
                    self.placeholder.loc[self.total_progress] = new_row
                    self.total_progress += 1


if __name__ == '__main__':
    text_data_pipeline_object = TextDataHandler(is_it_just_a_test=True, data_creation=False, verbose=True)
