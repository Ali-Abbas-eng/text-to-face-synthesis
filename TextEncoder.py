import json
import string
import warnings
import numpy as np
import pandas as pd
from utils import print_colored_text
from utils import block_print, enable_print, fetch_latest_check_point
import transformers.trainer_utils
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import EvalPrediction
from transformers import AutoModelForSequenceClassification
from TextDataHandler import TextDataHandler, get_mapping
import torch
from sklearn.metrics import roc_auc_score
import os
from nltk.stem import PorterStemmer
from datasets.dataset_dict import DatasetDict

os.environ["WANDB_DISABLED"] = "true"
sigmoid = torch.nn.Sigmoid()


def c_print(text, colour: str = 'green'):
    enable_print()
    print_colored_text(text, colour)
    block_print()


class TextEncoder:
    """
    this class takes the input from the TextDataHandler, encode it and then feed it to a torch model that will perform
    the training and will also save the weights
    """

    def __init__(self,
                 text_data_pipeline: TextDataHandler = None,
                 only_inference: bool = True,
                 verbose: bool = False,
                 truncation: bool = True,
                 max_length: int = 100,
                 num_train_epochs: int = 1,
                 batch_size: int = 32,
                 model_path: str = os.path.join('models', 'TextEncoder', 'BERTFineTuned')) -> None:
        """
        instance initializer, initialises all the parameters needed are the data instance and either or not you would
        like to print out indication statements about the process, for convenience, you can instantiate the class
        without passing any parameters the default values will work perfectly as planned (assuming you also ran the
        TextDataHandler with the default parameters (i.e. default path and filenames, batch size, ...etc)
        :param text_data_pipeline: TextDataHandler, an instance of the class to handle the text input to the model
        :param verbose: bool, set to true if you want to print indicating statements about the process, default false
        :param truncation: bool, set to False if you don't want sentences with length greater than max_length to be cut
        :param max_length: int, the maximum length (number of words) of a sentence at which sentences will be cut/padded
        """
        # assert the class is being used to train the ResNet50 or to use it for inference
        assert text_data_pipeline is not None or only_inference
        self.model_path = model_path
        self.verbose = verbose
        self.model = None
        self.threshold = None
        self.batch_size = batch_size
        self.epochs = num_train_epochs
        self.trainer = None
        self.stemmer = PorterStemmer()
        # get label <-> id mapping
        self.labels, self.id2label, self.label2id = get_mapping()
        self.max_length = max_length
        self.truncation = truncation
        mapping_arguments = {
            'text_column': 'description',
            'max_length': self.max_length,
            'truncation': self.truncation
        }

        # set tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', verbose=0)

        if text_data_pipeline is not None:
            # get training dataset
            self.train_dataset = text_data_pipeline.dataset['train']
            if self.verbose:
                c_print(f'Train Data was successfully uploaded to the encoder:\n\tExample: {self.train_dataset[0]}')

            # get testing dataset
            self.test_dataset = text_data_pipeline.dataset['test']
            if self.verbose:
                c_print(f'Test Data was successfully uploaded to the encoder:\n\tExample: {self.test_dataset[0]}')

            # get validation dataset
            self.val_dataset = text_data_pipeline.dataset['val']
            if self.verbose:
                c_print(f'Validation Data was successfully uploaded to the encoder:\n\tExample: {self.val_dataset[0]}')

            if self.verbose:
                c_print(f'Labels of the Dataset are: {self.labels}')

            if self.verbose:
                c_print(f'Tokenizer was successfully initiated')

            if self.verbose:
                c_print('Mapping datasets to the preprocess function', 'blue')

            # map data to the preprocessing function
            self.encoded_train_dataset = self.train_dataset.map(self.preprocess_data,
                                                                fn_kwargs=mapping_arguments,
                                                                batched=True,
                                                                remove_columns=self.train_dataset.column_names)
            self.encoded_test_dataset = self.test_dataset.map(self.preprocess_data,
                                                              fn_kwargs=mapping_arguments,
                                                              batched=True,
                                                              remove_columns=self.test_dataset.column_names)
            self.encoded_val_dataset = self.val_dataset.map(self.preprocess_data,
                                                            fn_kwargs=mapping_arguments,
                                                            batched=True,
                                                            remove_columns=self.val_dataset.column_names)

            if self.verbose:
                c_print('Mapping: Dataset <-> Preprocessing')

            if self.verbose:
                c_print('Initialising Model...', 'blue')

            # get the (pre-trained if possible) model
            self.load_model(only_inference=False)

            # initialise the trainer (to be used to train the model and/or predicting
            self.trainer_initialisation()
        else:
            self.truncation = True
            self.labels, self.id2label, self.label2id = get_mapping()
            self.load_model(True)
        if self.verbose:
            c_print('Done!')

    def preprocess_data(self,
                        examples: pd.DataFrame,
                        text_column: str = 'description',
                        truncation: bool = True,
                        max_length: int = 256) -> DatasetDict:
        """
        this function will process the input (batch) by converting the labels to two-dimensional array of the shape
        (batch_size, num_labels) and converting the text description to tokens with padding to the max length
        :param examples: pd.DataFrame, the dataset
        :param text_column: str, the name of the column which holds the textual data
        :param truncation: bool, specifies either or not to truncate sentences with length > max_length
        :param max_length: int, the maximum length of a sentence at which sentences will be truncated (or padded)
        :return: DatasetDict: an instance of DatasetDict which holds the encoded dataset
        """

        # take the column at which the textual description are located
        text = pd.Series(examples[text_column]).apply(self._preprocess).tolist()

        # encode the texts with the bert tokenizer using the passed arguments
        encoding = self.tokenizer(text, padding='max_length', truncation=truncation, max_length=max_length)

        # create a dictionary that holds the label at each label key for each example
        labels_batch = {k: examples[k] for k in examples.keys() if k in self.labels}

        # initialize a matrix that will hold the batch labels (start with zeros and will later be updated)
        labels_matrix = np.zeros((len(text), len(self.labels)))

        # for each label and its corresponding index in the columns that describes the labels of the examples
        for idx, label in enumerate(self.labels):
            # update the value at the same column index in the labels' matrix to have the same vector as in the
            # corresponding dictionary value
            labels_matrix[:, idx] = labels_batch[label]

        # add the labels key to the dataset dictionary with a value that is the labels matrix
        encoding['labels'] = labels_matrix.tolist()

        # return the final preprocessed batch
        return encoding

    def _preprocess(self, text: str) -> str:
        """
        preprocessing text step contains removing punctuation, and stemming each word in the text
        :param text: str, the text to be pre-processed
        returns: str, the preprocessed text
        """
        # remove punctuations
        text = "".join(char for char in text if char not in string.punctuation)
        # stem words
        text = " ".join(self.stemmer.stem(word=word) for word in text.split())

        # return the pre-processed text
        return text

    def compute_metrics(self,
                        arguments: EvalPrediction,
                        num_thresholds: int = 200) -> dict:
        """
        a function to compute the evaluation metrics of the model, this function implements the AUR-ROC metric for
        multi-class classification tasks, with 'num_threshold' different threshold, and gives and idea about the best
        threshold to be used in later inference
        :param arguments: the output of the model, MUST have 'predictions' and 'label_ids' attributes,
        note that the prediction are the logits (i.e., not yet activated by an activation function)
        :param num_thresholds: number of thresholds to test for, default = 200,
        return: dict, a dictionary which holds the maximum AUC-ROC for some threshold (the best threshold is saved to
        the class' attribute 'threshold'
        """
        # convert predictions to a pytorch acceptable format (i.e., torch.Tensor)
        predictions = torch.Tensor(arguments.predictions)

        # convert labels to a pytorch acceptable format (i.e., torch.Tensor)
        label_ids = torch.Tensor(arguments.label_ids)

        # set the default value of the winning threshold to .5
        winning_threshold = .5

        # starting (invalid) value of the AUC-ROC score
        max_auc = -1

        # create the list of thresholds as i/num_thresholds (for each threshold index i)
        threshold_list = [i * (1 / num_thresholds) for i in range(1, num_thresholds)]

        # loop through possible thresholds
        for i in range(len(threshold_list)):
            # get current threshold
            threshold = threshold_list[i]
            # predict the probabilities using sigmoid
            temp_y_pred = sigmoid(predictions)
            # set a placeholder for the predictions of the model
            temp_predictions = np.zeros(temp_y_pred.shape)
            # convert values at indexes to one if they satisfy the threshold
            temp_predictions[np.where(temp_y_pred > threshold)] = 1.

            try:
                # compute the current AUC score
                temp_score = roc_auc_score(y_true=label_ids,
                                           y_score=temp_predictions,
                                           average='macro')
            except ValueError:
                temp_score = 0

            # is it better?
            if temp_score > max_auc:
                # set the best threshold up until now
                winning_threshold = threshold
                # set the best score up until now
                max_auc = temp_score
        # save the best threshold to the class' attribute
        self.threshold = winning_threshold

        # return the computed value in the accepted name of the metric
        return {'AUC-ROC': max_auc}

    def trainer_initialisation(self,
                               evaluation_strategy: str = 'epoch',
                               save_strategy: str = 'epoch',
                               learning_rate: float = 2e-5,
                               weight_decay: float = .01,
                               load_best_model_at_end: bool = True,
                               metric_for_best_model: str = "AUC-ROC") -> None:
        """
        this is the function that will actually train the model for later inference
        :param evaluation_strategy: str, the strategy of the evaluation (at each epoch or step ...etc), default 'epoch'
        :param save_strategy: str, the model saving strategy (at each epoch or step ...etc), default 'epoch'
        :param learning_rate: float, the learning rate of the bert and the final layer, default 0.00002
        :param weight_decay: float, the value of the weight decay hyperparameter
        :param load_best_model_at_end: bool, specify either or not to load the best model at the end, to forcefully
        load the final model set it to False
        :param metric_for_best_model: str, the name of the metric which will evaluate how good/bad the model is, default
        "AUC-ROC"
        :return: None
        """

        # encapsulate the training parameters in a TrainingArgument instance
        training_arguments = TrainingArguments(evaluation_strategy=evaluation_strategy,
                                               save_strategy=save_strategy,
                                               output_dir=self.model_path,
                                               learning_rate=learning_rate,
                                               per_device_train_batch_size=self.batch_size,
                                               per_device_eval_batch_size=self.batch_size,
                                               num_train_epochs=self.epochs,
                                               weight_decay=weight_decay,
                                               load_best_model_at_end=load_best_model_at_end,
                                               report_to=None,
                                               metric_for_best_model=metric_for_best_model)

        # initialize a Trainer object that will finally train the model
        self.trainer = Trainer(
            self.model,
            training_arguments,
            train_dataset=self.encoded_train_dataset,
            eval_dataset=self.encoded_val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

    def train(self) -> None:
        """
        trains the model until the number of passed checkpoints reach the total number of training epochs
        :return: None
        """
        # check remaining epochs
        if self.epochs > 0.:
            # train the model
            self.trainer.train()
            # evaluate the model
            self.trainer.evaluate()
            # save the threshold's value to a json file (not a quality fix, but it'll do for now)
            with open(os.path.join(self.model_path, 'threshold.json'), 'w') as threshold_file:
                json.dump({'value': self.threshold}, threshold_file)

    def infer(self, text: str, return_probs: bool = True) -> np.ndarray:
        """
        inference function, takes a palin text input and outputs the labels as a torch.Tensor, the labels are one-hot
        vector with values equal to one at each index that satisfied the threshold condition
        :param text: str, plain text to be classified
        :param return_probs: bool, return the probability for each label (set to True), else returns one hot vector
        :return: np.ndarray
        """
        # perform data pre-processing on the given text
        text = self._preprocess(text)
        # tokenize the text
        tokenized_text = self.tokenizer(text, padding='max_length', truncation=self.truncation,
                                        max_length=self.max_length, return_tensors='pt')

        tokenized_text = {key: value for key, value in tokenized_text.items()}

        # get the logits
        prediction_output: transformers.trainer_utils.PredictionOutput = self.model(**tokenized_text)
        # activate logits with sigmoid
        predictions: torch.Tensor = torch.sigmoid(torch.Tensor(prediction_output[0]))
        # mark the indexes at which the threshold condition was satisfied
        label_ids = np.where(predictions >= .5, 1., 0.)[0]
        if return_probs:
            return label_ids
        else:
            # return the prolonged labels
            return predictions.detach().numpy()

    def load_model(self, only_inference: bool) -> None:
        """
        a helper function to load the model from the pre-trained models' directory, handles non-existence status
        :param only_inference: bool, if set to True the function raises a FileNotFoundException in case it couldn't find
        the saved model in the specified path
        :return: None
        """
        # get the most recent weights
        latest_checkpoint, total_checkpoints = fetch_latest_check_point(self.model_path, keyword='checkpoint')
        # if the result is something other than None then it's viable
        if latest_checkpoint is not None:
            if self.verbose:
                c_print(f'Found {total_checkpoints} checkpoints in total, loading latest...')
            # get the checkpoint path
            checkpoint_path = os.path.join(self.model_path, latest_checkpoint)
            # load the model from the defined path
            self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
            # subtract past epochs
            self.epochs -= total_checkpoints

            # check the resulted number of epochs
            if self.epochs <= 0:
                if self.verbose:
                    c_print(f'The models was trained for more epochs than the specified number of epochs, '
                            f'we assume that training is finished')
                # zero-out the training process
                self.epochs = 0.
            try:
                # get the best threshold from the checkpoints directory
                with open(os.path.join(self.model_path, 'threshold.json'), 'rb') as threshold_file:
                    # set the class' attribute
                    self.threshold = json.load(threshold_file)['value']

            except FileNotFoundError:
                self.threshold = 0.5

            if self.verbose:
                c_print(f'Model was Loaded SUCCESSFULLY')

        else:
            if not only_inference:
                # in case no checkpoints were found, set the model to the 'bert-base' model with a classification layer
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    'bert-base-uncased',
                    problem_type='multi_label_classification',
                    num_labels=len(self.labels),
                    id2label=self.id2label,
                    label2id=self.label2id)
            else:
                exit('No checkpoints were found, please make sure the path to the trained model is correctly defined,\n'
                     'or set only_inference parameter to False in the class init function to train a model '
                     'from scratch\notherwise the instance will not have data to be trained and/or model to infer from')
            if self.verbose:
                c_print('Loading Failed', 'red')


if __name__ == '__main__':
    input_pipeline = TextDataHandler()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        text_encoder = TextEncoder(text_data_pipeline=None,
                                   num_train_epochs=5,
                                   batch_size=32)

        # text_encoder.train()
        label_ids_ = text_encoder.infer("she's a beautiful black haired")
        labels = [text_encoder.id2label[idx] for idx, value in enumerate(label_ids_) if value == 1.]
        c_print(labels)
