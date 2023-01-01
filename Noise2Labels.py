import gc
import warnings

import torch
from torch import nn
from tqdm import tqdm
from torchmetrics import AUROC, Precision, Recall, Accuracy
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np
import pandas as pd
import utils


def c_print(text: str, color: str = 'green'):
    utils.enable_print()
    utils.print_colored_text(text, color)
    utils.block_print()


def orthogonalize(X, vectors=True, norm=True):
    if not vectors:
        X = X.T
    Y = X[0:1, :].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i, :].dot(Y.T) / np.linalg.norm(Y, axis=1) ** 2).flat).dot(Y)
        Y = np.vstack((Y, X[i, :] - proj.sum(0)))
    if norm:
        Y = np.diag(1 / np.linalg.norm(Y, axis=1)).dot(Y)
    if vectors:
        return Y
    else:
        return Y.T


class Noise2Labels:
    def __init__(self,
                 z_dim: int = 512,
                 num_classes: int = 40,
                 data_path: str = os.path.join('Data', 'Generated Dataset', 'Generated Classified Data Info.json'),
                 model_path: str = os.path.join('models', 'Noise2Labels', 'Noise2Labels.pth'),
                 verbose: bool = False):
        self.data_path = data_path
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.verbose = verbose
        if os.path.isfile(model_path):
            if self.verbose:
                c_print('model was found in specified directory, and is getting loaded')
            self.model = torch.load(model_path)
        else:
            if self.verbose:
                c_print('model was not found in specified directory, training from scratch', 'red')
            self.model = self.get_model(model_path=model_path, data_path=data_path)
        self.weights = self.model.weight.detach().cpu().numpy()

    def get_model(self,
                  batch_size: int = 4096,
                  epochs: int = 1000,
                  learning_rate: float = .001,
                  decay: float = 0.1,
                  decay_every: int = 250,
                  model_path: str = os.path.join('models', 'Noise2Labels', 'Noise2Labels.pth'),
                  data_path: str = os.path.join('Data', 'Generated Dataset', 'Generated Classified Data Info.json')
                  ) -> torch.nn.Module:

        model_directory = os.path.join(*model_path.split(os.path.sep)[:-1])
        if not os.path.isdir(model_directory):
            os.makedirs(model_directory)

        data_df = pd.read_json(data_path)

        data_splits = utils.train_test_val_split(data_df, train_proportion=.8, return_validation=False)
        train_data = data_splits['train']
        valid_data = data_splits['test']

        train_x = np.zeros(shape=[train_data.shape[0], self.z_dim])
        train_y = np.zeros(shape=[train_data.shape[0], self.num_classes])

        for i in range(train_data.shape[0]):
            train_x[i] += np.array(train_data.iloc[i, 0])
            train_y[i] += np.array(train_data.iloc[i, 3])

        valid_x = np.zeros(shape=[valid_data.shape[0], self.z_dim])
        valid_y = np.zeros(shape=[valid_data.shape[0], self.num_classes])

        num_batches_val = valid_x.shape[0] // batch_size + int(valid_x.shape[0] % batch_size != 0)
        for i in range(valid_data.shape[0]):
            valid_x[i] += np.array(valid_data.iloc[i, 0])
            valid_y[i] += np.array(valid_data.iloc[i, 3])

        model = nn.Sequential(nn.Linear(in_features=self.z_dim, out_features=self.num_classes, device='cuda'),
                              nn.Sigmoid())

        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
        criterion = torch.nn.BCELoss()
        learning_rate_scheduler = StepLR(optimizer=optimizer, step_size=decay_every, gamma=decay, last_epoch=-1)
        num_batches_train = train_data.shape[0] // batch_size + int(train_data.shape[0] % batch_size != 0)
        progress_bar = tqdm(range(epochs), desc='Training')
        class_losses = []

        # # sum the number of classes over the observations (axis 0)
        # class_frequencies = np.sum(train_y, axis=0)
        #
        # # compute class weights
        # class_weights = train_y.shape[0] / (train_y.shape[1] * class_frequencies)

        del train_data
        gc.collect()

        def compute_start_end_indexes(idx, sample_size, n_samples, num_batches):
            start_index = idx * sample_size
            end_index = idx * sample_size + (batch_size
                                             if index < num_batches - 1
                                             else n_samples % sample_size)
            return start_index, end_index

        def get_scores(predictions, ground_truth):
            """
            a helper function that calculates evaluation metrics given the y_true, y_pred
            params
            :param predictions: torch.Tensor, model's output for some batch
            :param ground_truth: torch.Tensor, actual labels in the dataset for the same batch
            :returns: list of two elements [AUCROC score, Accuracy Score]
            """
            # call callable objects for each metric
            scores = [AUROC(pos_label=1, num_classes=40, average='weighted')(predictions, ground_truth),
                      Accuracy()(predictions, ground_truth)]

            # return evaluation metrics
            return scores

        def validate(losses) -> str:
            """
            a helper function that calculates model's performance on the validation set
            params
            :param losses, list or torch.Tensor, the placeholder in which we'll keep saving results
            :returns: str, a message that tells us about the performance
            """
            for idx in range(num_batches_val):
                start_idx, end_idx = compute_start_end_indexes(idx, batch_size, valid_x.shape[0], num_batches_val)
                val_x = torch.Tensor(valid_x[start_idx:end_idx, :]).cuda()
                val_y = torch.Tensor(valid_y[start_idx:end_idx, :]).cuda()
                # compute predictions on the current batch
                val_pred = model(val_x)

                # compute loss value for the current batch
                val_class_loss = criterion(val_pred, val_y)

                # add the resulted loss to history (the placeholder)
                losses += [val_class_loss.item()]  # Keep track of the average classifier loss

                # compute scores for the current predictions
                val_scores_ = get_scores(val_pred.detach().cpu(), val_y.type(torch.IntTensor))

            # average class loss
            val_class_mean = sum(losses) / len(losses)

            # build the verbose message
            message = ""

            # add the metrics values to the indicator message
            message += f'Val Loss: {val_class_mean:3f}, ' \
                       f'Val AUROC: {val_scores_[0]:3f}, ' \
                       f'Val Accuracy: {val_scores_[1]:3f}'

            # return info
            return message

        postfix = ""
        for i in progress_bar:
            for index in range(num_batches_train):
                start, end = compute_start_end_indexes(index, batch_size, train_x.shape[0], num_batches_train)
                batch_x = torch.Tensor(train_x[start:end, :]).cuda()
                batch_y = torch.Tensor(train_y[start:end, :]).cuda()
                optimizer.zero_grad()
                y_pred = model(batch_x)
                loss = criterion(y_pred, batch_y)
                loss.backward()
                optimizer.step()
                train_scores = get_scores(y_pred.detach().cpu(), batch_y.type(torch.IntTensor))
                postfix = f'Loss: {loss:3f}, ' \
                          f'AUROC: {train_scores[0]:3f}, ' \
                          f'Accuracy: {train_scores[1]:3f}'
                info = validate([])
                progress_bar.set_postfix_str(postfix + ' || ' + info)
            learning_rate_scheduler.step()
        with torch.no_grad():
            weights = list(model.children())[0].weight.detach().cpu().numpy()
            weights = orthogonalize(weights)
            self.weights = weights
            model.weight = torch.Tensor(weights)

        torch.save(model, model_path)
        return model

    def infer(self, z, initial_labels, desired_labels):
        movement_vector = initial_labels - desired_labels
        z_hat = z + np.dot(movement_vector, self.weights)
        return z_hat


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Noise2Labels(verbose=True)
