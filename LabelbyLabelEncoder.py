import gc
import warnings
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
import matplotlib.pyplot as plt
from torchmetrics import AUROC, Precision, Recall, Accuracy
from torch.optim.lr_scheduler import StepLR
import torch
import os
import seaborn as sns
import pandas as pd
import numpy as np

facial_features = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
                   'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
                   'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
                   'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
                   'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
                   'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
                   'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
                   'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                   'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
                   'Wearing_Necktie', 'Young']
facial_features = np.array(facial_features, dtype=object)

indexes_of_interest = [4, 5, 9, 10, 13, 14, 15, 16, 17, 20, 22, 26, 30]

# torch.manual_seed(0)  # Set for our testing purposes
def compute_class_weights(dataset_path: str = os.path.join('Data', 'train_data.csv'),
                          column_of_interest: int = None):
    """
    a helper function to generate class weights based on the rare on major classes
    given the formula:
                                                                        Number of samples
                        weights of class (j) = ----------------------------------------------------------------
                                                Number of observations with class (j) * total number of classes
    :param dataset_path: str, the path to the dataset from which the weights will be calculated
    """

    # read the dataset
    data_df = pd.read_csv(dataset_path)

    data_df.drop('image_id', axis=1, inplace=True)

    # convert to numpy array
    data_df = data_df.to_numpy()

    # get number of observations
    n_samples = data_df.shape[0]

    # get number of classes
    n_classes = data_df.shape[1]

    # sum the number of classes over the observations (axis 0)
    class_frequencies = np.sum(data_df[:, column_of_interest], axis=0)

    # compute class weights
    class_weights = n_samples / (n_classes * class_frequencies)
    return class_weights


def show_tensor_images(image_tensor, num_images=16, size=(3, 64, 64), nrow=3):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    """
    # convert image from the dynamic range [-1, 1] to the plot-able [0, 1]
    image_tensor = (image_tensor + 1) / 2

    # convert to tensor on RAM
    image_unflat = image_tensor.detach().cpu()

    # create a grid (a placeholder for the images)
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)

    # show the image in the grid
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


class OneLabelData(Dataset):
    def __init__(self,
                 data_csv_file: str = os.path.join('Data', 'train_data.csv'),
                 column_of_interest: str = 'male',
                 images_directory: str = os.path.join('Data', 'faces (1.0 ratio)'),
                 target_size: int = 150):
        self.data_df = pd.read_csv(data_csv_file)
        for column in self.data_df.columns[1:]:
            if column != column_of_interest:
                try:
                    self.data_df.drop(column, inplace=True, axis=1)
                except KeyError:
                    pass
                except IndexError:
                    pass
        self.images_directory = images_directory
        self.target_size = target_size

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, index):
        image_id = self.data_df.iloc[index, 0]
        image_path = os.path.join(self.images_directory, image_id)
        image = read_image(image_path).float()
        image = transforms.Resize(self.target_size)(image)
        image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)
        label = self.data_df.iloc[index, 1]
        return image, label


class Classifier:
    def __init__(self,
                 class_of_interest,
                 file_path: str = os.path.join('models', 'ImageEncoder'),
                 image_channels: int = 3,
                 hidden_dim: int = 64,
                 num_classes: int = 1,
                 device: str = 'cuda'):
        self.num_classes = num_classes
        self.device = device
        self.feature = class_of_interest
        class_position = np.where(facial_features == class_of_interest)[0]
        if class_position.size == 0:
            exit('The specified feature to look for in the dataset cannot be found')

        self.class_id = class_position[0]
        self.file_path = os.path.join(file_path, f'{self.class_id}.pth')
        if os.path.isfile(self.file_path):
            self.model = torch.load(self.file_path)
            self.model.eval()
        else:
            self.model = nn.Sequential(
                self.make_classifier_block(image_channels, hidden_dim),
                self.make_classifier_block(hidden_dim, hidden_dim * 2),
                self.make_classifier_block(hidden_dim * 2, hidden_dim * 4, stride=3),
                self.make_classifier_block(hidden_dim * 4, hidden_dim * 4, stride=3),
                self.make_classifier_block(hidden_dim * 4, num_classes, final_layer=True),
            )

    @staticmethod
    def make_classifier_block(input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a classifier block;
        a convolution, a batchnorm (except in the final layer), and an activation (except in the final layer).
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise
                      (affects activation and batchnorm)
        '''
        if final_layer:
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=2304, out_features=output_channels),
                nn.Sigmoid()
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )

    def train(self,
              batch_size: int = 256,
              epochs: int = 50,
              learning_rate: float = .001,
              decay: float = 0.1,
              decay_every: int = 5,
              stats_update_step: int = 10) -> None:
        """
        a helper function to train the model
        params
        :param batch_size: int, number of instances in one step of the training, default = 256
        :param epochs: int, number of complete passes through the dataset
        :param learning_rate: float, hyperparameter, step size in the training process
        :param decay: float, a scaling factor by which the learning rate will be multiplied during training, default .1
        :param decay_every: int, number of epochs between each decay step, default 5
        :param stats_update_step: int, number of steps between each calculation of the loss and metrics, default 10
        """

        # get the indexes of the labels based on the number of classes to account for
        labels_indexes = range(self.num_classes)

        # load training data
        train_data_loader = DataLoader(OneLabelData(column_of_interest=self.feature), batch_size=batch_size,
                                       shuffle=True)

        # load validation data
        val_data_loader = DataLoader(
            OneLabelData(data_csv_file=os.path.join('Data', 'test_data.csv'), column_of_interest=self.feature),
            batch_size=batch_size // 2,
            shuffle=False)

        # set optimizer hyper parameters
        beta_1 = 0.5
        beta_2 = 0.999

        # set model's optimizer
        class_opt = torch.optim.Adam(self.model.parameters(),
                                     lr=learning_rate,
                                     betas=(beta_1, beta_2))

        # move the model to the desired device
        self.model.to(self.device)

        # set functionality to calculate loss function
        criterion = nn.BCELoss()

        # set decay schedule for the learning rate
        learning_rate_scheduler = StepLR(optimizer=class_opt, step_size=decay_every, gamma=decay, last_epoch=-1)

        # define functionalities to compute evaluation metrics for the model
        metric = Accuracy()

        # initialize training process
        cur_step = 0
        classifier_losses = []
        classifier_val_losses = []
        display_step = len(train_data_loader)
        postfix = ""
        class_weights = compute_class_weights(column_of_interest=self.class_id)

        def validate(losses) -> str:
            """
            a helper function that calculates model's performance on the validation set
            params
            :param losses, list or torch.Tensor, the placeholder in which we'll keep saving results
            :returns: str, a message that tells us about the performance
            """
            val_scores_ = 0.
            val_steps = 0
            # loop through batches in the validation set
            for val_real, val_labels in val_data_loader:
                # move batch to the desired device
                val_real = val_real.to(self.device)

                # take the labels for which the model should account
                val_labels = val_labels.to(self.device).float().view(-1, 1)

                # compute predictions on the current batch
                val_class_pred = self.model(val_real)

                # compute loss value for the current batch
                val_class_loss = criterion(val_class_pred, val_labels)

                # add the resulted loss to history (the placeholder)
                losses += [val_class_loss.item()]  # Keep track of the average classifier loss

                # compute scores for the current predictions
                val_scores_ += metric(val_class_pred.detach().cpu(), val_labels.type(torch.IntTensor))
                val_steps += 1

            # average class loss
            val_class_mean = sum(losses) / len(losses)

            # build the verbose message
            message = ""

            # add the metrics values to the indicator message
            message += f'Val Loss: {val_class_mean:.3f}, ' \
                       f'Val Accuracy: {val_scores_ / val_steps:.3f}'

            # return info
            return message

        # loop through desired number of epochs
        for epoch in range(epochs):
            # set an epoch progress bar
            epoch_progress_bar = tqdm(train_data_loader, desc=f'Epoch: {epoch}', total=len(train_data_loader))

            train_scores = 0.
            train_steps = 0
            # Dataloader returns the batches
            for real, labels in epoch_progress_bar:

                # move batch to the desired device
                real = real.to(self.device)

                # take only the labels for which the model should account
                labels = labels.to(self.device).float().view(-1, 1)

                # initialize gradients
                class_opt.zero_grad()

                # make predictions
                class_pred = self.model(real)

                # compute loss
                class_loss = criterion(class_pred, labels)

                # account for the imbalanced labels in the dataset
                class_loss = (class_loss * torch.Tensor([class_weights]).cuda()).mean()

                # calculate the gradients
                class_loss.backward()

                # update the weights
                class_opt.step()

                # keep track of the classifier loss
                classifier_losses += [class_loss.item()]  # Keep track of the average classifier loss

                # compute evaluation metrics for the model w.r.t training data
                train_scores += metric(class_pred.detach().cpu(), labels.type(torch.IntTensor))

                train_steps += 1
                # in case training reached the stats updating interval
                if cur_step % stats_update_step == 0 and cur_step > 0:
                    # compute average classifier los
                    class_mean = sum(classifier_losses[-stats_update_step:]) / stats_update_step

                    # set indicating message
                    postfix = f'Loss: {class_mean:.6f}, ' \
                              f'Accuracy: {train_scores / train_steps:.6f}'
                    epoch_progress_bar.set_postfix_str(postfix)

                # in the end of the epoch (where the display step condition is satisfied
                if (cur_step + 1) % display_step == 0 and cur_step > 0:
                    # compute the model's statistics on the validation set, and get the info message
                    info = validate(classifier_val_losses)

                    # update the progress bar to hold the new validation message
                    epoch_progress_bar.set_postfix_str(postfix + ' || ' + info)

                    # set number of bins in the plot
                    step_bins = 20

                    # plot the training and test losses to the screen
                    x_axis = sorted([i * step_bins for i in range(len(classifier_val_losses) // step_bins)] * step_bins)
                    sns.lineplot(x_axis, classifier_losses[:len(x_axis)], label="Train Loss")
                    sns.lineplot(x_axis, classifier_val_losses[:len(x_axis)], label="Valid Loss")
                    plt.legend()
                    plt.show()
                cur_step += 1

            # save the model with respect to the current epoch
            torch.save(self.model, self.file_path)

            # take a learning rate decay step
            learning_rate_scheduler.step()


def global_forward(image):
    if not type(image) == torch.Tensor():
        image = torch.Tensor(image)
    image = transforms.Resize(150)(image)
    image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)
    image_encodings = np.zeros(shape=[image.shape[0], len(indexes_of_interest)])
    for index, i in enumerate(indexes_of_interest):
        model = torch.load(os.path.join('models', 'ImageEncoder', f'{i}.pth'), map_location='cuda')
        model.eval()
        temp = model(image.cuda()).detach().cpu().numpy()
        image_encodings[:, index] = temp
    del model
    gc.collect()
    return image_encodings


# if __name__ == '__main__':
    # with warnings.catch_warnings():
    #     gaps = [36, 37, 38, 39]
    #     warnings.simplefilter('ignore')
    #     for index, feature in enumerate(facial_features):
    #         if index in gaps:
    #             print(f'Training Classifier on Label #{index}: {feature} ')
    #             classifier = Classifier(feature)
    #             classifier.train(batch_size=64, epochs=5)
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    image = plt.imread(os.path.join('Data', 'Generated Dataset', 'images', '000002.jpg'))
    image = image[np.newaxis, :, :, :]
    image = np.transpose(image, [0, 3, 1, 2])
    image_embeddings = global_forward(image)
    plt.imshow(np.transpose(image, [0, 2, 3, 1])[0])
    plt.show()
    dictionary = {facial_features[i]: image_embeddings[0][index] > .5 for index, i in enumerate(indexes_of_interest)}
    print(dictionary)
