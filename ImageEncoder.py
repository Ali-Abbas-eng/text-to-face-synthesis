import warnings
import torchsummary
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchmetrics import AUROC, Precision, Recall, Accuracy
from torch.optim.lr_scheduler import StepLR
import torch
import os
import seaborn as sns
import pandas as pd
import numpy as np

torch.manual_seed(0)  # Set for our testing purposes

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


def compute_class_weights(dataset_path: str = os.path.join('Data', 'cleaned_data.csv')):
    """
    a helper function to generate class weights based on the rare on major classes
    given the formula:
                                                                        Number of samples
                        weights of class (j) = ----------------------------------------------------------------
                                                Number of observations with class (j) * total number of classes
    :param dataset_path: str, the path to the dataset from which the weights will be calculated
    """

    # read the dataset
    dataset = pd.read_csv(dataset_path)

    # drop the image_id column
    dataset.drop('image_id', axis=1, inplace=True)

    # convert to numpy array
    dataset = dataset.to_numpy()

    # get number of observations
    n_samples = dataset.shape[0]

    # get number of classes
    n_classes = dataset.shape[1]

    # sum the number of classes over the observations (axis 0)
    class_frequencies = np.sum(dataset, axis=0)

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


class ImageEncoder:
    """
    Encapsulation of the Image Encoder model, builds, trains and downloads all the required stuff to get the model to
    function
    """

    def __init__(self,
                 file_path: str = os.path.join('models', 'ImageEncoder', 'ImageEncoder'),
                 device: str = 'cuda',
                 image_boarder: int = 64,
                 num_classes: int = 40) -> None:
        """
        initializer for the model, needs to know how many classes to account for in the final layer, plus the image size
        and whither or not to use the GPU (you'll need to manually check for CUDA availability), plus the path to which
        the trained model should be saved
        params
        :param file_path: str, the path to the file, on which the saved model will be saved,
                        default: 'models/ImageEncoder/ImageEncoder.pth'
        :param device: str, can be either 'cuda' to use gpu on training and inference or 'cpu', default: 'cuda'
        :param image_boarder: int, desired width and height of the input image (must be a square image), default: 64
        :param num_classes: int, number of classes the model needs to account for in the dataset, default 40 for celeba
        """
        # infer file directory
        directory = os.path.join(*file_path.split(os.path.sep)[:-1])

        # in case the director does not exist
        if not os.path.isdir(directory):
            # create it
            os.makedirs(directory)
        # set instance attributes
        self.file_path = file_path
        self.pytorch_implementation = True
        self.num_classes = num_classes
        self.device = device
        self.image_boarder = image_boarder

        # define pre-processing pipeline
        self.preprocessing_pipeline = transforms.Compose([
            transforms.Resize(self.image_boarder),
            transforms.CenterCrop(self.image_boarder),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.file_path = file_path + '.pth'
        if os.path.isfile(self.file_path):
            self.model = torch.load(self.file_path)
        else:
            from torchvision.models import resnet50, ResNet50_Weights
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.model = torch.nn.Sequential(self.model,
                                             nn.Dropout(.8),
                                             nn.Linear(in_features=1000, out_features=self.num_classes),
                                             nn.Sigmoid())

        self.model.eval()
        self.model.to(device)

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
        labels_indexes = indexes_of_interest

        # load training data
        train_data_loader = DataLoader(
            CelebA('Data', split='train', download=True, transform=self.preprocessing_pipeline),
            batch_size=batch_size,
            shuffle=True)

        # load validation data
        val_data_loader = DataLoader(
            CelebA('Data', split='valid', download=True, transform=self.preprocessing_pipeline),
            batch_size=batch_size,
            shuffle=True
        )

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
        metrics = [AUROC(pos_label=1, num_classes=self.num_classes, average='weighted'),
                   Accuracy()]

        # initialize training process
        cur_step = 0
        classifier_losses = []
        classifier_val_losses = []
        display_step = len(train_data_loader)
        postfix = ""
        class_weights = compute_class_weights()

        def get_scores(predictions, ground_truth):
            """
            a helper function that calculates evaluation metrics given the y_true, y_pred
            params
            :param predictions: torch.Tensor, model's output for some batch
            :param ground_truth: torch.Tensor, actual labels in the dataset for the same batch
            :returns: list of two elements [AUCROC score, Accuracy Score]
            """
            # call callable objects for each metric
            scores = [metrics[0](predictions, ground_truth),
                      metrics[1](predictions, ground_truth)]

            # return evaluation metrics
            return scores

        def validate(losses) -> str:
            """
            a helper function that calculates model's performance on the validation set
            params
            :param losses, list or torch.Tensor, the placeholder in which we'll keep saving results
            :returns: str, a message that tells us about the performance
            """
            val_loss = 0.
            val_auc_roc = 0.
            val_accuracy = 0.
            val_step = 0
            # loop through batches in the validation set
            for val_real, val_labels in val_data_loader:
                # move batch to the desired device
                val_real = val_real.to(self.device)

                # take the labels for which the model should account
                val_labels = val_labels[:, labels_indexes].to(self.device).float()

                # compute predictions on the current batch
                val_class_pred = self.model(val_real)

                # compute loss value for the current batch
                val_class_loss = criterion(val_class_pred, val_labels)

                # add the resulted loss to history (the placeholder)
                losses += [val_class_loss.item()]  # Keep track of the average classifier loss

                # compute scores for the current predictions
                val_scores_ = get_scores(val_class_pred.detach().cpu(), val_labels.type(torch.IntTensor))
                val_loss += val_class_loss.item()
                val_auc_roc += val_scores_[0]
                val_accuracy += val_scores_[1]
                val_step += 1
            # average class loss
            val_class_mean = sum(losses) / len(losses)

            # build the verbose message
            message = ""

            # add the metrics values to the indicator message
            message += f'Val Loss: {val_loss / val_step:3f}, ' \
                       f'Val AUROC: {val_auc_roc / val_step:3f}, ' \
                       f'Val Accuracy: {val_accuracy / val_step:3f}'

            # return info
            return message

        # loop through desired number of epochs
        for epoch in range(epochs):
            # set an epoch progress bar
            epoch_progress_bar = tqdm(train_data_loader, desc=f'Epoch: {epoch}', total=len(train_data_loader))
            train_loss = 0.
            train_auc_roc = 0.
            train_accuracy = 0.
            train_step = 0
            # Dataloader returns the batches
            for real, labels in epoch_progress_bar:

                # move batch to the desired device
                real = real.to(self.device)

                # take only the labels for which the model should account
                labels = labels[:, labels_indexes].to(self.device).float()

                # initialize gradients
                class_opt.zero_grad()

                # make predictions
                class_pred = self.model(real)

                # compute loss
                class_loss = criterion(class_pred, labels)

                # account for the imbalanced labels in the dataset
                class_loss = (class_loss * torch.Tensor(class_weights).cuda()).mean()

                # calculate the gradients
                class_loss.backward()

                # update the weights
                class_opt.step()

                # keep track of the classifier loss
                classifier_losses += [class_loss.item()]  # Keep track of the average classifier loss

                # compute evaluation metrics for the model w.r.t training data
                scores_ = get_scores(class_pred.detach().cpu(), labels.type(torch.IntTensor))
                train_auc_roc += scores_[0]
                train_accuracy += scores_[1]
                train_loss += class_loss.item()
                train_step += 1
                # in case training reached the stats updating interval
                if cur_step % stats_update_step == 0 and cur_step > 0:
                    # compute average classifier los
                    class_mean = sum(classifier_losses[-stats_update_step:]) / stats_update_step

                    # set indicating message
                    postfix = f'Loss: {train_loss / train_step:3f}, ' \
                              f'AUROC: {train_auc_roc / train_step:3f}, ' \
                              f'Accuracy: {train_accuracy / train_step:3f}'
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

    def infer(self,
              images: np.ndarray or torch.Tensor,
              return_logits: bool = True,
              data_format: str = 'channels_first'):
        """
        a helper function to classify new images
        :param images: np.ndarray or torch.Tensor, an array-like object of images to be classified
        :param return_logits: bool, set whither or not to return the probability of a class (True) or the values after
        thresholding (False), default True
        :param data_format: str, specify where to find the channels axis
        """
        self.model.eval()
        if type(images) != torch.Tensor():
            images = torch.Tensor(images)
        self.preprocessing_pipeline = transforms.Compose([
            transforms.Resize(self.image_boarder),
            transforms.CenterCrop(self.image_boarder),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        # pass the images in the pre-processing pipeline
        images = self.preprocessing_pipeline(images)

        # get the image encodings
        encodings = self.model(images.cuda())

        # return desired results
        if return_logits:
            return encodings
        else:
            return torch.sigmoid(encodings)


if __name__ == '__main__':
    with warnings.catch_warnings():
        file_path = os.path.join('models', 'ImageEncoder', 'ImageEncoder')
        warnings.simplefilter('ignore')
        model = ImageEncoder(image_boarder=64, file_path=file_path, num_classes=len(indexes_of_interest))
        model.train(batch_size=64, epochs=100, learning_rate=1e-6, decay=.1, decay_every=1, stats_update_step=5)
        model.infer(np.random.uniform(low=0., high=1., size=(3, 64, 64)))
