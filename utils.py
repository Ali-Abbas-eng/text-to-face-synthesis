import os
import torch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import pandas as pd
from termcolor import colored
import sys

facial_features = [feature.lower().replace('_', " ")
                   for feature in ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
                                   'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
                                   'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
                                   'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
                                   'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
                                   'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
                                   'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
                                   'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                                   'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
                                   'Wearing_Necktie', 'Young']]
facial_features = np.array(facial_features, dtype=object)
neighbour_values = list(range(1, 12))
scale_values = [float(f"1.{x}") for x in range(1, 10)] + [float(f"2.{x}") for x in range(1, 10)]
# pairs = [(x, y) for x in neighbour_values for y in scale_values]
# original_pairs = pairs.copy()
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def block_print() -> None:
    """
    disables printing all at once (to prevent codes that we're sure work the way we want from printing statements)
    :return: None
    """
    sys.stdout = open(os.devnull, 'w')


def enable_print() -> None:
    """
    enables printing
    :return: None
    """
    sys.stdout = sys.__stdout__


def print_colored_text(text, color: str = 'green') -> None:
    """
    functionality for printing colored text with minimal statements
    :param text: str, text to be printed
    :param color: str, colour of the text
    :return: None
    """
    print(colored(text, color))


def download_data(path,
                  url: str = None) -> None:
    """
    downloads the pickled model from a specific url
    :param path: str, path to the file to which we want to download the pickled file
    :param url: str, link to the pickled model
    :return: None
    """
    import requests
    try:
        # get the file (move to RAM)
        response = requests.get(url)
        # write the file to path
        open(path, "wb").write(response.content)
    except requests.RequestException as exception:
        print_colored_text(str(exception), 'red')


def clean_data(dirty_data_path: str = os.path.join('Data', 'list_attr_celeba.txt'),
               cleaned_data_path: str = os.path.join('Data', 'cleaned_data.csv')) -> None:
    """
    a helper function to convert the data info from the dirty text to a data frame (to CSV file)
    :param dirty_data_path: str, the path to the text file at which data info is stored
    :param cleaned_data_path: str, the path to which we want to save the cleaned data
    :return: None
    """
    temp_handler = open(dirty_data_path)  # a temp handle to the text file
    texts = temp_handler.read().split("\n")  # convert to a list (each line is an element)
    temp_handler.close()  # close the handler
    columns = np.array(texts[1].split(" "))  # ignore first row (which is the number of images)
    columns = columns[columns != ""]  # remove non-words from column names
    df = []  # store attribute values
    for txt in texts[2:]:
        txt = np.array(txt.split(" "))  # the values in each row are separated by a space " "
        txt = txt[txt != ""]  # in case of zero length values caused by str.split function
        df.append(txt)  # add the numpy.ndarry to the list df that holds attribute values
    df = pd.DataFrame(df)  # convert the list to a DataFrame (there are no column names right now)
    if df.shape[1] == len(columns) + 1:
        columns = ["image_id"] + list(columns)  # add image_id (i.e. images' paths) to the column names list

    df.columns = columns  # name the columns in the DataFrame
    df = df.dropna()  # remove N/A values
    for column in df.columns:
        if column not in "image_id":
            df[column] = pd.to_numeric(df[column], downcast="integer")  # convert 1s and -1s to actual integers
    df.replace(-1, 0, inplace=True)  # replace -1s with zeros (for the classification process)

    # save the final data to the desired path
    df.to_csv(cleaned_data_path,
              index_label=False,
              index=False)


def plot_images(images: tf.Tensor or torch.Tensor or np.ndarray,
                labels: np.ndarray or list = None,
                data_format: str = 'channels_last',
                rows: int = None,
                cols: int = None,
                total_title: str = None) -> None:
    """
    plots images in a figure of shape n * m
    :param data_format: str, either 'channels_first' or 'channels_last' determines at which dimensions the channels are
    :param images: Tensor, images to plot
    :param labels: Tensor, labels of the images (facial features)
    :param rows: integer, number or rows (n)
    :param cols: integer, number of columns (m)
    :param total_title: str, the title for the whole plot with rows * cols images
    :return: None
    """
    if data_format == 'channels_first':
        try:
            images = tf.transpose(images, [0, 2, 3, 1])
        except:
            images = np.transpose(images, [0, 2, 3, 1])

    if rows is None and cols is None:
        rows = int(np.sqrt(images.shape[0]))
        if rows == 0:
            plt.imshow(images)
            plt.title(labels[0])
            return
        cols = images.shape[0] // rows + int(images.shape[0] % rows != 0)

    elif rows is None and cols is not None:
        rows = images.shape[0] // cols + int(images.shape[0] % cols != 0)

    else:
        cols = images.shape[0] // rows + int(images.shape[0] % rows != 0)

    fig_size = images.shape[1] // 12

    # set the figure of matplotlib.pyplot to hold rows * columns subplots
    fig, axs = plt.subplots(rows, cols, figsize=(fig_size, fig_size))  # create a placeholder for the plots

    if total_title is not None:
        plt.suptitle(total_title, fontsize='xx-large')

    if len(images.shape) == 3:
        plt.imshow(images)
        if total_title is not None:
            plt.title(total_title)
        plt.show()
        return

    for i in range(images.shape[0]):  # for each image do
        # compute the current row the subplot (the result of integer division (iteration number / total number of rows)
        row = i // cols

        # the column of the current subplot is the mod result between current iteration and total number of columns
        col = i % cols

        if rows != 1:
            # plot the image to the current subplot
            axs[row, col].imshow(images[i, :, :, :])
            # set the title of the current subplot
            if labels is not None:
                axs[row, col].set_title(labels[i], wrap=True)

        else:
            # plot the image to the current subplot
            try:
                axs[i].imshow(images[i, :, :, :])
            except TypeError:
                axs.imshow(images[i, :, :, :])
            # set the title of the current subplot
            if labels is not None:
                axs[i].set_title(labels[i], wrap=True)

    plt.show()


def make_text(attribute_vectors: np.ndarray, num_descriptions: int = 10) -> list:
    """
    this function converts an attribute vector that contains facial features (CelebA dataset) in terms of integer
    values to a string that contains the same description in terms of feature names, num_description time so
    :param attribute_vectors: numpy.ndarray, a vector of length 40 ([-1, 1, 1, -1, ..., 1]) facial features encoding
    :param num_descriptions: int, number of description per attribute vector to output
    :return: str, a string containing the feature names separated by a comma
    """
    if len(attribute_vectors.shape) == 1:
        assert attribute_vectors.shape == (40, ) or attribute_vectors.shape == (1, 40)
        attribute_vectors = attribute_vectors[:, None]

    # making sure the shape is supported (must contain 40 elements for CelebA dataset)
    assert len(attribute_vectors.shape) == 2 and attribute_vectors.shape[0] == 40,\
        f'Unsupported shape for the Attribute Vector, expected (40, 1) got {attribute_vectors.shape}'

    k_sentences = []

    # def generate_text(attributes, lookup):
    #     text = "".join([lookup[i] + ', ' for i in range(attributes.shape[0]) if attributes[i] == 1])
    #     text = text[:len(text) - 2] + '.'
    #     return text
    # get the facial description for each vector in the array
    for i in range(num_descriptions):
        permutation = np.random.permutation(attribute_vectors.size)
        attribute_vectors_permutation = attribute_vectors[permutation]
        facial_features_permutation = facial_features[permutation]
        text = "".join([facial_features_permutation[i] + ', '
                        for i in range(attribute_vectors_permutation.shape[0])
                        if attribute_vectors_permutation[i] == 1])[:-2] + "."
        k_sentences.append(text)

    return k_sentences


def crop_face(image: np.ndarray,
              hyper_parameters_sets: list = None) -> tf.Tensor or None:
    """
    this function crops an image according the position of the face in the image, and pads the image with two columns
    of black pixels to make sure we have 1:1 scale images
    :param image: numpy.ndarry, image to be cropped
    :param hyper_parameters_sets: list, list of pairs of hyperparameters for the face detection model
    trying a different set of hyperparameters
    :return: tf.Tensor the cropped image
    """
    if type(hyper_parameters_sets) == list \
            and len(hyper_parameters_sets) == 0:  # if the list of possible pairs was exhausted
        return None  # give up

    if hyper_parameters_sets is None:
        hyper_parameters_sets = [(x, y) for x in neighbour_values for y in scale_values]

    hyper_parameters_set = hyper_parameters_sets[len(hyper_parameters_sets) // 2]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert image to grayscale

    faces = face_detector.detectMultiScale(
        gray_image,
        minNeighbors=hyper_parameters_set[0],
        scaleFactor=hyper_parameters_set[1],
        minSize=(32, 32)
    )  # create an instance from the face detector with obtained hyperparameters
    hyper_parameters_sets.pop(len(hyper_parameters_sets) // 2)  # remove the used pair
    if len(faces) != 1:  # if the number of faces isn't exactly 1
        # try a different set of hyperparameters
        if len(hyper_parameters_sets) != 0:
            return crop_face(image,
                             hyper_parameters_sets=hyper_parameters_sets)
        else:
            return None
    # store the result of the prediction process which is a tuple that contains the coordinates of the face + dimensions
    try:
        x, y, width, height = faces[0]
    except IndexError:
        return None

    center_x = (x + width) // 2  # compute the center of the horizontal coordinates
    crop_value = 40
    # if the center on x is in the center of the image (plus or minus 30 pixels)
    if image.shape[0] // 2 + crop_value > center_x > image.shape[0] // 2 - crop_value:
        # crop crop_value / 2 pixels from the top and crop_value / 2 pixels from the bottom
        result = image[crop_value // 2: image.shape[0] - crop_value // 2, :, :]

    # if the center on x is closer to the coordinate origin
    elif center_x < image.shape[0] // 2 - crop_value:
        # crop crop_value pixels from the bottom
        result = image[10:image.shape[0] - crop_value + 10, :, :]
    # otherwise (the only remaining case is that x is closer to the end of the image)
    else:
        # crop crop_value pixels from the top of the image
        result = image[crop_value:image.shape[0], :, :]

    # since the original shape of the image is (218, 178) and we cropped 30 pixels up and down
    # we still need to meet partway, so we pad 10 pixels left and right to get (188, 188) images
    padding_value = 10

    # create a placeholder to contain the image
    padded_result = np.zeros((result.shape[0], result.shape[1] + padding_value, image.shape[2]), dtype=np.uint8)

    # add the image to the placeholder
    padded_result[:, 5:result.shape[1] + 5, :] += result

    # return the padded result as the cleaned and cropped image
    return result


def train_test_val_split(data: pd.DataFrame,
                         train_proportion: float = .95,
                         test_proportion: float = .025,
                         val_proportion: float = .025,
                         shuffle: bool = True,
                         return_validation: bool = True) -> dict:
    """
    a helper function to split a given DataFrame (which presumably contains the whole dataset info) into train test and
    validation splits given the proportions
    :param data: pandas.DataFrame, the original cleaned dataframe to split
    :param train_proportion: float, the proportion of the train split in the data, default is .95
    :param test_proportion: float, the proportion of the test split in the data, default is .025
    :param val_proportion: float, the proportion of the validation split in the data, default is .025
    :param shuffle: bool, set to False if you want to maintain the order of the data upon splitting, default is False
    :param return_validation: bool, set to False if you want to return only train and test splits (no validation)
    :return: dict, a dictionary containing the splits of the data ('train', 'test', and optionally 'val')
    """
    # in case the proportions of the data splits doesn't cover the whole dataset
    if train_proportion + test_proportion + val_proportion != 1.:
        # take only the train proportion into account (and infer the other two)
        train_proportion = max(train_proportion, test_proportion, val_proportion)
        # in this case consider the same size for test and validation splits
        test_proportion = (1. - train_proportion) / 2 if return_validation else 1

    # number of examples is the same as the number of rows
    n_examples = data.shape[0]
    # the index at which the train split ends
    train_end = int(n_examples * train_proportion)
    # the index at which the test split ends
    test_end = train_end + int(n_examples * test_proportion)

    # define an index handler (to play the two possible scenarios at once [shuffle])
    indexer = np.random.permutation(n_examples) if shuffle else np.arange(n_examples)

    data_split = {
        # train split is the indexes' indexer[0] through indexer[train_end] (a permutation or a sequence of number)
        'train': data.iloc[indexer[:train_end]],
        # test split is the indexes' indexer[train_end] through indexer[test_end] (roughly the same as the line above)
        'test': data.iloc[indexer[train_end:test_end]]
    }
    if return_validation:
        # val split is the last n_examples - train_split - test_split indexes of the indexer
        data_split['val'] = data.iloc[indexer[test_end:]]

    # return the dictionary that holds the final split of the dataset
    return data_split


def resize(image_1: tf.Tensor,
           image_2_shape: tuple or list,
           data_format: str = 'channels_first') -> tf.Tensor or np.ndarray:
    """
    bi-linear interpolation to upgrade the size of one image to match another image's shape
    :param image_1: tf.Tensor, batch of images to be resized
    :param image_2_shape: tuple or list, target shape for resizing process
    :param data_format: str, specify what axis holds the number of channels (tf.resize only deals with channels_last
    data format)
    :return: tf.Tensor, the resized batch
    """
    if len(image_1.shape) == 3:
        image_1 = image_1[None, :, :, :]
    if data_format == 'channels_first':
        # move the channels' axis to the last dimension and introduce the Mini Batch axis
        image_1 = tf.transpose(image_1, [0, 2, 3, 1])
    # interpolate
    upsampled_image = tf.image.resize(image_1, image_2_shape[:2])

    if data_format == 'channels_first':
        # move the channels' axis back to the second dimension
        upsampled_image = tf.transpose(upsampled_image, [0, 3, 1, 2])

    return upsampled_image


def fetch_latest_check_point(directory: str = 'models', keyword: str = 'cp') -> tuple or None:
    """
    this function returns the last element in a directory (after sorting), which will be used to retrieve saved models
    later in training
    :param directory: str, the directory in which the saved check_points are saved
    :param keyword: str, the keyword (a substring that's contained in all checkpoints)
    :return: str, the directory of the last check_point
    """
    try:  # encapsulating the retrieval process in a try except clause in case the directory doesn't exist
        contents = list(sorted(os.listdir(directory), reverse=True))  # get the contents of the directory
        if type(contents) == list:  # making sure the listdir process return a list
            contents = [content for content in contents if keyword in content]
            if len(contents) > 0:  # making sure the list isn't empty
                return contents[0], len(contents)  # return the element of interest and number of elements
            else:
                return None, 0
    except FileNotFoundError as ex:
        print(str(ex))  # print the error status
        return None, 0


def get_mapping(csv_file: str = os.path.join('Data', 'cleaned_data.csv')) -> list:
    data_df = pd.read_csv(csv_file)
    # set the labels to be the column names of each column in the data except the image id and face description
    labels: list = [label for label in data_df.keys() if label not in ['image_id', 'description']]

    # initialize a lookup table to retrieve the label given the id of the label
    id2label: dict = {idx: label for idx, label in enumerate(labels)}

    # initialize a lookup table to retrieve the id of the label
    label2id: dict = {label: idx for idx, label in enumerate(labels)}

    return [labels, id2label, label2id]


def plot_function(values: tf.Tensor or np.ndarray, legend: list or str = None) -> None:
    """
    plots the values of the function(s) in one plot for comparison
    :param values: tf.Tensor, a tensor with rank one or two, if the tensor is rank two the function will assume that
    the value of the first axis is the index of the function to plot and the values in the second axis are the
    function values
    :param legend: list, can be None, with length equal to the number of functions to plot
    :return: None
    """
    # if a legend is just one string then set it as title
    if type(legend) == str:
        plt.title(legend)

    if tf.rank(values) == 1:
        values = values[None, :]
    steps = range(values.shape[1])
    possible_colours = ['r', 'g', 'b', 'k']
    assert values.shape[0] <= 4, 'Function only supports plotting 4 function tops'
    # iterate through the first dimension
    for i in range(values.shape[0]):
        # plot the current function with the corresponding color
        plt.plot(steps, values[i, :], possible_colours[i])

    # set Legend only if it's not None
    if legend is not None:
        plt.legend(legend)

    # show the plot
    plt.show()


if __name__ == "__main__":
    def test1():
        x = fetch_latest_check_point('Data')
        print(x)

        random_images = np.random.uniform(size=(4, 64, 64, 3), low=0., high=1.)
        attr_vec = [1, -1, -1, 1, 1, 1, -1, -1, 1, 1,
                    1, -1, -1, 1, 1, 1, -1, -1, 1, 1,
                    1, -1, -1, 1, 1, 1, -1, -1, 1, 1,
                    1, -1, -1, 1, 1, 1, -1, -1, 1, -1]
        attr_vec = np.array(attr_vec).reshape(-1)

        random_labels = np.array([attr_vec[np.random.permutation(attr_vec.size)] for _ in range(4)])
        print_colored_text(f'{random_labels.shape}', 'blue')
        random_labels = [make_text(random_labels[i])[0] for i in range(random_labels.shape[0])]
        plot_images(random_images, random_labels, data_format='channels_last', rows=2, cols=2)
        text_label = make_text(attr_vec)
        data_folder = os.path.join('Data', 'images')
        files = os.listdir(data_folder)
        number_of_images_to_plot = 30
        print_colored_text("Reading 50 Images from original dataset...", 'blue')
        original_images = [plt.imread(os.path.join(data_folder, file))
                           for file in files[50:50 + number_of_images_to_plot]]

        print_colored_text("Cropping the faces out....", 'blue')
        from tqdm import tqdm
        cropped_images = [crop_face(image) for image in tqdm(original_images)]
        cropped_images = [image for image in cropped_images if image is not None]
        original_images = tf.convert_to_tensor(original_images)
        original_labels = ['Original'] * number_of_images_to_plot
        cropped_images = tf.convert_to_tensor(cropped_images)
        cropped_labels = ['Cropped'] * number_of_images_to_plot
        plot_images(original_images, original_labels)
        plot_images(cropped_images, cropped_labels)
        print(cropped_images.shape)
        print(original_images.shape)


    def test2():
        clean_data()

    def test3():
        images = tf.random.uniform(shape=[16, 256, 256, 3])
        plot_images(images=images, total_title='Current Batch')

    def test4():
        values = tf.random.normal(shape=[3, 100])
        legend = ['red', 'green', 'blue']
        plot_function(values=values, legend=legend)

    test1()
    # test2()
    # test3()
    # test4()
