import json
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
from torchvision.io import read_image

import LabelbyLabelEncoder
from utils import print_colored_text
from StyleGAN_UTILS import generate_noise
from utils import block_print, enable_print
from utils import get_mapping
from Generator import import_generator
from ImageEncoder import ImageEncoder

block_print()


def c_print(text: str, color: str = 'green') -> None:
    """
    enables printing, prints and then disables printing again
    :param text: str, text to be printed
    :param color: str, colour of the text to be printed
    :return: None
    """
    enable_print()
    print_colored_text(text, color=color)
    block_print()


def generate_dataset(new_directory: str = os.path.join('Data', 'Generated Dataset'),
                     num_samples: int = 100_000,
                     target_size=None,
                     generator_pickle: str = os.path.join('models', 'ImageGenerator', 'StyleGANPT', 'ffhq.pkl'),
                     classifier: torch.nn.Module = None,
                     classification_tolerance: bool = False,
                     batch_size: int = 8,
                     resume: bool = True,
                     final_output_file: str = None,
                     write_interval: int = 10,
                     truncation: float = .7):
    """
    uses the generator from the .pkl file to generate 'num_samples' images and stores the images and their corresponding
    noise for later use in the controlling phase
    :param new_directory: str, the directory to which this function will output the new dataset
    :param num_samples: str, number of images to generate
    :param target_size: tuple or list, iterable with exactly two elements which are the width and height of the
    generated images
    :param generator_pickle: str, path to the pickle file that holds the StyleGAN2 model
    :param classifier: torch.nn.Module (or equivalent), a model to label generated images
    :param classification_tolerance: bool, specify if you want to generate the new dataset without labels
    :param batch_size: int, number of images to generate each iteration
    :param resume: bool, specify wither to create the dataset from scratch (set to False) or continue based on what's
    found in the Generated Dataset Folder
    directory of the new Dataset (default, True)
    :param final_output_file, str, path to the json file that will accompany the generated images
    :param write_interval: int, number of batches before saving a file
    :param truncation: float, the maximum value in the generated noise
    (i.e., the generated noise will be in the range [-truncation, +truncation])
    :return: None
    """
    total_dictionary = {}  # a placeholder to the generated data info in all the temp files
    # set the name of the output json file
    if final_output_file is None:
        final_output_file = os.path.join(new_directory, 'Generated Dataset Info.json')

    mapping = get_mapping()[1]
    # set the default starting index (in case you have electricity in your country you don't have to worry about a lot
    # of operations we'll do later to this variable)
    starting_index = 0
    # start over?
    if not resume:
        # delete everything
        if os.path.isdir(new_directory):
            import shutil
            shutil.rmtree(new_directory)
    # otherwise (smart choice)
    else:
        # check the existence of the final_output_file from the last run (MUST be there no matter what)
        if os.path.isfile(final_output_file):
            with open(final_output_file, 'r') as saved_file:
                total_dictionary = json.load(saved_file)
            effective_number_of_images = len(total_dictionary.keys())
            starting_index = effective_number_of_images
            temp = num_samples
            num_samples -= effective_number_of_images
            if num_samples == 0:
                c_print(f'Last run of this function generated exactly: {temp} samples, execution will return now.')
                return None
        else:
            c_print(f'The Json file that MUST accompany the generated data does not exist in the specified directory,'
                    f'We will start from scratch, with no resuming')
            return generate_dataset(resume=False,  # most important parameter is the resume and is set to false
                                    new_directory=new_directory,
                                    num_samples=num_samples,
                                    target_size=target_size,
                                    generator_pickle=generator_pickle,
                                    classifier=classifier,
                                    classification_tolerance=classification_tolerance,
                                    batch_size=batch_size,
                                    final_output_file=final_output_file,
                                    write_interval=write_interval)

    # in case the directory to which the new dataset will be written doesn't exist
    if not os.path.isdir(new_directory):
        # create it
        os.mkdir(new_directory)

    # set the directory to which the images will be written
    images_folder = os.path.join(new_directory, 'images')

    # create the directory if it doesn't exist
    if not os.path.isdir(images_folder):
        os.mkdir(images_folder)
    num_batches = num_samples // batch_size + int(num_samples % batch_size != 0)
    reminder = num_samples % batch_size
    file_index = 0
    temp_files = []

    def write_all(files):
        # loop through the files
        for file in files:
            # read
            with open(file, 'r') as temp_file_handle:
                # load
                temp_dict = json.load(temp_file_handle)
                # loop through all the keys and values
                for key, value in temp_dict.items():
                    # add them to the total dictionary of th dataset
                    total_dictionary[key] = value
        # write a new file that hold the information of the generated images
        with open(final_output_file, 'w') as output_file:
            # dump the info in the specified file
            json.dump(total_dictionary, output_file)
        # remove all temp files after finishing
        for file in files:
            os.remove(file)

    def write_file(file, dictionary):
        with open(file, 'w') as temp_file:
            json.dump(dictionary, temp_file)

    try:
        block_print()
        generated_dataset_dictionary = {}

        # disable warnings
        with warnings.catch_warnings():
            # ignore all warnings
            warnings.simplefilter('ignore')
            # get the generator model
            generator = import_generator(generator_pickle).cuda()
            if classifier is None:
                if classification_tolerance:
                    c_print(f'The Images will not have any labels..', 'blue')
                else:
                    exit('Could not load the classification model,'
                         'If you want to get the images, please set the value of classification_tolerance to True, '
                         'or double check the path to the ImageEncoder Model')

            # loop over num_samples
            for i in tqdm(range(num_batches), desc='Generating Dataset'):

                # generate noise
                if i < num_batches - 1:
                    np_noise = generate_noise(batch_size, generator.z_dim, truncation)
                else:
                    if reminder == 0:
                        reminder = batch_size
                    np_noise = generate_noise(reminder, generator.z_dim, truncation)

                # convert to torch.Tensor (on the CUDA device)
                z = torch.Tensor(np_noise).cuda()
                block_print()

                # generate a batch of images
                images: torch.Tensor = generator(z, None).cpu()

                # convert the dynamic range [-1, 1] into the other [0, 1]
                images = (images + 1) / 2 * 255.
                images = torch.clip(images, min=0., max=255).type(torch.uint8)

                # get labels
                if classifier is not None:
                    label_ids = classifier.infer(images.detach().numpy(), return_logits=True)
                    label_ids = np.where(label_ids > 0.5, 1., 0.)

                # loop over images in the batch
                for j in range(images.shape[0]):
                    current_sample_index = starting_index + i * batch_size + j
                    # set a path to save the image to
                    image_file_name = os.path.join(images_folder, f'{current_sample_index:06}.jpg')
                    # save the image to said path
                    plt.imsave(image_file_name, np.transpose(images[j].detach().cpu().numpy(), [1, 2, 0]))
                    # add the noise and its corresponding image to image's information
                    current_image_info = {'noise': np_noise[j].tolist(), 'image': image_file_name}
                    # get labels if applicable
                    if classifier is not None:
                        label_id = label_ids[j]
                        labels = [mapping[idx] for idx, value in enumerate(label_id) if value == 1.]
                        current_image_info['label_ids'] = label_id.tolist()
                        current_image_info['labels'] = labels
                    generated_dataset_dictionary[current_sample_index] = current_image_info

                # check if the writing interval was reached
                if ((i + 1) % write_interval == 0) or (((i + 1) % write_interval != 0) and (i + 1) == num_batches):
                    # create a path to a temporary file to hold the current batch of data
                    temp_file_path = os.path.join(new_directory, f'temp{file_index:04}.json')
                    write_file(temp_file_path, generated_dataset_dictionary)
                    temp_files.append(temp_file_path)
                    # reset the dictionary
                    generated_dataset_dictionary = {}
                    # increment the running index of the generated temp files
                    file_index += 1

        write_all(temp_files)

        enable_print()
        c_print('Generating Data Was Done Successfully!!')
    except KeyboardInterrupt:
        enable_print()
        c_print('I know you want to  exit immediately but for stability purposes we have to write everything we did,...'
                '\nSorry for the inconvenience'
                f'\nTotal Number of Generated Images: {starting_index + file_index * write_interval * batch_size}',
                'red')
        block_print()
        write_all(temp_files)


def classify(classifier,
             data_path: str = os.path.join('Data', 'Generated Dataset', 'Generated Dataset Info.json'),
             new_dataset_file_path: str = os.path.join('Data',
                                                       'Generated Dataset',
                                                       'Generated Classified Data Info (V2).json')):
    """
    a separate helper function to classify generated images with the desired ImageEncoder, to reduce the time needed for
    the fused generation and classification in the previous function
    :param data_path: str, the path to generated images infor
    :param classifier: callable, the model which will be used to classify generated images
    :param new_dataset_file_path: str, the path to the new Generated Classified Images
    """
    # importing pandas here (since it's only needed here)
    import pandas as pd
    feature_names = LabelbyLabelEncoder.facial_features[LabelbyLabelEncoder.indexes_of_interest]
    # read the generated images infor json file
    dataset = pd.read_json(data_path)

    # transpose it because it looks stupid
    dataset = dataset.transpose()

    dataset = dataset.head(dataset.shape[0] // 1000)
    columns = ['image_path', 'probs', 'label_ids', 'labels']
    data_df = pd.DataFrame(columns=columns)

    # logits_global = [None] * dataset.shape[0]
    #
    # # a placeholder for the label_ids
    # label_ids_global = [None] * dataset.shape[0]
    #
    # # a placeholder for the labels
    # labels_global = [None] * dataset.shape[0]
    #
    # # ids to label names mapper
    # mapping = get_mapping()[1]

    # loop through the dataset of generated images
    for i in tqdm(range(dataset.shape[0]), desc='Classifying'):
        # get the current image's path
        current_image_path = dataset['image'].loc[i]
        print(current_image_path)
        # read the current image
        current_image = read_image(current_image_path).float()[None, :, :, :]

        # classify current image
        probs = classifier(current_image, )[0].detach().cpu().numpy()

        # get the label ids by thresholding
        label_ids = np.where(probs >= 0.5, 1., 0).astype(int)
        # dictionary = {LabelbyLabelEncoder.facial_features[i]: label_ids[i]
        #               for index, i in enumerate(indexes_of_interest)}
        # get label names using mapper
        labels = [feature_names[k]
                  for k in range(len(feature_names))
                  if label_ids[k] == 1.]

        # temp_dictionary = {
        #     'image_path': current_image_path,
        #     'probs': probs.tolist(),
        #     'label_ids': label_ids.tolist(),
        #     'labels': labels
        # }
        row = [current_image_path, probs.tolist(), label_ids.tolist(), labels]
        data_df.append(pd.DataFrame(row, columns=columns, index=data_df.index))

    # save progress into another file
    data_df.to_json(new_dataset_file_path)
    print(data_df.head(10))


if __name__ == '__main__':
    with warnings.catch_warnings():
        # generate_dataset(num_samples=100_000,
        #                  write_interval=4,
        #                  batch_size=8,
        #                  resume=True,
        #                  classification_tolerance=True,
        #                  classifier=None,
        #                  truncation=1.)
        enable_print()
        classify(classifier=ImageEncoder(file_path=os.path.join('models', 'ImageEncoder', 'ImageEncoder')).infer)
