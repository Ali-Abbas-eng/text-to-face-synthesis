import os
import utils
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
from data_downloader import download_preprocessed_data, download_dataset
import pandas as pd
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf
from utils import plot_images, make_text, crop_face, clean_data, train_test_val_split, resize
from utils import print_colored_text as c_print
from tqdm import trange, tqdm


class ImageDataHandler:
    """
    This class handles the large amount of Image Data we have on the local machine, to use the data for training
    create an instance from the class and then use 'instance'.image_encoding_data_train to get the train dataset
    generator
    note: you should use instance.train_data.take(-some integer-) to get a tuple of a data batch of size 'batch_size'
    where the first element of the batch is the batch of images, and the second is the labels
    """

    def __init__(self,
                 images_folder: str = os.path.join("Data", "images"),
                 annotations_file: str = os.path.join("Data", "annotations", "list_attr_celeba.txt"),
                 un_cropped_images_folder: str = os.path.join("Data", "un_cropped_images"),
                 cropped_images_directory: str = os.path.join("Data", "faces (1.0 ratio)"),
                 train_batch_size: int = 64,
                 image_final_size: tuple = (256, 256, 3),
                 test_val_batch_size: int = 128,
                 crop: bool = False,
                 data_format: str = 'channels_last',
                 is_it_just_a_test: bool = False,
                 verbose: bool = False,
                 show: bool = False,
                 download: bool = False,
                 validate: bool = False) -> None:
        """
        initialize the instance by providing the necessary information
        :param images_folder: the path to the folder that holds the images, default "Data/Images"
        :param annotations_file: the path to the txt FILE that contains the facial features of each image,
         default "Data/list_attr_celeba.txt"
         :param un_cropped_images_folder: str, path to the original images
        :param train_batch_size: number of examples to load in each batch, default 64
        :param image_final_size: the final shape of the image to which all images will be resized,
        default None (keep dimensions as they are)
        :param test_val_batch_size: int, batch size of the test and evaluation sets
        :param crop: bool, set to true if you want the class to consider cropping the images with respect ot the faces
        :param show: bool, set to True if you would like to view some images after the cleaning process
        :param is_it_just_a_test: bool, set to True to take only 1% of the dataset into account
        :param verbose: set to True if you want to see messages indicating success or failure of each step
        """
        if not os.path.exists('Data'):
            os.mkdir('Data')
        if not os.path.isdir(images_folder):
            os.mkdir(images_folder)

        if download:
            download_dataset(unzipped_directories={'celeb_a_directory': images_folder,
                                                   'celeb_a_hq_directory': images_folder + ' (HQ)'},
                             delete=False)
            download_preprocessed_data(download_path=images_folder)

        self.images_folder: str = images_folder
        self.un_cropped_images_folder: str = un_cropped_images_folder
        self.data_file: str = annotations_file
        self.verbose: bool = verbose
        self.data_df: pd.DataFrame = pd.DataFrame()
        self.batch_size: int = train_batch_size
        self.test_val_batch_size = test_val_batch_size
        self.data_format = data_format
        self.n_features = 40
        self.show = show
        self.is_it_just_a_test = is_it_just_a_test
        self.cropped_images_directory = cropped_images_directory
        self._customize_data_df(crop=crop, validate=validate)  # get the final clean and ready to use DataFrame
        self.train_data_df: pd.DataFrame = pd.DataFrame()
        self.test_data_df: pd.DataFrame = pd.DataFrame()
        self.val_data_df: pd.DataFrame = pd.DataFrame()
        self.target_size: tuple = image_final_size \
            if image_final_size is not None \
            else plt.imread(os.path.join(self.cropped_images_directory, '000001.jpg')).shape
        self.cleaned_data_path = None
        self.image_encoding_data_train = None
        self.image_encoding_data_test = None
        self.image_encoding_data_val = None
        self.steps_per_train_epoch = None
        self.data_gen = None
        self.steps_per_validation = None
        self.steps_per_test = None
        self.create_image_encoding_dataset()

    def _customize_data_df(self, crop: bool = False, validate: bool = False) -> None:
        """
        helper function that generates the final data frame from which data will be loaded
        :param crop: bool, set to True in case faces were never cropped from the original images
        :return: None
        """
        self.cleaned_data_path = os.path.join("Data", "cleaned_data.csv")

        if os.path.isfile(self.cleaned_data_path):
            # in case the cleaned data file exists load it and don't mind the tedious steps above
            self.data_df = pd.read_csv(self.cleaned_data_path)
            # self.images_folder = os.path.join("Data", "faces (1.0 ratio)")
            if crop:
                if self.verbose:
                    c_print('Cropping Faces from the Original (218, 178) sized Images', 'red')

                self._crop_faces()  # crop the images, so it's 1:1 ratio

                if self.verbose:
                    c_print('Cropping Faces, Done!')

                self.data_df.to_csv(self.cleaned_data_path,
                                    index_label=False,
                                    index=False)  # save the DataFrame as a CSV file
            self.images_folder = self.cropped_images_directory
            if os.path.isdir(self.cropped_images_directory) and validate:
                cropped_images = os.listdir(self.cropped_images_directory)
                rows_to_drop = []
                for i in tqdm(range(self.data_df.shape[0]), desc='Deleting unvalidated rows'):  # loop over all the images
                    row = self.data_df.iloc[i]  # take the current row
                    if not row['image_id'] in cropped_images:
                        rows_to_drop.append(i)
                self.data_df.drop(rows_to_drop, axis=0, inplace=True)
                self.data_df.to_csv(self.cleaned_data_path,
                                    index_label=False,
                                    index=False)
            if self.verbose:
                c_print('Cleaned Data CSV File was found, and loaded successfully!')

        else:
            if self.verbose:
                c_print("Cleaned Data CSV File was not found, initiating manual load", "red")
            clean_data(self.data_file, self.cleaned_data_path)
            self._customize_data_df(crop=crop)
            # edit the 'image_id' column in the dataframe, so it holds the actual path to the images
        self.data_df['image_id'] = [os.path.join(self.images_folder, img_id) for img_id in self.data_df['image_id']]

        if self.verbose:
            print(f"Total number of annotations {self.data_df.shape}\n")
            print(self.data_df.head())

    def _crop_faces(self) -> None:
        """
        crop images according to the place of the face in the picture in order to get 1:1 ratio images
        which is much more robust in the resizing process since we need to perform this operation to each image
        when we train the image encoder, if the face detector embedded in the crop function failed to detect one face in
        the image will be discarded (i.e., the image won't be considered in the training set
        :return: None
        """
        if not os.path.isdir(self.cropped_images_directory):
            os.mkdir(self.cropped_images_directory)
        rows_to_drop = []  # a storage for the rows we need to remove
        for i in trange(self.data_df.shape[0]):  # loop over all the images
            row = self.data_df.iloc[i]  # take the current row
            # get the path of the original image to be cropped
            original_image_path = os.path.join(self.images_folder, row['image_id'])

            # specify the path to which the function will save the cropped image
            cleaned_image_path = os.path.join(self.cropped_images_directory, row['image_id'])

            # read the image
            image = plt.imread(original_image_path)

            # save the image to the desired path
            if not os.path.isfile(cleaned_image_path):
                # crop the image according to the face position
                image = crop_face(image)

                # in case of success
                if image is not None:
                    plt.imsave(arr=image, fname=cleaned_image_path)
                else:
                    # otherwise save the row index for later elimination
                    rows_to_drop.append(i)

        # eliminate each row that corresponded to an image to which the detector failed to find one face

        self.data_df.drop(rows_to_drop, axis=0, inplace=True)

        # set the image_folder attribute to be the folder that holds the cropped images
        self.images_folder = self.cropped_images_directory

        if self.verbose:
            print("Data After Cropping faces")
            print(self.data_df.head(10))

    def resize(self, image: tf.Tensor) -> tf.Tensor:
        """
        resize a given image to the desired size
        :param image: tf.Tensor image (batch) to be resized
        :return: tf.Tensor, resized batch
        """
        result = resize(image, self.target_size, data_format=self.data_format)
        return result

    def create_image_encoding_dataset(self) -> None:
        """
        a function that generates the final ImageDataGenerator instance (FlatMapDataset) (stored in self.train_data),
        which we will later feed to the ImageEncoder
        :return: None
        """

        self.data_gen = ImageDataGenerator(rescale=1 / 255.,
                                           data_format=self.data_format)  # declare an ImageDataGenerator instance
        labels_shape = [None, self.n_features]  # batch shape for the generator (labels)
        if self.target_size is None:
            # if image shape is not provided get the size of one image,
            # and set it as the target size (i.e. don't resize)
            path = os.path.join(self.images_folder, os.listdir(self.images_folder)[0])
            image = plt.imread(path)
            self.target_size = image.shape

        if self.verbose:
            c_print(f"Images will be resized to the shape: {self.target_size[0:2]}", "blue")

        images_shape = [None, *self.target_size[:2], 3]  # specify the exact batch shape for the generator (data)
        if self.data_format == 'channels_first':
            images_shape = [None, 3, *self.target_size]

        if self.is_it_just_a_test:
            self.data_df = self.data_df.head(self.data_df.shape[0] // 100)
        # split the data into train test and validation sets
        data_split = train_test_val_split(self.data_df, shuffle=False)

        self.train_data_df = data_split['train']
        self.test_data_df = data_split['test']
        self.val_data_df = data_split['val']

        # note batch_size argument is set to None to avoid issues related to the final batch in the dataset
        self.image_encoding_data_train = tf.data.Dataset.from_generator(
            lambda: self.data_gen.flow_from_dataframe(self.train_data_df,
                                                      target_size=self.target_size[0:2],
                                                      class_mode="raw",
                                                      x_col="image_id",
                                                      shuffle=False,
                                                      verbose=0,
                                                      validate_filenames=True,
                                                      batch_size=self.batch_size,
                                                      y_col=list(self.train_data_df.columns[1:41])),

            output_signature=(
                tf.TensorSpec(shape=tf.TensorShape(images_shape), dtype=tf.float32),
                tf.TensorSpec(shape=tf.TensorShape(labels_shape), dtype=tf.int8)
            ),

        )  # create dataset generator from the DataFrame that holds all the required info
        # calculate the number of iteration through the train set to reach its end
        self.steps_per_train_epoch = self.train_data_df.shape[0] // self.batch_size

        # create a dataset generator for the validation split
        self.val_data_df.reset_index()
        self.image_encoding_data_val = tf.data.Dataset.from_generator(
            lambda: self.data_gen.flow_from_dataframe(self.val_data_df,
                                                      target_size=self.target_size[0:2],
                                                      class_mode="raw",
                                                      x_col='image_id',
                                                      shuffle=False,
                                                      validate_filenames=False,
                                                      batch_size=self.test_val_batch_size,
                                                      y_col=list(self.val_data_df.columns[1:41])),
            output_signature=(
                tf.TensorSpec(shape=tf.TensorShape(images_shape), dtype=tf.float32),
                tf.TensorSpec(shape=tf.TensorShape(labels_shape), dtype=tf.int8)
            ),
        )

        # calculate the number of iteration through the validation set to reach its end
        self.steps_per_validation = self.val_data_df.shape[0] // 128

        # create a dataset generator for the test split
        self.test_data_df.reset_index()
        self.image_encoding_data_test = tf.data.Dataset.from_generator(
            lambda: self.data_gen.flow_from_dataframe(self.test_data_df,
                                                      target_size=self.target_size[0:2],
                                                      class_mode='raw',
                                                      x_col='image_id',
                                                      shuffle=False,
                                                      validate_filenames=False,
                                                      batch_size=self.test_val_batch_size,
                                                      y_col=list(self.val_data_df.columns[1:41])),
            output_signature=(
                tf.TensorSpec(shape=tf.TensorShape(images_shape), dtype=tf.float32),
                tf.TensorSpec(shape=tf.TensorShape(labels_shape), dtype=tf.int8)
            ),
        )
        # calculate the number of iterations through the test set to reach its end
        self.steps_per_test = self.test_data_df.shape[0] // 128

        if self.verbose:
            c_print('Creating ImageEncoding Dataset was Done Successfully!!')
            images, labels = None, None
            for images, labels in self.image_encoding_data_train.take(1):
                pass

            c_print('Train Set Info: ', 'blue')
            print('\tImages.shape: ', images.shape)
            print('\tLabels.shape: ', labels.shape)
            print('\tImages.dtype: ', images.dtype)
            print('\tLabels.dtype: ', labels.dtype)

            c_print('Test Set Info: ', 'blue')
            for images, labels in self.image_encoding_data_test.take(1):
                pass
            print('\tTestData: images.shape : ', images.shape)
            print('\tTestData: images.dtype : ', images.dtype)
            print('\tTestData: labels.shape : ', labels.shape)
            print('\tTestData: labels.dtype : ', labels.dtype)

            c_print('Validation Set Info: ', 'blue')
            for images, labels in self.image_encoding_data_val.take(1):
                pass
            print('\tTestData: images.shape : ', images.shape)
            print('\tTestData: images.dtype : ', images.dtype)
            print('\tTestData: labels.shape : ', labels.shape)
            print('\tTestData: labels.dtype : ', labels.dtype)
            if self.show:
                plot_images(images[:16], make_text(labels[:16]))

        if self.show and not self.verbose:
            train_images, train_labels = None, None
            for train_images, train_labels in self.image_encoding_data_train.take(1):
                pass
            plot_images(train_images[:16], make_text(train_labels[:16]))


if __name__ == '__main__':
    data_pipeline_object = ImageDataHandler()
    # data_pipeline_object.create_image_encoding_dataset()