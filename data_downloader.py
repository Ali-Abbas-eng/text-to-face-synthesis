import os
import gdown
from zipfile import ZipFile
from tqdm import tqdm
from sys import exit


def download_dataset(urls: dict = None,
                     output_files: dict = None,
                     unzipped_directories: dict = None,
                     delete: bool = False,
                     replace: bool = True) -> None:
    """
    this is a helper function that will download the dataset and extract it (by invoking another helper function), so
    the final architecture is the same as I have in my local repository
    :param urls: list, three strings, url of the celeba_a data zip file, celeb_a_hq data zip file and list_attribute.txt
    :param output_files: three strings, the file names to which downloaded data will be saved
    :param unzipped_directories: str
    :param delete: bool, delete the zipped files after extracting
    :param replace: bool, replace old files if the specified output directories already exist
    :return: None
    """
    if urls is None:
        # get the default links from my Google Drive (access is set to be Anyone with the link)
        celeb_a_zip = "https://drive.google.com/uc?id=1-3dAQrOZ686n_rEggtJPTah6grwac7TK"
        celeb_a_hq_zip = "https://drive.google.com/uc?id=1-3DM6UvMAiBYwG2Q9QpSxzvgG2SHsjsF"
        annotation_url = "https://drive.google.com/uc?id=1lkeWsa3MNT8LSRO8h2XFqKMSKoYIDopc"
        urls = dict(celeb_a_zip=celeb_a_zip, celeb_a_hq_zip=celeb_a_hq_zip, annotation_url=annotation_url)

    # in case the user didn't specify the directories of unzipped files
    if unzipped_directories is None:
        # set default directory for celeb_a dataset images
        celeb_a_directory = os.path.join('Data', 'images')
        if not os.path.isdir(celeb_a_directory):
            os.mkdir(celeb_a_directory)
        celeb_a_hq_directory = os.path.join('Data', 'images (HQ)')
        if not os.path.isdir(celeb_a_hq_directory):
            os.mkdir(celeb_a_hq_directory)
        unzipped_directories = dict(celeb_a_directory=celeb_a_directory, celeb_a_hq_directory=celeb_a_hq_directory)

    if output_files is None:
        if not os.path.isdir('Data'):
            os.mkdir('Data')
        # set the default name of the output directory of compressed files
        zipped_files_destination_directory = os.path.join('Data', 'Original Files (compressed)')

        # set annotations file path
        anno_file = os.path.join('Data', 'annotations')
        if not os.path.isdir(anno_file):
            os.mkdir(anno_file)
        anno_file = os.path.join(anno_file, 'annotations.txt')

        # delete the file if it already exists
        if os.path.exists(anno_file) and replace:
            os.remove(anno_file)

        # in case the directory doesn't exist
        if not os.path.isdir(zipped_files_destination_directory):
            # make it
            os.mkdir(zipped_files_destination_directory)
            # set celeb_a file
        celeb_a_file = os.path.join(zipped_files_destination_directory, 'celeb_a.zip')
        # delete the file if it already exists
        if os.path.exists(celeb_a_file) and replace:
            os.remove(celeb_a_file)
        # set celeb_a_hq file
        celeb_a_hq_file = os.path.join(zipped_files_destination_directory, 'celeb_a_hq.zip')
        # delete the file if it already exists
        if os.path.exists(celeb_a_hq_file) and replace:
            os.remove(celeb_a_hq_file)

        # wrap the file names
        output_files = dict(celeb_a_file=celeb_a_file, celeb_a_hq_file=celeb_a_hq_file, anno_file=anno_file)

    else:
        values = output_files.values()
        assert len(values) == 3
        assert (values[0] != values[1]) and (values[0] != values[2]) and (
                values[1] != values[2]), "The output directory for each file MUST be unique"

    # download celeb_a file
    gdown.download(url=urls['celeb_a_zip'], output=output_files['celeb_a_file'], quiet=False)

    # extract files from celeb_a zipped file
    unzip(zip_file=output_files['celeb_a_file'],
          output_directory=unzipped_directories['celeb_a_directory'],
          delete=delete)

    # download celeb_a_hq file
    gdown.download(url=urls['celeb_a_hq_zip'], output=output_files['celeb_a_hq_file'], quiet=False)

    # extract files from celeb_a_hq zipped file
    unzip(zip_file=output_files['celeb_a_hq_file'],
          output_directory=unzipped_directories['celeb_a_hq_directory'],
          delete=delete)

    # download celeba_a annotations
    gdown.download(url=urls['annotation_url'], output=output_files['anno_file'], quiet=False)


def unzip(zip_file: str = os.path.join('Data', 'Original Files (compressed)', 'data.zip'),
          output_directory: str = os.path.join('Data', 'Original Files'),
          delete: bool = False) -> None:
    """
    extracts all the files in a given zipped file to a given directory
    :param zip_file: str, the path to the file to be unzipped
    :param output_directory: str, the directory to which the function will unzip
    :param delete: bool, delete zipped files after extracting ?!
    :param replace: bool, replace already existing files ?!
    :return: None.
    """
    # in case the file does not exist at all
    if not os.path.isfile(zip_file):
        # don't bother with exception handling it's the user's responsibility to provide the correct path, just exit
        exit("File not found, Please make sure the file was downloaded successfully")

    # create a reference to the zip file
    with ZipFile(zip_file, 'r') as zip_ref:
        # for each member of the zipped file (iterate through tqdm to obtain nice progress bar)
        for member in tqdm(zip_ref.infolist(), desc='Extracting: '):
            try:
                zip_ref.extract(member, output_directory)
            except ValueError:
                exit("an unexpected error occurred")
    if delete:
        # remove the zipped file given the user's choice
        os.remove(zip_file)


def download_preprocessed_data(url: str = "https://drive.google.com/uc?id=1BBe7u4ri-sdgjxvSluDgj1sMxl81uvn7",
                               download_path: str = None,
                               zipped_file_name: str = 'pre_processed.zip',
                               unzip_directory: str = None):
    """
    download the files of the cropped images, instead of performing the preprocessing again
    :param zipped_file_name: str, the name of the zipped file to be decompressed
    :param url: str, the url of the preprocessed data zip file
    :param download_path: str, the directory to which the preprocessed data will be downloaded
    :param unzip_directory: str, the directory to which the preprocessed data zip file will be extracted
    :return: None
    """
    if download_path is None:
        if not os.path.isdir('Data'):
            os.mkdir('Data')
        if not os.path.isdir(os.path.join('Data', 'Pre-Processed (compressed)')):
            os.mkdir(os.path.join('Data', 'Pre-Processed (compressed)'))
        download_path = os.path.join('Data',
                                     'Pre-Processed (compressed)',
                                     'pre_processed.zip')
    else:
        if not os.path.isdir(download_path):
            exit(f'[Error 2] No such file or directory: {download_path}')
        else:
            download_path = os.path.join(download_path, zipped_file_name)

    if unzip_directory is None:
        unzip_directory = 'Data'
        if not os.path.isdir(unzip_directory):
            os.mkdir(unzip_directory)

    gdown.download(url=url, output=download_path)
    unzip(zip_file=download_path, output_directory=unzip_directory, delete=True)


if __name__ == '__main__':
    download_dataset()

