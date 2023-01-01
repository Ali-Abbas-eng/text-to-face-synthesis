import gc
import warnings
import torch
import numpy as np
import os
import torchvision.transforms
from GLOBAL_NAMES import CACHE_DIRECTORY, DESIRED_LABELS, GENERATED_IMAGES, DESIRED_NOISE
from StyleGAN_UTILS import generate_noise
from TextEncoder import TextEncoder
from Generator import import_generator
from Noise2Labels import Noise2Labels
from ImageEncoder import ImageEncoder
from googletrans import Translator
import utils


def encode_text(description: str = "He's a bald man wearing eyeglasses") -> np.ndarray:
    text_encoder = TextEncoder(only_inference=True, text_data_pipeline=None)
    translated_description = Translator().translate(description).text
    description = translated_description if translated_description is not None else description
    desired_labels = text_encoder.infer(description)
    del text_encoder
    gc.collect()
    return desired_labels


def encode_image(image: np.ndarray, data_format: str = 'channels_last') -> np.ndarray:
    if len(image.shape) == 3:
        image = image[np.newaxis, :, :, :]
    image_encoder = ImageEncoder()
    observed_labels = image_encoder.infer(image, data_format=data_format, return_logits=True).detach().cpu().numpy()
    return np.where(observed_labels > 0, 1., 0.)


def get_desired_noise(original_noise, original_labels, desired_labels) -> np.ndarray:
    noise_re_mapper = Noise2Labels(verbose=False)
    z_hat = noise_re_mapper.infer(z=original_noise, desired_labels=desired_labels, initial_labels=original_labels)
    return z_hat


def generate_images(noise: np.ndarray = None) -> np.ndarray:
    if noise is None:
        try:
            noise = np.load(os.path.join(CACHE_DIRECTORY, DESIRED_NOISE))
        except FileNotFoundError:
            noise = generate_noise(n_samples=4, z_dim=512, truncation=.5)
    generator = import_generator().cuda()
    images = generator(torch.Tensor(noise).cuda(), None).detach()
    images = torchvision.transforms.Resize(256)(images)
    images = torch.clip(input=(images + 1) / 2, min=.0, max=1.)
    return images.cpu()


def encapsulated_generation(description: str = 'she was wearing eyeglasses', plot: bool = True) -> np.ndarray:
    desired_labels = encode_text(description=description)
    original_noise = generate_noise(n_samples=4, z_dim=512, truncation=.5)
    initial_images = generate_images(noise=original_noise)
    initial_labels = encode_image(initial_images)
    desired_noise = get_desired_noise(original_noise=original_noise,
                                      desired_labels=desired_labels,
                                      original_labels=initial_labels)
    desired_images = generate_images(noise=desired_noise).detach().cpu().numpy()
    if plot:
        utils.plot_images(desired_images, data_format='channels_first')
    from matplotlib.pyplot import imsave
    for i in range(desired_images.shape[0]):
        image = np.transpose(desired_images[i], [1, 2, 0])
        imsave(os.path.join(CACHE_DIRECTORY, f'desired image {i}.png'), image)

    return desired_images


if __name__ == '__main__':
    with warnings.catch_warnings():
        utils.block_print()
        warnings.simplefilter('ignore')
        encapsulated_generation(description='someone wearing eyeglasses')
