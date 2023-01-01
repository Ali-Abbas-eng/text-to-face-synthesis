import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Resize

import LabelbyLabelEncoder
import utils
from utils import plot_images
from StyleGAN_UTILS import generate_noise
from Generator import import_generator
from LabelbyLabelEncoder import indexes_of_interest
from commands_wrapper import encode_text
import os


classifier = torch.load(os.path.join('models', 'ImageEncoder', 'ImageEncoder.pth'))
classifier.eval()
learning_rate = 0.001
beta_1, beta_2 = 0.5, 0.999
optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate, betas=(beta_1, beta_2))
generator = import_generator()
generator.eval()


def calculate_updated_noise(noise, grad, weight):
    new_noise = noise + grad * weight
    return new_noise


def gradient_ascent_entangled_generation(target_indexes, num_images: int = 8):
    noise = generate_noise(n_samples=num_images, truncation=.5, z_dim=512)
    num_grad_steps = 5
    noises = torch.zeros(size=[num_grad_steps + 1, 512])
    resizer = Resize(256)
    noise = torch.Tensor(noise).to('cuda').requires_grad_()
    noises[num_grad_steps] = noise
    for i in range(num_grad_steps):
        optimizer.zero_grad()
        fake = generator(noise, None)
        fake_classes_score = classifier(fake)[:, target_indexes].mean()
        fake_classes_score.backward()
        noise.data = calculate_updated_noise(noise, noise.grad, 1 / num_grad_steps)
        noises[i] = noise
    noises = noises.detach().cpu().numpy()
    intermediate_noise = torch.Tensor(np.sum(noises, axis=0) / noises.shape[0]).to('cuda')
    images = generator(intermediate_noise[None, :], None).detach()
    images = resizer(images).cpu().numpy()
    images = np.clip((images + 1) / 2, 0. , 1.)
    images = np.transpose(images[0], [1, 2, 0])[np.newaxis, :, :, :]
    plot_images(images=images)


def get_score(current_classifications, original_classifications, target_indices, other_indices, penalty_weight):
    other_distances = current_classifications[:,other_indices] - original_classifications[:,other_indices]
    other_class_penalty = -torch.norm(other_distances, dim=1).mean() * penalty_weight
    # Take the mean of the current classifications for the target feature
    target_score = torch.mean(current_classifications[:, target_indices])
    return target_score + other_class_penalty


def gradient_ascent_disentangled_generation(target_indexes, num_images: int = 8):
    num_grad_steps = 5
    fake_image_history = []
    feature_names = LabelbyLabelEncoder.facial_features[LabelbyLabelEncoder.indexes_of_interest]
    other_indexes = list(range(len(feature_names)))
    [other_indexes.remove(target_index) for target_index in target_indexes]
    resizer = Resize(256)
    noise = torch.Tensor(generate_noise(n_samples=num_images, z_dim=512, truncation=.5)).to('cuda').requires_grad_()
    utils.block_print()
    noises = torch.zeros(size=[num_grad_steps + 1, 512])
    noises[num_grad_steps] = noise
    original_classifications = classifier(generator(noise, None)).detach()
    for i in range(num_grad_steps):
        optimizer.zero_grad()
        fake = generator(noise, None)
        fake_score = get_score(
            classifier(fake),
            original_classifications,
            target_indexes,
            other_indexes,
            penalty_weight=0.1
        )
        fake_score.backward()
        noise.data = calculate_updated_noise(noise, noise.grad, 1 / num_grad_steps)
        noises[i] = noise
    noises = noises.detach().cpu().numpy()
    intermediate_noise = torch.Tensor(np.sum(noises, axis=0) / noises.shape[0]).to('cuda')
    images = generator(intermediate_noise[None, :], None).detach()
    images = resizer(images).cpu().numpy()
    images = np.clip((images + 1) / 2, 0., 1.)
    images = np.transpose(images[0], [1, 2, 0])[np.newaxis, :, :, :]
    plot_images(images=images)


def generate(description: str = "she's woman wearing eyeglasses"):

    text_embeddings = encode_text(description=description)[indexes_of_interest]
    target_indexes = np.where(text_embeddings > .5)[0].tolist()
    utils.block_print()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gradient_ascent_disentangled_generation(target_indexes=target_indexes, num_images=1)


if __name__ == '__main__':
    generate()
