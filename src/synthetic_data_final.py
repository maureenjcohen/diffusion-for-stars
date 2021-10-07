# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 22:16:37 2019

@author: Maureen Cohen, University of Edinburgh
@updated: 7 October 2021

This script creates the 200x200 synthetic image used as a test data set for
the algorithm. The input images and the output synthetic image are found in
the /img directory.

"""
from __future__ import division
from skimage import io
import numpy as np


def add_gaussian_noise(image_in, noise_sigma):
    """This function adds Gaussian noise to a grayscale or colour image.

    Inputs: Clean image, Gaussian noise sigma
    Output: Noisy image                                             """

    # Copy image, convert to float64 data type, assign to new variable
    temp_image = np.float64(np.copy(image_in))

    h = temp_image.shape[0]  # Number of rows of temp_image is the height
    w = temp_image.shape[1]  # Number of columns of temp_image is the width
    # Array of random numbers with same dimensions as temp_image, multiplied by
    # sigma. St dev is 1, mean is 0
    noise = np.random.randn(h, w) * noise_sigma

    # Array of zeros with same dimensions as temp_image and data type float64
    noisy_image = np.zeros(temp_image.shape, np.float64)
    # If image is grayscale (only first 2 arguments of .shape attribute are
    # defined)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise  # Add noise to image copy
    else:  # If image has more arguments
        # Add noise to all elements in blue channel
        noisy_image[:, :, 0] = temp_image[:, :, 0] + noise
        # Add noise to all elements in green channel
        noisy_image[:, :, 1] = temp_image[:, :, 1] + noise
        # Add noise to all elements in red channel
        noisy_image[:, :, 2] = temp_image[:, :, 2] + noise

    return noisy_image  # Output copy of image with Gaussian noise added


"""Read in a clean image, set the background value, and add noise"""

nebula = io.imread('/path/to/nebula/img/figure2a.jpg', as_gray=True)
# Read in starless nebula as grayscale image

background = np.zeros([nebula.shape[0], nebula.shape[1]], dtype=np.float64)
# Empty background

# Set a background value taken from e.g. similar natural images
background[:] = 46
background = add_gaussian_noise(background, 17.1)  # Add Gaussian noise with
# sigma calculated from noise level of e.g. similar natural images

""" Add background and noise to clean image"""

for i in range(len(nebula[0])):
    for j in range(len(nebula[1])):
        if nebula[i, j] < background[i, j]:
            nebula[i, j] = background[i, j]
            # Add background and noise behind the image

""" Add stars to noisy image """

starry_cloud = np.float64(nebula.copy())
stars = io.imread('path/to/stars/img/figure2c.jpg', as_gray=True)
# Read in stars as grayscale image

for i in range(len(starry_cloud[0])):
    for j in range(len(starry_cloud[1])):
        if starry_cloud[i, j] + stars[i, j] > 255:
            starry_cloud[i, j] = 255
        # If adding oversaturates image, clip at 255
        else:
            starry_cloud[i, j] = starry_cloud[i, j] + stars[i, j]
        # If not, simply add the two images

io.imsave('/path/to/output', starry_cloud)
# Save image
