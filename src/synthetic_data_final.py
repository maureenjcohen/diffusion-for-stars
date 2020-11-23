# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 22:16:37 2019

@author: Maureen Cohen, University of Edinburgh
@updated: 23 November 2020

"""
from __future__ import division
import numpy as np
import cv2

def add_gaussian_noise(image_in, noise_sigma):
    
    """This function adds Gaussian noise to a grayscale or colour image.
    
    Inputs: Clean image, Gaussian noise sigma
    Output: Noisy image                                             """
    
    temp_image = np.float64(np.copy(image_in)) # Copy image, convert to float64 data type, assign to new variable

    h = temp_image.shape[0] # Number of rows of temp_image is the height
    w = temp_image.shape[1] # Number of columns of temp_image is the width
    noise = np.random.randn(h, w) * noise_sigma # Array of random numbers with same dimensions as temp_image, multiplied by sigma. St dev is 1, mean is 0

    noisy_image = np.zeros(temp_image.shape, np.float64) # Array of zeros with same dimensions as temp_image and data type float64
    if len(temp_image.shape) == 2: # If image is grayscale (only first 2 arguments of .shape attribute are defined)  
        noisy_image = temp_image + noise # Add noise to image copy
    else: # If image has more arguments
        noisy_image[:,:,0] = temp_image[:,:,0] + noise # Add noise to all elements in blue channel
        noisy_image[:,:,1] = temp_image[:,:,1] + noise # Add noise to all elements in green channel
        noisy_image[:,:,2] = temp_image[:,:,2] + noise # Add noise to all elements in red channel

    return noisy_image # Output copy of image with Gaussian noise added

"""Read in a clean image, set the background value, and add noise"""

nebula = cv2.imread('/path/to/nebula') # Read in clean image
nebula = cv2.cvtColor(nebula, cv2.COLOR_BGR2GRAY) # Convert to grayscale

background = np.zeros([200,200], dtype=np.float64) # Empty background
background[:] = 46 # Set a background value taken from e.g. similar natural images
background = add_gaussian_noise(background, 17.1) # Add Gaussian noise with 
# sigma calculated from noise level of e.g. similar natural images

""" Add background and noise to clean image"""

for i in range(len(nebula[0])):
    for j in range(len(nebula[1])):
        if nebula[i,j] < background[i,j]:
            nebula[i,j] = background[i,j]
            # Add background and noise behind the image
            
""" Add stars to noisy image """

starry_cloud = np.float64(nebula.copy())
stars = cv2.imread('path/to/stars') # Read in stars
stars = cv2.cvtColor(stars, cv2.COLOR_BGR2GRAY) # Convert to grayscale
 
for i in range(len(starry_cloud[0])):
    for j in range(len(starry_cloud[1])):
        if starry_cloud[i,j] + stars[i,j] > 255:
            starry_cloud[i,j] = 255
        # If adding oversaturates image, clip at 255
        else:
            starry_cloud[i,j] = starry_cloud[i,j] + stars[i,j]
        # If not, simply add the two images

cv2.imwrite('/path/to/output', starry_cloud)
# Save image

