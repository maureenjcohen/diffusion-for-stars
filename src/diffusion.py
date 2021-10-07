# -*- coding: utf-8 -*-
"""
@author: Maureen Cohen, University of Edinburgh
@updated: 7 October 2021

Removes background stars from astronomical images using a diffusion-based
method.

This program was written in the course of a Master of Physics research project
at Heriot-Watt University, Edinburgh, under the supervision of Dr Weiping Lu.

References:
Perona, P. & Malik, J. Scale-Space and Edge Detection Using Anisotropic
Diffusion. IEEE Transactions on Pattern Analysis and Machine Intelligence 12
(7), 629-639 (1990)

Chao, S. & Tsai, D. Astronomical Image Restoration Using an Improved
Anisotropic Diffusion. Pattern Recognition Letters 27, 335-344 (2006)

Muldal, A. Implementation of Perona-Malik diffusion in Python.
https://pastebin.com/sBsPX4Y7 (2012)

"""

from __future__ import division
import numpy as np
import scipy.ndimage as sc
from skimage.measure import label

""" Initialise parameters """
avg_background_intensity = 46
# Average background intensity of image to be processed, to be empirically
# determined before processing. 46 is the average background intensity of the
# synthetic image created to test the algorithm
neighbourhood_size = 24
# Size of region in pixels to be sampled to determine whether a pixel is in a
# signal region or background region. This value can be changed without
# significant impact on the processed image.
signal_intensity_threshold = 2.5
# Regions of the image 2.5 times brighter than the background are defined as
# part of the signal - a nebula, streak, etc.
signal_size_threshold = 50
# Size threshold in pixels for considering a connected region in the relative
# residue to be signal rather than star/noise. Should be chosen to reflect the
# typical scale of signal features in the image.


def make_binary(original_img, relative_residue, iteration):
    """ This function creates the 'local stop' by transforming the relative
    residue into a binary image. A 1 in the local stop means that the
    corresponding pixel of the image being processed will be modified at the
    next time step while a 0 indicates that the pixel intensity value will not
    change. The local stop is updated at each iteration.

    Input: Original image to be processed, relative residue, iteration of
    algorithm
    Output: A binary image called the local stop which controls the next
    iteration
    of the algorithm """

    # Copies relative residue array for processing
    img = np.float64(relative_residue.copy())
    img[img < 0] = 0  # Sets negative values to 0

    # The section of code below pinpoints pixels in the signal region of the
    # original image where large changes are being made at the specified
    # iteration.

    signal = []  # Initialises list of pixels judged by the algorithm to be
    # signal
    neighbourhood = sc.uniform_filter(
        original_img, neighbourhood_size, mode='reflect')/avg_background_intensity
    # Calculates average value in a neighbourhood around each pixel in the
    # original image and divides it by the average background value to
    # determine whether the pixel is in a signal region or background region.
    # A high ratio means we are in a signal region; a ratio of 1 means we are
    # in a background region.

    xdim, ydim = img.shape
    for i in range(0, xdim):
        for j in range(0, ydim):
            if neighbourhood[i, j] >= signal_intensity_threshold:
                signal.append(img[i, j])
    # For loop checks whether each pixel in the relative residue is in a signal
    # region or background region of the original image and assigns signal
    # values to the signal list

    # Calculates average intensity of the signal appearing in the relative
    # residue
    avg_value = sum(signal)/len(signal)
    # Sets relative residue to 1 if value is greater than average
    img[img > avg_value] = 1
    img[img <= avg_value] = 0
    # Sets relative residue to 0 if value is less than average. This background
    # is ignored when labelling connected regions.

    # The section of code below checks connected regions where large changes
    # are being made to the original image at the specified iteration and
    # decides whether these regions are signal or a star based on a size
    # threshold.
    # Signal regions are then protected while stars may continue to be smoothed
    # away at the next iteration.

    stop = np.ones_like(img)

    # Labels connected pixels in the relative residue by an integer starting
    # from 1, 2, 3...
    labelled = label(img, connectivity=2)
    regions = np.amax(labelled)  # Number of connected regions

    count_list = []  # Initialises list to count how many pixels are in each
    # labelled region
    for i in range(1, (regions+1)):
        # Counts number of pixels in a connected region i
        count = np.count_nonzero(labelled == i)
        count_list.append(count)

    local_stop = stop + labelled
    for j in range(0, len(count_list)):
        if count_list[j] >= signal_size_threshold:
            local_stop[local_stop == (j+2)] = 0
            # If a connected region contains enough pixels that it must be
            # signal, the local stop is set to 0, stopping diffusion in this
            # region.
        else:
            local_stop[local_stop == (j+2)] = 1
            # Otherwise, the local stop is set to 1 and diffusion can proceed

    for i in range(0, xdim):
        for j in range(0, ydim):
            if neighbourhood[i, j] < 1.5:
                local_stop[i, j] = 1
                # The local stop is set to 1 in regions considered background,
                # as these can be diffused indefinitely without compromising
                # the signal

    return local_stop


def remove_stars(image_in, nt, gamma=0.25, step=(1., 1.)):
    """ This function removes background stars from the input image using
    a diffusion-based method with a novel diffusion coefficient. """

    img = np.float64(np.copy(image_in))  # Copies image for processing
    delta_x, delta_y = np.zeros_like(img), np.zeros_like(
        img)  # Initialises matrices for x- and y-gradients
    S, E, NS, EW = np.zeros_like(img), np.zeros_like(
        img), np.zeros_like(img), np.zeros_like(img)
    # Initialises matrix for diffusion coefficient
    diffusion_coefficient = np.ones_like(img)

    asym, asym0, asym1, asym2, asym3, asym4, asym5 = np.ones_like(img), np.ones_like(img), np.ones_like(
        img), np.ones_like(img), np.ones_like(img), np.ones_like(img), np.ones_like(img)
    # Initialises matrices for asymmetry parameter at different window sizes
    asym[1:-1, 1:-1] = np.abs(img[2:, 1:-1]-img[:-2, 1:-1]) + np.abs(img[1:-1, 2:]-img[1:-1, :-2]) + \
        np.abs(img[2:, :-2]-img[:-2, 2:]) + np.abs(img[2:, 2:]-img[:-2, :-2])
    # Calculates local asymmetry of original, unprocessed image
    # Calculates average local asymmetry of original image
    avg_asym = np.mean(asym)
    kf = 2*avg_asym  # Fixes normalisation constant using initial average local
    # asymmetry

    orig_img = img.copy()
    avg_orig = sc.uniform_filter(orig_img, 3, mode='reflect')
    binary_stop = np.ones_like(img)
    iteration = 0
    res_list = []
    second_res_list = []
    # For loop initialisations

    for timestep in range(nt):

        iteration += 1  # Updates iteration count
        # N-th discrete difference along x-axis
        delta_y[:-1, :] = np.diff(img, axis=0)
        # N-th discrete difference along y-axis
        delta_x[:, :-1] = np.diff(img, axis=1)

        avg_img = sc.uniform_filter(img, 3, mode='reflect')

        asym1[1:-1, 1:-1] = np.abs(img[2:, 1:-1]-img[:-2, 1:-1]) + np.abs(img[1:-1, 2:]-img[1:-1, :-2]) + \
            np.abs(img[2:, :-2]-img[:-2, 2:]) + \
            np.abs(img[2:, 2:]-img[:-2, :-2])
        asym2[2:-2, 2:-2] = np.abs(img[4:, 2:-2]-img[:-4, 2:-2]) + np.abs(img[2:-2, :-4] -
                                                                          img[2:-2, 4:]) + np.abs(img[4:, :-4]-img[:-4, 4:]) + np.abs(img[4:, 4:]-img[:-4, :-4])
        asym3[3:-3, 3:-3] = np.abs(img[6:, 3:-3]-img[:-6, 3:-3]) + np.abs(img[3:-3, :-6] -
                                                                          img[3:-3, 6:]) + np.abs(img[6:, :-6]-img[:-6, 6:]) + np.abs(img[6:, 6:]-img[:-6, :-6])
        asym4[4:-4, 4:-4] = np.abs(img[8:, 4:-4]-img[:-8, 4:-4]) + np.abs(img[4:-4, :-8] -
                                                                          img[4:-4, 8:]) + np.abs(img[8:, :-8]-img[:-8, 8:]) + np.abs(img[8:, 8:]-img[:-8, :-8])
        asym5[5:-5, 5:-5] = np.abs(img[10:, 5:-5]-img[:-10, 5:-5]) + np.abs(img[5:-5, :-10]-img[5:-5, 10:]) + \
            np.abs(img[10:, :-10]-img[:-10, 10:]) + \
            np.abs(img[10:, 10:]-img[:-10, :-10])
        # Calculates local asymmetry in 5 window sizes

        asymavg = ((asym1+asym2+asym3+asym4+asym5)/5)
        # Averages local asymmetry over 5 window sizes

        if iteration <= 2:
            asym0 = asymavg*np.exp(((avg_img/avg_background_intensity)-1))
            # Weights average local asymmetry by enhancement factor
        else:
            asym0 = asymavg * \
                np.exp(((avg_img/avg_background_intensity)/(binary_stop)-1))
            # From iteration 3, weights average local asymmetry by enhancement
            # factor and applies local stop constructed using the make_binary
            # function above

        diffusion_coefficient = (1./(1. + (asym0/kf))/step[0])
        # Calculates diffusion coefficient

        S[:] = diffusion_coefficient*delta_y
        E[:] = diffusion_coefficient*delta_x
        # Multiplies image gradient by new diffusion coefficient

        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]
        # Subtracts a copy that has been shifted 'North/West' by one
        # pixel. Don't ask questions. Just do it. Trust me.
        # (comment courtesy of Muldal 2012)

        img += gamma*(NS+EW)  # Updates image

        avg_imgnew = sc.uniform_filter(img, 3, mode='reflect')
        res = np.abs(avg_orig - avg_imgnew)
        res_list.append(res)
        # Calculates residue by differencing original image and updated image
        # Stores residue in a list

        if iteration > 1:
            second_res = res_list[iteration-1] - res_list[iteration-2]
            second_res_list.append(second_res)
            # Calculates relative residue by differencing two absolute residues
            # Stores relative residue in a list
        if iteration > 4:
            avg_second_res = (second_res_list[iteration-2] + second_res_list[iteration-3] +
                              second_res_list[iteration-4] + second_res_list[iteration-5])/4
            binary_stop = make_binary(orig_img, avg_second_res, iteration)
            # Average relative residues of preceding 4 iterations
            # Inputs averaged relative residue to make_binary function to
            # construct the local stop

    return img
