"""

@author: Mo Cohen, Heriot-Watt University

Removes background stars from astronomical images. Based on the Perona-Malik anisotropic method with modifications by Chao & Tsai.

References: 
Perona, P. & Malik, J. Scale-Space and Edge Detection Using Anisotropic Diffusion. IEEE Transactions on Pattern Analysis and Machine Intelligence 12 (7), 629-639 (1990)
Chao, S. & Tsai, D. Astronomical Image Restoration Using an Improved Anisotropic Diffusion. Pattern Recognition Letters 27, 335-344 (2006)
Muldal, A. Implementation of Perona-Malik diffusion in Python. https://pastebin.com/sBsPX4Y7 (2012)

"""

from __future__ import division
import numpy as np
import cv2
import scipy.ndimage as sc

def diffusion(image_in, nt, option, gamma=0.25, step=(1.,1.)): 
    
    img = np.float64(np.copy(image_in)) # Copy of image in float64 data format to perform calculations
    deltaS = np.zeros_like(img) # Zero matrix, same size as image
    deltaE = deltaS.copy() # Zero matrix, same size as image
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(img) # Matrix of ones, same size as image
    gE = gS.copy()
    Icon, peak_avg, surr_avg = deltaS.copy(), deltaS.copy(), deltaS.copy()

    
    for i in range(nt):
        deltaS[:-1,: ] = np.diff(img,axis=0) # N-th discrete difference along x-axis
        deltaE[: ,:-1] = np.diff(img,axis=1) # N-th discrete difference along y-axis
#        Imin = sc.minimum_filter(img, 3, mode='reflect')
        if option == 1:
            peak_avg[2:-2,2:-2] = (img[2:-2,2:-2] + img[1:-3,2:-2] + img[3:-1:,2:-2] + img[2:-2,1:-3] + img[2:-2,3:-1])/5
            surr_avg[2:-2,2:-2] = (img[:-4,2:-2]+img[:-4,1:-3]+img[:-4,:-4]+img[:-4,3:-1]+img[:-4,4:]+img[1:-3,4:]+img[2:-2,4:]+img[3:-1,4:]+img[4:,4:]+img[4:,3:-1]+img[4:,2:-2]+img[4:,1:-3]+img[4:,:-4]+img[3:-1,:-4]+img[2:-2,:-4]+img[1:-3,:-4]+img[1:-3,1:-3]+img[3:-1,1:-3]+img[3:-1,3:-1]+img[1:-3,3:-1])/20           
            peak_avg[peak_avg==0] = 1
            surr_avg[surr_avg==0] = 1
            Icon = peak_avg/surr_avg
            Icon[Icon==0] = 1
#            Icon = ((peak_avg-surr_avg)**2 + 1)/(peak_avg+surr_avg)#            
            wmean, wsqrmean = (cv2.boxFilter(x, -1, (3,3), borderType=cv2.BORDER_REFLECT) for x in (img, img*img))
            local_var = wsqrmean - wmean*wmean
            local_var[local_var==0] = 1
            kappa = 0.05
            kappaprime = (Icon**2)/(1+Icon)
            kappa0 = kappa*kappaprime

            np.seterr(divide='ignore', invalid='ignore')   
            gS = 1./(1.+(deltaS/(kappa0*local_var))**2.)/step[0] # Diffusion coefficient function
            gE = 1./(1.+(deltaE/(kappa0*local_var))**2.)/step[1]
            
        elif option == 2:
            # Base version by Chao & Tsai
            wmean, wsqrmean = (cv2.boxFilter(x, -1, (3, 3), borderType=cv2.BORDER_REFLECT) for x in (img, img*img))
            local_var = wsqrmean - wmean*wmean
            local_var[local_var==0] = 1
            kappa0 = 0.05
            
            np.seterr(divide='ignore', invalid='ignore')   
            gS = 1./(1.+(deltaS/(kappa0*local_var))**2.)/step[0] # Diffusion coefficient function
            gE = 1./(1.+(deltaE/(kappa0*local_var))**2.)/step[1]
            
        
        S = gS*deltaS
        E = gE*deltaE
        NS[:] = S
        EW[:] = E
        NS[1:,:] -= S[:-1,:]
        EW[:,1:] -= E[:,:-1]
        
        img_change = (NS+EW)

        img += gamma*img_change        

    return img
