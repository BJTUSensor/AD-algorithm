import numpy as np

def anisodiff2D(img,kappa, num_iter, delta_t,option):
    '''
    Anisotropic diffusion using a typical PM model
    This model performs anisotropic diffusion on a grey image, which is considered as diffusion 
    conduction of 8 adjacent nodes (directions) in a two-dimensional network structure.
    
    parameters：
            IMG         - Original image
            NUM_ITER    - Iteration number
            DELTA_T     - Diffusion coefficient (0 <=  delta_t <= 1/7), usually set to max for data stability.
            KAPPA       - Gradient modulus threshold controls the smoothness of the diffusion. The larger it is, the smoother the image would be.
            OPTION      - Choice of the diffusion functions by Perona & Malik:
                          1 - c(x,y,t) = exp(-(nablaI/Kappa)**2)
                          2 - c(x,y,t) = 1/(1 + (nalbaI/kappa)**2)
    '''

    #Distance to central pixel
    dx = 1
    dy = 1
    dd = np.sqrt(2)

    #2D Convolution Mask - Differential gradient in 8 directions
    hN = np.array([[0,1,0],[0,-1,0],[0,0,0]])
    hS = np.array([[0,0,0],[0,-1,0],[0,1,0]])
    hE = np.array([[0,0,0],[0,-1,1],[0,0,0]])
    hW = np.array([[0,0,0],[1,-1,0],[0,0,0]])
    hNE = np.array([[0,0,1],[0,-1,0],[0,0,0]])
    hSE = np.array([[0,0,0],[0,-1,0],[0,0,1]])
    hSW = np.array([[0,0,0],[0,-1,0],[1,0,0]])
    hNW = np.array([[1,0,0],[0,-1,0],[0,0,0]])
    
    #Anisotropic diffusion
    rows = img.shape[0]
    cols = img.shape[1]

    diff_img=img
    for t in range(num_iter):
  
        #Calculate the gradient in eight directions
        #nablaN = cv2.filter2D(diff_img, -1, kernel=hN)          #Keep the same depth of the target and original images
        #nablaS = cv2.filter2D(diff_img, -1, kernel=hS)
        #nablaW = cv2.filter2D(diff_img, -1, kernel=hW)
        #nablaE = cv2.filter2D(diff_img, -1, kernel=hE)
        #nablaNE = cv2.filter2D(diff_img, -1, kernel=hNE)
        #nablaSE = cv2.filter2D(diff_img, -1, kernel=hSE)
        #nablaSW = cv2.filter2D(diff_img, -1, kernel=hSW)
        #nablaNW = cv2.filter2D(diff_img, -1, kernel=hNW)

        diff1 = np.zeros((rows+2,cols+2))
        diff1[1:rows+1,1:cols+1]=diff_img
        nablaN = diff1[0:rows,1:cols+1] - diff_img
        nablaS = diff1[2:,1:cols+1] - diff_img
        nablaE = diff1[1:rows+1,2:] - diff_img
        nablaW = diff1[1:rows+1,0:cols] - diff_img
        nablaNE = diff1[0:rows,2:] - diff_img
        nablaSE = diff1[2:,2:] - diff_img
        nablaSW = diff1[2:,0:cols] - diff_img
        nablaNW = diff1[0:rows,0:cols] - diff_img

        #Diffusion function
        if option==1:
            cN = np.exp(-(nablaN/kappa)**2)
            cS = np.exp(-(nablaS/kappa)**2)
            cW = np.exp(-(nablaW/kappa)**2)
            cE = np.exp(-(nablaE/kappa)**2)
            cNE = np.exp(-(nablaNE/kappa)**2)
            cSE = np.exp(-(nablaSE/kappa)**2)
            cSW = np.exp(-(nablaSW/kappa)**2)
            cNW = np.exp(-(nablaNW/kappa)**2)
        elif option == 2:
            cN = 1 / (1 + (nablaN / kappa)**2)
            cS = 1 / (1 + (nablaE / kappa)**2)
            cW = 1 / (1 + (nablaW / kappa)**2)
            cE = 1 / (1 + (nablaE / kappa)**2)
            cNE = 1 / (1 + (nablaNE / kappa)**2)
            cSE = 1 / (1 + (nablaSE / kappa)**2)
            cSW = 1 / (1 + (nablaSW / kappa)**2)
            cNW = 1 / (1 + (nablaNW / kappa)**2)

        #Calculate with discrete partial differential equations
        delta = delta_t* (1/dy**2 * cN*nablaN +
                          1/dy**2 * cS*nablaS +
                          1/dx**2 * cW*nablaW +
                          1/dx**2 * cE*nablaE +
                          1/dd**2 * cNE*nablaNE +
                          1/dd**2 * cSE*nablaSE +
                          1/dd**2 * cSW*nablaSW +
                          1/dd**2 * cNW*nablaNW)

        diff_img = diff_img + delta
        
    return diff_img
