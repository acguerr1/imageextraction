import numpy as np
import warnings
import cv2
# import Image



def anisotropic_diff(img='', niter=1,kappa=50, gamma = 0.1, step=(1.0,1.0), option = 1, ploton=False):
    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)

    img = img.astype('float32')
    imgout = img.copy()
    
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    if ploton:
        import matplotlib.pyplot as plt
        from time import sleep

        fig = plt.figure(figsize=(20, 5.5), num="Anisotropic diffusion")
        ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

        ax1.imshow(img, interpolation='nearest')
        ih = ax2.imshow(imgout, interpolation='nearest', animated=True)
        ax1.set_title("Original image")
        ax2.set_title("Iteration 0")

        plt.show()


    for ii in range(niter):

        # calculate the diffs
        deltaS[:-1,: ] = np.diff(imgout,axis=0)
        deltaE[: ,:-1] = np.diff(imgout,axis=1)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaS/kappa)**2.)/step[0]
            gE = np.exp(-(deltaE/kappa)**2.)/step[1]
        elif option == 2:
            gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
            gE = 1./(1.+(deltaE/kappa)**2.)/step[1]

        # update matrices
        E = gE*deltaE
        S = gS*deltaS

        NS[:] = S
        EW[:] = E
        NS[1:,:] -= S[:-1,:]
        EW[:,1:] -= E[:,:-1]

        imgout += gamma*(NS+EW)

        if ploton:
            iterstring = "Iteration %i" %(ii+1)
            ih.set_data(imgout)
            ax2.set_title(iterstring)
            fig.canvas.draw()
            # sleep(0.01)
 
    return imgout

# # pages = convert_from_path('./Ketchem.pdf', 500)
# pages = convert_from_path('./Leifson.pdf', 500)
# for count, page in enumerate(pages):
#     page.save(f'out{count}.jpg', 'JPEG')

# anisotropic_diff()
img = cv2.imread('./brain-high-res-mri.jpeg')

img_filtered = anisotropic_diff(img)
cv2.imwrite('filter.jpg',img_filtered)
# img = img.mean(2)
# # imgplot2 = plt.imshow(img)
# # imgplot = plt.imshow(img_filtered)
# # plt.show()
# hash0 = imagehash.average_hash(img)
# hash1 = imagehash.average_hash(img_filtered)
# print(hash0)
# print(hash1)

