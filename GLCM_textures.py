import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from skimage import data


image = data.camera()

grass = image[474: 474+21, 291:291+21]
sky = image[90:90+25, 380:380+25]

def ext_text(image):
    GLCM = greycomatrix(image, [5], [0, np.pi/4, np.pi/2, (3*np.pi)/4], symmetric=True, normed=True)  # "angular" mean for 0°, 45°, 90°, 135° as defined in Albregtsen
    
    cont = np.mean(greycoprops(GLCM, 'contrast'))
    diss = np.mean(greycoprops(GLCM, 'dissimilarity'))
    hom = np.mean(greycoprops(GLCM, 'homogeneity'))
    asm = np.mean(greycoprops(GLCM, 'ASM'))
    E = np.mean(greycoprops(GLCM, 'energy'))
    cor = np.mean(greycoprops(GLCM, 'correlation'))
    
    return cont, diss, hom, asm, E, cor

grass_prop = ext_text(grass)
sky_prop = ext_text(sky)

print(grass_prop)
print(sky_prop)
#plt.imshow(grass, cmap = "gray", vmin=0, vmax=255)