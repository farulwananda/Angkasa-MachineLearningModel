import numpy as np
from skimage.feature import greycomatrix, greycoprops


properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']


def calculate_glcm(img, dists=[5], agls=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], lvl=256, sym=True,
                  norm=True):
    glcm = greycomatrix(img,
                        distances=dists,
                        angles=agls,
                        levels=lvl,
                        symmetric=sym,
                        normed=norm)
    feature = []
    glcm_props = [propery for name in properties for propery in greycoprops(glcm, name)[0]]
    for item in glcm_props:
        feature.append(item)

    return feature
