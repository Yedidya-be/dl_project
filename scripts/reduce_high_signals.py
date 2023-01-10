from skimage import morphology

def reduce(image, footprint = 5):
    # signal = morphology.white_tophat(image, footprint)
    return image