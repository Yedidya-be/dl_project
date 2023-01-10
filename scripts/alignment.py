import scipy.ndimage

from skimage.registration import phase_cross_correlation

def align(phase, channel2proj):
    shift, error, diffphase = phase_cross_correlation(phase, channel2proj,upsample_factor=100)
    shifted = scipy.ndimage.shift(channel2proj, shift)
    return shifted
