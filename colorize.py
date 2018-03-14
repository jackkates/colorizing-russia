# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.measure as skmeasure
import skimage.transform as sktransform


OFFSET = 15
def normalize(channel):
    shifted = (channel - np.mean(channel))
    return shifted / np.linalg.norm(shifted)

def similarity(channel, blue):
    channel = channel[30:-30, 30:-30].ravel()
    blue = blue[30:-30, 30:-30].ravel()
    norm_channel = normalize(channel)
    norm_blue = normalize(blue)

    return np.sum(np.square(channel - blue))
    #return inner_product

def shift(channel, rows, cols):
    height, width = channel.shape[0], channel.shape[1]
    # Rows, cols at front, end
    rf, re, cf, ce = 0, 0, 0, 0
    if rows > 0:
        rf, re = 0, rows # remove 0 at front and rows at end
    else:
        rf, re = abs(rows), 0 # other way around

    if cols > 0:
        cf, ce = 0, cols
    else:
        cf, ce = abs(cols), 0

    trimmed = channel[rf:(height - re), cf:(width - ce)]
    padded = np.pad(trimmed, ((re, rf), (ce, cf)), 'mean')

    return padded

def align(channel, blue):
    max_row, max_col = 0, 0
    max_response = float("inf")
    for row in range(-OFFSET, OFFSET):
        for col in range(-OFFSET, OFFSET):
            response = similarity(shift(channel, row, col), blue)
            if response < max_response:
                max_response = response
                max_row, max_col = row, col

    print(max_row, max_col, max_response)
    return shift(channel, max_row, max_col)

# name of the input file
imname = 'images/cathedral.jpg'
# read in the image
im = sktransform.rescale(skio.imread(imname), 1)

# convert to double (might want to do this later on to save memory)
im = sk.img_as_float(im)

# compute the height of each part (just 1/3 of total)
height = int(np.floor(im.shape[0] / 3.0))
# separate color channels
b = im[:height][20:-20, 20:-20]
g = im[height: 2*height][20:-20, 20:-20]
r = im[2*height: 3*height][20:-20, 20:-20]

# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)

ar = align(r, b)
ag = align(g, b)
# create a color image
im_out = np.dstack([r, g, b])
im_out_aligned = np.dstack([ar, ag, b])

# save the image
fname = '/out_path/out_fname.jpg'
#skio.imsave(fname, im_out)

# display the image
#skio.imshow(im_out)
skio.imshow(im_out_aligned)
skio.show()
