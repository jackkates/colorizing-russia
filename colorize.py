# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.measure as skmeasure
import skimage.transform as sktransform
from skimage import filters


OFFSET = 15
def normalize(channel):
    shifted = (channel - np.mean(channel))
    return shifted / np.linalg.norm(shifted)

def similarity(channel, blue):
    channel = channel.ravel()
    blue = blue.ravel()
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

def align(channel, blue, offset, row_start=0, col_start=0):
    max_row, max_col = 0, 0
    max_response = float("inf")
    for r in range(-offset, offset):
        for c in range(-offset, offset):
            row = r + row_start
            col = c + col_start
            response = similarity(shift(channel, row, col), blue)
            if response < max_response:
                max_response = response
                max_row, max_col = row, col

    print(max_row, max_col)
    return max_row, max_col

def layers(composite):
    # compute the height of each part (just 1/3 of total)
    height = int(np.floor(composite.shape[0] / 3.0))
    # separate color channels
    b = composite[:height][30:-30, 30:-30]
    g = composite[height: 2*height][30:-30, 30:-30]
    r = composite[2*height: 3*height][30:-30, 30:-30]

    return r, g, b

def process(composite, original):
    offset = OFFSET
    red_row, red_col, green_row, green_col = 0, 0, 0, 0
    scale = 8
    scaled = sktransform.rescale(composite, 1.0 / scale)
    scaled = sk.img_as_float(scaled)
    r, g, b = layers(scaled)
    while scale > 1:
        red_row, red_col = align(r, b, offset, red_row, red_col)
        green_row, green_col = align(g, b, offset, green_row, green_col)

        print("Aligned R=({}, {}) G=({}, {}) at scale={}.".format(scale * red_row,
            scale * red_col, scale * green_row, scale * green_col, scale))
        red_row *= 2
        red_col *= 2
        green_row *= 2
        green_col *= 2
        scale = int(scale / 2)
        scaled = sktransform.rescale(composite, 1.0 / scale)
        scaled = sk.img_as_float(scaled)
        r, g, b = layers(scaled)
        offset = 7

    R, G, B = layers(original)
    ar = shift(R, red_row * scale, red_col * scale)
    ag = shift(G, green_row * scale, green_col * scale)
    return np.dstack([ar, ag, B])


# name of the input file
imname = 'images/emir.tif'
im = skio.imread(imname)
im = sk.img_as_float(im)
sobel = filters.sobel(im)
aligned = process(sobel, im)

# save the image
#skio.imsave('./out/original.jpg', im_out)
skio.imsave('./out/aligned.jpg', aligned)
# display the image
#skio.imshow(im_out)
#skio.imshow(im_out_aligned)
#skio.show()
