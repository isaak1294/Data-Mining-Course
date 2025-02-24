## You can run this file as a script from the terminal, using ipython, as:
##     ipython -i randrotate.py
##
## Or, to run from within ipython:
##     (1) Run ipython as "ipython --pylab
##     (2) Then, from the ipython command line, do "%run randrotate.py"
##
## Or, just copy the code into a Jupyter notebook :-)


import numpy as np # of course you need this
import matplotlib.pyplot as plt # for plotting
from skimage.transform import rotate # for rotating
from sklearn.datasets import fetch_lfw_people # you won't need this for the assignment

rng = np.random.default_rng() # call this once (not each time you want to do a random rotation)


# this function does a random rotation of an image (stored as a 2D array), where the number of degrees rand_degrees is drawn uniformly from [-r, r]
def randrotate(x_2d, r):
    rand_degrees = 2 * r * (rng.random() - 0.5) # uniform random rotation in [-r, r] degrees
    return rotate(x_2d, rand_degrees)



## The rest of this file just shows an example by:
##   (1) getting a dataset;
##   (2) getting one image from that dataset and reshaping it as a 2D array
##   (3) displaying the image
##   (4) randomly rotating and displaying the result (done twice, each time displaying in a new window for easy comparison)


# get the Labeled Faces in the Wild dataset to show an example (do not use this dataset for the assignment)
lfw_people = fetch_lfw_people() # download the data; it might take a minute
n_samples, h, w = lfw_people.images.shape # get some useful info for later; h = height and w = width
x = lfw_people.data[0] # store the first image in x (represented as a vector)
x_2d = x.reshape([h, w]) # reshape it so we can display and rotate

plt.figure()
plt.imshow(x_2d, cmap = plt.cm.gray) # display original image
plt.title('original face')

r = 30 # set range parameter for random rotation; rotations will be uniformly at random from -r degrees to r degrees

x_2d_rot = randrotate(x_2d, r)
plt.figure()
plt.imshow(x_2d_rot, cmap = plt.cm.gray) # display rotated image
plt.title('a first randomly rotated face')

x_2d_rot = randrotate(x_2d, r)
plt.figure()
plt.imshow(x_2d_rot, cmap = plt.cm.gray) # display rotated image
plt.title('a second randomly rotated face')
plt.show()

print("hi")
