"""
4.24.2024 - Camelyon17-WILDS display images demo
pwhsu2
"""

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
import torch

import matplotlib.pyplot as plt # for visualizing images

# dataset represents our entire dataset of histology images
# dataset = get_dataset(dataset="camelyon17", download=True)
dataset = get_dataset("camelyon17", root_dir="../../data")

# train_data represents the training dataset used to train models
# later, we will want to test how good our model is by using the
# "validation" and "test" datasets
train_data = dataset.get_subset(
    "train",
    # the transform parameter allows us to specify transformations we want on our data
    # for example, we could make the images larger or smaller at this step if we wanted to
    # for now, I'll just ask for it to turn the data into tensors
    transform=transforms.Compose(
        [transforms.ToTensor()]
    ),
)
# tensors are tables of numbers (aka matrices)
# here, tensors are used to store the pixel values of our histology images

# this train_loader will give us our data
# we don't want all the data at once since that's a lot of data (computers can handle
# only so much memory at a time)
# so we ask for batch_size=16, meaning that we get 16 images each time
train_loader = get_train_loader("standard", train_data, batch_size=16)

# this loop repeatedly asks train_loader for data (16 images each time)
for x, y, metadata in train_loader:
    # x represents your images as tensors (16 tensors for 16 images)
    # these tensors store the pixel values of the images
    # images from this dataset are in RGB format.
    # images are made of pixels, and each pixel is represented by computers
    # by 3 numbers, representing R (red), G (green), and B (blue) mixed together to
    # get the color you see.
    # for example, you can search on google "rgb color picker" to see how this works.
    # RGB values are between 0 to 255, inclusive (and as integers).
    # R, G, and B are called "color channels" (or just channels).
    # HOWEVER, often in image processing, we divide all the pixel values by 255 to "normalize" them
    # so now, the pixel values are between 0 to 1, inclusive. (as decimal numbers, aka floats).
    
    # if we don't change the size of the images, they will be 96 x 96 (96 pixels high, 96 pixels wide)
    # we can call x.shape to see what the image dimensions are
    print("shape of x: ", x.shape)
    print(x)
    # remember: we have 16 images, each 96x96, with 3 channels
    # if you are using a google colab, say "show data" instead of seeing the image to see the numbers
    # in case it happens

    print("-----------------------------")

    # alright, so y is the labels (0 = not cancer, 1 = cancer)
    # there will be 16 numbers here, each representing the labels of our images
    print("shape of y: ", y.shape)
    print(y)

    # enough talk! let's look at some images
    # I want to look at only one image...
    # here, tensors are "0-indexed", meaning that the first image is 0,
    # second image is 1, ..., sixteenth image is 15.
    img = x[0] # I ask x to give me only the first image (denoted as 0)
    print("shape of img: ", img.shape)
    # now I can do this because when I called x.shape, it gave me [16, 3, 96, 96]
    # the order is important!
    # if you are unfamiliar with accessing from arrays (i.e., the subscript [] notation),
    # google "2d array access" - tensors are just more than 2 dimensional

    # we are going to use a library called matplotlib (google it!)
    # to see our images
    # right now, img is [3, 96, 96]
    # if you look at the matplotlib documentation for the imshow function,
    # we need the image as (M, N, 3) (i.e., we need the channel dimension last)

    # permute does this
    # read the documentation before proceeding to understand what the following line means
    # https://pytorch.org/docs/stable/generated/torch.permute.html
    img = torch.permute(img, (1, 2, 0))

    # fingers crossed!!!!!
    plt.imshow(img)
    plt.savefig("img.png") # saves image to disk

    # try looking at another image! how would you get the second image from the batch?
    # how would you get the 20th image from the dataset?
    
    # break runs this loop once and stops it
    # remove it when you need to run through the entire dataset
    break

# sanity check:
# what are you seeing in the images?
# read the Camelyon17 section of the WILDS paper to learn more: https://arxiv.org/pdf/2012.07421.pdf

# what's next?
# look at the pytorch tutorials and see how you can make a neural network