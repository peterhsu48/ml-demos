"""
4.25.2024 - Camelyon17-WILDS training
pwhsu2
"""

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from wilds.common.data_loaders import get_eval_loader
import torchvision.transforms as transforms
import torch
import torch.nn as nn

# step 1: get dataset
dataset = get_dataset(dataset="camelyon17", download=True)

# step 2: get train and validation subsets
train_data = dataset.get_subset(
    "train",
    transform=transforms.Compose(
        [transforms.ToTensor()]
    ),
)

# this is the validation subset
# read more about subsets here: https://developers.google.com/machine-learning/crash-course/training-and-test-sets/video-lecture
# and https://developers.google.com/machine-learning/crash-course/validation/another-partition
val_data = dataset.get_subset(
    "val",
    transform=transforms.Compose(
        [transforms.ToTensor()]
    ),
)

# step 3: get train and validation loaders (which gives us images from our subsets in smaller batches)
batch_size = 16
train_loader = get_train_loader("standard", train_data, batch_size=batch_size)
val_loader = get_eval_loader("standard", val_data, batch_size=batch_size)

# step 4: specify a device variable
# we often train models on GPUs (hardware that's really fast at matrix operations but not as versatile as CPUs)
# the device variable below denotes whether a GPU is available or not
# we need a GPU to be available otherwise training will take a long time
# we will use this variable later (usually to tell pytorch to put tensors onto our GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# step 5: create a (very) simple binary classifier neural network
# learn about what a neural network is first: https://www.youtube.com/watch?v=aircAruvnKk&pp=ygUObmV1cmFsIG5ldHdvcms%3D
# google about python classes if you are unfamilar
class Model(nn.Module):

  # Constructor
  # __init__ essentially defines what layers of our neural network we are going to use
  def __init__(self):
    super().__init__()

    # learn about convolution here: https://developers.google.com/machine-learning/practica/image-classification
    # I recommend this video series: https://developers.google.com/machine-learning/crash-course/ml-intro
    # to learn more about the other layers
    # But play around with the numbers and number of layers and see what happens!

    # Convolutional layers
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
    self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

    # Pooling layer
    self.pool = nn.MaxPool2d(kernel_size=2)

    # ReLU layer
    self.relu = nn.ReLU()

    # Flatten
    self.flatten = nn.Flatten()

    # Fully connected layers
    self.fc1 = nn.Linear(6400, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 2)

    # Sigmoid
    self.sigmoid = nn.Sigmoid()

  # forward uses the layers we defined in __init__ and specifies what order they should be arranged in
  # this is the actually function that gets called when we put our image data into the model
  def forward(self, x):
    x = self.pool(self.relu(self.conv1(x)))
    x = self.pool(self.relu(self.conv2(x)))
    x = self.pool(self.relu(self.conv3(x)))
    x = self.flatten(x)
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.sigmoid(self.fc3(x))
    x = nn.Flatten(start_dim=0)(x)
    return x

# creates the model and sends it to the GPU
model = Model().to(device)

# google what learning rate and epoch are
learning_rate = 0.01
epochs = 3

# step 6: define a loss_fn (loss function)
# loss functions tell us how good our model is (e.g., if our model predicts very strongly
# that an image is non-cancerous but it in reality is cancerous, then the loss function
# will tell us that we have "high loss" and that the model is not doing well, whereas
# if our model predicts very strongly that the image is cancerous and it is indeed cancerous
# then we will have low loss)
# the amount of loss informs us how we should train the model (kind of like getting a test score
# from an exam and changing your study strategies to get a better test score next time)
loss_fn = torch.nn.CrossEntropyLoss() # look up what a cross entropy loss is

# step 7: define an optimizer
# optimizers deal with updating model parameters
# model parameters define how the model behaves
# our goal in all of this is to find the best parameters that makes a model
# that classifies histology images well
# in the beginning, the model parameters are completely random and nonsense
# training helps to adjust the parameters again and again and again
# until we get good parameters that yield low loss results
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate) # look up what RMS prop is
# how does updating model parameters work? that's a more involved
# question but really really interesting
# see backpropagation if you are interested

# this function will train your model
def train(dataset, model, loss_fn, optimizer, epochs, batch_size):
  
  loss_total = 0.0
  
  model.train()

  # same loop as before when we displayed the images!!!!
  for batch, (x, y, metadata) in enumerate(dataset):

    x = x.to(device)
    y = y.to(device)

    # Input training data into the model and calculate the loss
    pred = model(x)
    loss = loss_fn(pred, y.to(torch.float))

    # Backpropagation steps (update model parameters)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_total += loss.item()

    if (batch % 100 == 0):
      print("Training Batch {0} Loss: {1:0.3}".format(batch, loss.item()))
        
  return loss_total / len(train_loader)

# this function performs validation
# think about it this way: you are studying for an exam
# studying = training a model
# taking a practice test to see how you're doing = model validation
# taking the actual test = model testing
def validation(dataset, model, loss_fn, batch_size):
  
  model.eval()

  with torch.no_grad():

    loss_total = 0.0

    # Go through the validation data in batches
    for batch, (x, y, metadata) in enumerate(dataset):

      x = x.to(device)
      y = y.to(device)

      # Predict and calculate loss on validation data
      pred = model(x)
      loss = loss_fn(pred, y.to(torch.float))

      loss_total += loss.item()

  return loss_total / len(val_loader)

# training
for epoch in range(epochs):

  print("---------------------------")

  print("Epoch", epoch + 1)
  train_loss = train(train_loader, model, loss_fn, optimizer, epochs, batch_size)
  val_loss = validation(val_loader, model, loss_fn, batch_size)

  print("Training loss for this epoch: {0}".format(train_loss))
  print("Validation loss for this epoch: {0}".format(val_loss))

  print("---------------------------")