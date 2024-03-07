import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Load the MNIST dataset for both training and testing.
# The dataset contains 28x28 grayscale images of handwritten digits.
training_data = datasets.MNIST(
    root="data",  # Directory where the datasets are stored.
    train=True,  # Specifies that this dataset is used for training.
    download=True,  # Downloads the dataset from the internet if it's not available at `root`.
    transform=ToTensor(),  # Transforms the dataset images into PyTorch tensors.
)

test_data = datasets.MNIST(
    root="data",  # Directory where the datasets are stored.
    train=False,  # Specifies that this dataset is used for testing.
    download=True,  # Downloads the dataset from the internet if it's not available at `root`.
    transform=ToTensor(),  # Transforms the dataset images into PyTorch tensors.
)

# DataLoader wraps the dataset and provides mini-batches during training and testing.
train_dataloader = DataLoader(training_data, batch_size=64)  # Training DataLoader.
test_dataloader = DataLoader(test_data, batch_size=64)  # Testing DataLoader.


# Define a simple Neural Network model for classifying images from the MNIST dataset.
class NeuralNetwork(nn.Module):
    """
    A simple fully-connected neural network with two hidden layers.
    """

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = (
            nn.Flatten()
        )  # Flattens the 28x28 images into 1D tensors of 784 elements.
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),  # First layer with 512 neurons.
            nn.ReLU(),  # ReLU activation function.
            nn.Linear(512, 512),  # Second layer with 512 neurons.
            nn.ReLU(),  # ReLU activation function.
            nn.Linear(512, 10),  # Output layer with 10 neurons (one for each digit).
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.
        """
        x = self.flatten(x)  # Flattens the input images.
        logits = self.linear_relu_stack(
            x
        )  # Passes the flattened images through the network.
        return logits


model = NeuralNetwork()  # Instantiate the model.

# Define the loss function and optimizer.
loss_fn = nn.CrossEntropyLoss()  # Loss function suitable for classification.
optimizer = torch.optim.SGD(
    model.parameters(), lr=1e-3
)  # Stochastic Gradient Descent optimizer.


def train(dataloader, model, loss_fn, optimizer):
    """
    Trains the model for one epoch using the given dataloader, model, loss function, and optimizer.
    """
    size = len(dataloader.dataset)
    model.train()  # Sets the model to training mode.
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)  # Generates predictions.
        loss = loss_fn(pred, y)  # Calculates loss.

        # Backpropagation
        optimizer.zero_grad()  # Resets the gradients of model parameters.
        loss.backward()  # Backward pass to compute gradients.
        optimizer.step()  # Updates the model parameters.

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    """
    Evaluates the model's performance on the test dataset.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()  # Sets the model to evaluation mode.
    test_loss, correct = 0, 0
    with torch.no_grad():  # Disables gradient calculation to save memory and computations during inference.
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # The '.item()' converts the loss tensor to a Python number. This is accumulated to compute the total loss for the test dataset.
            # Computes the accuracy of the predictions.
            # 'pred.argmax(1)' finds the predicted class label with the highest score for each image in the batch. The '1' argument specifies
            # that we're looking across the columns (class dimension). This is compared to the actual labels 'y' to check for correctness.
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            # The comparison 'pred.argmax(1) == y' yields a boolean tensor where 'True' indicates a correct prediction.
            # '.type(torch.float)' converts the boolean tensor to floats of 0.0 (for False) and 1.0 (for True), allowing us to sum up
            # the correct predictions. The '.sum().item()' converts the tensor holding the sum of correct predictions to a Python number.
            # This accumulated sum is then used to calculate the accuracy of the model on the test dataset.

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


# Training loop
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
