
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Load mnist data
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# create DataLoader
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# Define Model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.flattern = nn.Flatten()
        self.linear_relu = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512,10),
        )

    def forward(self, x):
            x = self.flattern(x)
            logits = self.linear_relu(x)
            return logits

model = NeuralNetwork()

#define loss
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# to train model

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)
    # executes train
    model.train()

    # now evaluation
    for batch, (X,y) in enumerate(dataloader):
        pred = model(X)

        # compute loss
        loss = loss_fn(pred, y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch*len(X)
            print(f"loss: {loss:>7f} [{current:>5d} {size:>5d}]")

def test(dataloader, model, loss):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0,0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            test_loss /= num_batches
            correct /= size
            print(f"Test error: \n Accuracy {(100*correct): 0.1f}% \n "
                  f"Avg Loss: {test_loss: >8f} \n")
epochs = 5
for i in range(epochs):
    print(f"Epoch {i +1} \n ----------------------")
    train(train_dataloader, model,loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)