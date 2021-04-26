import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Lambda


# dataset has PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# creating training dataset
training_data = datasets.CIFAR10(
    
    root="images",
    train=True,
    download=True,
    transform=transform
)
#creating test dataset
test_data = datasets.CIFAR10(
    root="images",
    train=False,
    download=True,
    transform=transform
)

train_dataloader = DataLoader(training_data, batch_size=4, shuffle= True)
test_dataloader = DataLoader(test_data, batch_size=4,shuffle= False)

# copying class to be able assign it to the model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.con_layer = nn.Sequential(

            # Convolutional layer 1
            nn.Conv2d(3,32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # output: 64 x 16 x 16

            # Convolutional layer 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # output: 128 x 8 x 8

            # Convolutional layer 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # output: 256 x 4 x 4
        )

        # creating fully connected layer
        self.fc_layer = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )


    def forward(self, logits):
        """Perform forward."""
        
        # applying conv layers
        logits = self.con_layer(logits)
        
        # flattening 
        logits = logits.view(logits.size(0), -1)
        
        # applying fully connected layer
        logits = self.fc_layer(logits)

        return logits




device = torch.device("cuda")
loaded_model = CNN()
loaded_model.load_state_dict(torch.load("model.pth"))
loaded_model.to(device)
loaded_model.eval()

#defining test loop
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")




# creating formula for loss function
loss_fn = nn.CrossEntropyLoss()


print("Doing test on the given data...")
test_loop(test_dataloader, loaded_model, loss_fn)

print("Done!")