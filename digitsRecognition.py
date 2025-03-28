from torchvision import datasets
from torchvision.transforms import ToTensor


train_data = datasets.MNIST(
    root = 'data',
    train = True, #Loads the train set, not the eval set
    transform = ToTensor(),
    download = True # Download the data.
)

test_data = datasets.MNIST(
    root = 'data',
    train = False, #Loads the train set, not the eval set
    transform = ToTensor(),
    download = True # Download the data.
)

print(train_data)
print(test_data)
print(train_data.data.shape)


#We create the data loader  for train data and test data
from torch.utils.data import DataLoader

loaders = {
   'train': DataLoader(train_data,
                       batch_size=100,
                       shuffle=True,
                       num_workers=1,),

   'test': DataLoader(test_data,
                       batch_size=100,
                       shuffle=True,
                       num_workers=1,),
}

#We create the neural network

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) #we create a convolutional 2d layer 1 channel input and 10 channels out with kernel size 5.
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d() # dropout layer after second convolutional layer. It deactivates certain neurons during training.
        self.fc1 = nn.Linear(320,50) # fully connected layer
        self.fc2 = nn.Linear(50,10)

    def forward(self, x): #defines the activation function for each layer
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x = x.view(-1,320) # Flat
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training) #Last layer don't need activation function.
        x = self.fc2(x)

        return F.softmax(x)


# We put the model on the gpu, define optimizer and loss function.
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001) #lr = learning rate
loss_fn = nn.CrossEntropyLoss()

#We put the model into training mode and do the training.
def train(epoch):
    model.train()
    for batch_index, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # we put gradients to zero.
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward() # Backpropagation
        optimizer.step() # Finally we optimize
        if batch_index % 20 == 0:
            print(f'Train Epoch: {epoch} [{batch_index * len(data)}/{len(loaders["train"].dataset)} ({100. * batch_index / len(loaders["train"]):.0f}%)]\t{loss.item():6f}')


def test():
    model.eval()

    test_loss = 0
    correct = 0

    #We disable gradient because we are in evaluation.
    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True) #The prediction
            correct += pred.eq(target.view_as(pred)).sum().item() #We add the correct matches

    test_loss /= len(loaders['test'].dataset)
    print(f'\nTest set: Average loss: {test_loss:4.f}, Accuracy {correct}/{len(loaders["test"].dataset)} ({100. * correct / len(loaders["test"].dataset):.0f}%\n)')

    #We start the  training of 10 epoch
    for epoch in range(1, 11):
        train(epoch)
        test()

print(device)

#We test and see the first number  in test database being predicted and show by matplotlib
import matplotlib.pyplot as plt

model.eval()

data, target = test_data[0]

data = data.unsqueeze(0).to(device)

output = model(data)
prediction = output.argmax(dim=1, keepdim=True).item()

print(f'Prediction: {prediction}')

image = data.squeeze(0).squeeze(0).cpu().numpy()

plt.imshow(image, cmap='gray')
plt.show()
