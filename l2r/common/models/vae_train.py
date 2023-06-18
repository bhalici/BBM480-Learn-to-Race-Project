import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from encoder import VariationalEncoder
from decoder import Decoder

NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
LATENT_SPACE = 32


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.model_file = os.path.join('/home/racetrack/Desktop/race/l2r-starter-kit/l2r/common', 'vae_144w_42h_32latent.pth')
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        return self.decoder(z)
    
    def save(self):
        torch.save(self.state_dict(), self.model_file)
        self.encoder.save()
        self.decoder.save()
    
    def load(self):
        self.load_state_dict(torch.load(self.model_file))
        self.encoder.load()
        self.decoder.load()

def train(model, trainloader, optim):
    model.train()
    train_loss = 0.0
    for(x, _) in trainloader:
        x = x.to(device)
        x_hat = model(x)
        loss = ((x - x_hat)**2).sum() + model.encoder.kl
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss+=loss.item()
    return train_loss / len(trainloader.dataset)


def test(model, testloader):
    model.eval()
    val_loss = 0.0
    with torch.no_grad(): 
        for x, _ in testloader:
            x = x.to(device)
            encoded_data = model.encoder(x)
            x_hat = model(x)
            loss = ((x - x_hat)**2).sum() + model.encoder.kl
            val_loss += loss.item()

    return val_loss / len(testloader.dataset)


def main():

    data_dir = 'dataset/'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.ToTensor()])

    train_data = datasets.ImageFolder(data_dir+'train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir+'test', transform=test_transforms)
    
    m=len(train_data)
    train_data, val_data = random_split(train_data, [int(m-m*0.2), int(m*0.2)])
    

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)
    
    model = VariationalAutoencoder(latent_dims=LATENT_SPACE).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
  

    for epoch in range(NUM_EPOCHS):
        train_loss = train(model,trainloader, optim)
        val_loss = test(model,validloader)
        
    
    model.save()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
