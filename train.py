import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataset_loader import CustomImageDataset


# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Define the training function
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)


def save_images(inputs, outputs, epoch, batch_number):
    random_index = random.randint(0, len(inputs) - 1)
    input_image, output_image = inputs[random_index], outputs[random_index]
    input_image = transforms.functional.to_pil_image(input_image)
    output_image = transforms.functional.to_pil_image(output_image)
    input_file_name = os.path.join(output_dir, f"image_epoch_{epoch}_batch_{batch_number}_input.jpg")
    output_file_name = os.path.join(output_dir, f"image_epoch_{epoch}_batch_{batch_number}_output.jpg")
    input_image.save(input_file_name)
    output_image.save(output_file_name)


def save_model(model, filepath):
    """Save PyTorch model to disk"""
    torch.save(model.state_dict(), filepath)


def load_model(model, filepath):
    """Load PyTorch model from disk"""
    model.load_state_dict(torch.load(filepath))
    return model


def train(model, dataloader, num_epochs):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        running_loss = 0.0
        start_time = time.time()

        for batch_count, (image1, image2, index) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(image1)
            batch_count += 1
            loss = criterion(outputs, image2)
            save_images(image2, outputs, epoch, batch_count)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        end_time = time.time()
        epoch_time = end_time - start_time
        if epoch % 10 == 0:
            save_model(model, 'model.pth')
        print('Epoch [{}/{}], Loss: {:.4f}, Time: {:.4f}s'.format(epoch + 1, num_epochs, running_loss / len(dataloader),
                                                                  epoch_time))
    save_model(model, 'model.pth')


resize_transform = transforms.Compose([
    transforms.Resize((352, 512)),
])
train_dataset = CustomImageDataset('/Users/nikhildevnani/Downloads/wm-nowm/train', resize_transform)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Train the model
mps_device = torch.device("mps")
autoencoder = Autoencoder()
autoencoder = load_model(autoencoder, 'model.pth')
train(autoencoder, train_dataloader, num_epochs=100)
