import torch 
import torch.nn as nn
import torchvision
import numpy as np


image_size = [1, 28, 28]
latent_dim = 96
batch_size = 64
use_gpu = torch.cuda.is_available()
use_mps = torch.backends.mps.is_built()

class Generator(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.GELU(),

            nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.GELU(),
            nn.Linear(256, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.GELU(),
            nn.Linear(512, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.GELU(),
            nn.Linear(1024, np.prod(image_size, dtype=np.int32)),
            #  nn.Tanh(),
            nn.Sigmoid(),
        )

    def forward(self, z):
        # shape of z: [batch_size, 1*28*28]
        output = self.model(z)
        output = output.view(output.size(0), *image_size)
        return output

class Discriminator(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(np.prod(image_size, dtype=np.int32), 512),
            torch.nn.GELU(),
            nn.Linear(512, 256),
            torch.nn.GELU(),
            nn.Linear(256, 128),
            torch.nn.GELU(),
            nn.Linear(128, 64),
            torch.nn.GELU(),
            nn.Linear(64, 32),
            torch.nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, image):
        # shape of image: [batch_size, 1, 28, 28]
        output = image.view(image.size(0), -1)
        prob = self.model(output)
        return prob

# training
dataset = torchvision.datasets.MNIST("data", train=True, download=True,
                                     transform=torchvision.transforms.Compose([
                                         torchvision.transforms.ToTensor(),
                                        #  torchvision.transforms.Normalize((0.5,), (0.5,))
                                     ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                         )

generator = Generator()
discriminator = Discriminator()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)

loss_fn = nn.BCELoss()
labels_one = torch.ones(batch_size, 1)
labels_zero = torch.zeros(batch_size, 1)

if use_gpu:
    print("use gpu for training")
    device = "cuda"
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    loss_fn = loss_fn.cuda()
    labels_one = labels_one.to(device)
    labels_zero = labels_zero.to(device)
elif use_mps:
    device = "mps"
    print("use mps for training")
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    loss_fn = loss_fn.to(device)
    labels_one = labels_one.to(device)
    labels_zero = labels_zero.to(device)

num_epochs = 100
for epoch in range(num_epochs):
    for i, mini_batch in enumerate(dataloader):
        gt_images, _ = mini_batch
        z = torch.randn(batch_size, latent_dim)
        gt_images = gt_images.to(device)
        z = z.to(device)
        pred_images = generator(z)
        
        optimizer_G.zero_grad()
        g_loss = loss_fn(discriminator(pred_images), labels_one)
        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        real_loss = loss_fn(discriminator(gt_images), labels_one)
        fake_loss = loss_fn(discriminator(pred_images.detach()), labels_zero)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        if i % 500 == 0:
            torchvision.utils.save_image(pred_images, f"GAN/output/{i}.png", nrow=8, normalize=True)
        
    
    print("finished epoch", epoch)
    print(f"generator loss: {g_loss.item()}, discriminator loss: {d_loss.item()}")
