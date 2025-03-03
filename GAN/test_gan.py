import torch 
import torch.nn as nn
import torchvision

device = 'mps' if torch.backends.mps.is_built() else 'cpu'
print(f"Using {device} device")
torch.set_default_device(device)

image_size = (1, 28, 28)
image_dim = torch.prod(torch.tensor(image_size)).item()

class Generator(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(image_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, image_dim),
            nn.Tanh()
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
            nn.Linear(image_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
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
                                         torchvision.transforms.Normalize((0.5,), (0.5,))
                                     ]))

batch_size = 64
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                                         generator=torch.Generator(device='mps'))

generator = Generator().to(device)
discriminator = Discriminator().to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

loss_fn = nn.BCELoss().to(device)

num_epochs = 100
for epoch in range(num_epochs):
    for i, mini_batch in enumerate(dataloader):
        gt_images, _ = mini_batch
        gt_images = gt_images.to(device)
        cur_batch_size = gt_images.size(0)
        z = torch.randn(cur_batch_size, image_dim, device=device)
        # if i == 0:
        #     torchvision.utils.save_image(z.reshape(cur_batch_size, *image_size), f"GAN/output/{epoch}_raw.png", normalize=True)
        pred_images = generator(z)
        if i == 0:
            # print(pred_images[0])
            # print(pred_images[1])
            torchvision.utils.save_image(pred_images, f"GAN/output/{epoch}.png", normalize=True)
        
        optimizer_G.zero_grad()
        g_loss = loss_fn(discriminator(pred_images), torch.ones(cur_batch_size, 1))
        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        d_loss = 0.5 * (loss_fn(discriminator(gt_images), torch.ones(cur_batch_size, 1)) + \
                 loss_fn(discriminator(pred_images.detach()), torch.zeros(cur_batch_size, 1)))
        d_loss.backward()
        optimizer_D.step()
        
    
    print("finished epoch", epoch)
    print(f"generator loss: {g_loss.item()}, discriminator loss: {d_loss.item()}")
