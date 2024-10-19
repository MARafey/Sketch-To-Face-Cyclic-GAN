import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import itertools
import random

# Generator architecture
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, n_residual_blocks=9):
        super().__init__()
        
        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channels, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# Discriminator architecture
class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.model(x)

class FaceSketchDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.photo_dir = os.path.join(root_dir, split, 'photos')
        self.sketch_dir = os.path.join(root_dir, split, 'sketches')
        self.photos = sorted([f for f in os.listdir(self.photo_dir) if f.endswith(('.jpg', '.png'))])
        
    def __len__(self):
        return len(self.photos)
    
    def __getitem__(self, idx):
        photo_name = self.photos[idx]
        photo_path = os.path.join(self.photo_dir, photo_name)
        sketch_path = os.path.join(self.sketch_dir, photo_name)
        
        photo = Image.open(photo_path).convert('RGB')
        sketch = Image.open(sketch_path).convert('RGB')  # Convert to RGB for consistency
        
        if self.transform:
            photo = self.transform(photo)
            sketch = self.transform(sketch)
            
        return {'A': photo, 'B': sketch}

class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        result = []
        for element in data.detach():
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                result.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    result.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    result.append(element)
        return torch.cat(result)

def train_cyclegan(root_dir, num_epochs=200, batch_size=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Models
    G_AB = Generator(input_channels=3, output_channels=3).to(device)
    G_BA = Generator(input_channels=3, output_channels=3).to(device)
    D_A = Discriminator(input_channels=3).to(device)
    D_B = Discriminator(input_channels=3).to(device)
    
    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    
    # Optimizers
    optimizer_G = optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()),
        lr=0.0002,
        betas=(0.5, 0.999)
    )
    optimizer_D = optim.Adam(
        itertools.chain(D_A.parameters(), D_B.parameters()),
        lr=0.0002,
        betas=(0.5, 0.999)
    )
    
    # Learning rate schedulers
    lr_scheduler_G = optim.lr_scheduler.LambdaLR(
        optimizer_G, 
        lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / float(100)
    )
    lr_scheduler_D = optim.lr_scheduler.LambdaLR(
        optimizer_D,
        lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / float(100)
    )
    
    # Buffers
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()
    
    # Dataset
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = FaceSketchDataset(root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Training
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)
            
            # Generate fake samples
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)
            
            # Train Generators
            optimizer_G.zero_grad()
            
            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) * 5.0
            
            # GAN loss
            loss_GAN_AB = criterion_GAN(D_B(fake_B), torch.ones_like(D_B(fake_B)))
            loss_GAN_BA = criterion_GAN(D_A(fake_A), torch.ones_like(D_A(fake_A)))
            loss_GAN = loss_GAN_AB + loss_GAN_BA
            
            # Cycle loss
            recovered_A = G_BA(fake_B)
            recovered_B = G_AB(fake_A)
            loss_cycle_A = criterion_cycle(recovered_A, real_A)
            loss_cycle_B = criterion_cycle(recovered_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) * 10.0
            
            # Total generator loss
            loss_G = loss_GAN + loss_cycle + loss_identity
            loss_G.backward()
            optimizer_G.step()
            
            # Train Discriminators
            optimizer_D.zero_grad()
            
            # Real loss
            loss_real_A = criterion_GAN(D_A(real_A), torch.ones_like(D_A(real_A)))
            loss_real_B = criterion_GAN(D_B(real_B), torch.ones_like(D_B(real_B)))
            
            # Fake loss
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake_A = criterion_GAN(D_A(fake_A_.detach()), torch.zeros_like(D_A(fake_A_)))
            loss_fake_B = criterion_GAN(D_B(fake_B_.detach()), torch.zeros_like(D_B(fake_B_)))
            
            # Total discriminator loss
            loss_D_A = (loss_real_A + loss_fake_A) * 0.5
            loss_D_B = (loss_real_B + loss_fake_B) * 0.5
            loss_D = loss_D_A + loss_D_B
            loss_D.backward()
            optimizer_D.step()
            
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")
        
        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D.step()
        
        # Save models
        if (epoch + 1) % 10 == 0:
            torch.save({
                'G_AB': G_AB.state_dict(),
                'G_BA': G_BA.state_dict(),
                'D_A': D_A.state_dict(),
                'D_B': D_B.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'epoch': epoch
            }, f'checkpoint_{epoch+1}.pth')

def generate_images(model_path, input_path, output_path, direction='AtoB'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    G_AB = Generator().to(device)
    G_BA = Generator().to(device)
    
    checkpoint = torch.load(model_path)
    G_AB.load_state_dict(checkpoint['G_AB'])
    G_BA.load_state_dict(checkpoint['G_BA'])
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load and process input image
    img = Image.open(input_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if direction == 'AtoB':
            output = G_AB(img)
        else:
            output = G_BA(img)
    
    # Save output
    output = output.cpu().squeeze(0)
    output = output * 0.5 + 0.5  # Denormalize
    transforms.ToPILImage()(output).save(output_path)

if __name__ == "__main__":
    root_dir = "archive"
    train_cyclegan(root_dir)
