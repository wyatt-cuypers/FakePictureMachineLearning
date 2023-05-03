import os
import warnings
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.utils import save_image
import scipy.linalg
from scipy.stats import entropy

# 1. Choose a dataset
# dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
#                                        transform=transforms.Compose([
#                                            transforms.Resize(64),
#                                            transforms.CenterCrop(64),
#                                            transforms.ToTensor(),
#                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                                        ]))

# dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

if __name__ == '__main__':
    # Load the dataset and create DataLoader
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)


    # 2. Preprocess the data (handled in the dataset definition)

    # 3. Define the generator and discriminator
    class Generator(nn.Module):
        def __init__(self, nz=100, ngf=64, nc=3):
            super(Generator, self).__init__()
            self.main = nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
            )

        def forward(self, input):
            return self.main(input)

    class Discriminator(nn.Module):
        def __init__(self, nc=3, ndf=64):
            super(Discriminator, self).__init__()
            self.main = nn.Sequential(
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input)

    generator = Generator()
    discriminator = Discriminator()

    # 4. Define the loss function
    criterion = nn.BCELoss()

    # 5. Train the DCGAN
    epochs = 25
    lr = 0.0002
    beta1 = 0.5

    optimizerG = Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    real_label_value = 1.0
    fake_label_value = 0.0


    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            # Train Discriminator
            discriminator.zero_grad()
            real_images = data[0].to(device)
            batch_size = real_images.size(0)
            real_label = torch.full((batch_size,), real_label_value, device=device, dtype=torch.float32)
            output = discriminator(real_images).view(-1)
            errD_real = criterion(output, real_label)
            errD_real.backward()

            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake_images = generator(noise)
            fake_label = torch.full((batch_size,), fake_label_value, device=device, dtype=torch.float32)
            output = discriminator(fake_images.detach()).view(-1)
            errD_fake = criterion(output, fake_label)
            errD_fake.backward()

            errD = errD_real + errD_fake
            optimizerD.step()

            # Train Generator
            generator.zero_grad()
            output = discriminator(fake_images).view(-1)
            errG = criterion(output, real_label)
            errG.backward()
            optimizerG.step()

            # Print losses
            if i % 2 == 0:
                print(f'[{epoch+1}/{epochs}] Loss_D: {errD.item()} Loss_G: {errG.item()}')

    # 6. Evaluate the DCGAN
    def inception_score(preds, splits=10):
        scores = []
        for i in range(splits):
            part = preds[i * (preds.shape[0] // splits): (i + 1) * (preds.shape[0] // splits), :]
            kl = entropy(part, np.tile(np.mean(part, 0), (part.shape[0], 1)), base=2, axis=1)
            scores.append(np.exp(np.mean(kl)))
        return np.mean(scores), np.std(scores)

    def compute_fid(real_features, fake_features, eps=1e-6):
        mu1, mu2 = np.mean(real_features, axis=0), np.mean(fake_features, axis=0)
        sigma1, sigma2 = np.cov(real_features, rowvar=False), np.cov(fake_features, rowvar=False)

        diff = mu1 - mu2
        covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        if not np.isfinite(covmean).all():
            msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates"
            warnings.warn(msg % eps)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        return fid

    inception_model = models.inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    def get_features(images, model):
        resize_transform = transforms.Resize((299, 299))
        resized_images = torch.stack([resize_transform(img) for img in images])
        resized_images = Variable(resized_images.to(device))
        with torch.no_grad():
            pred = model(resized_images)
        return pred

    num_eval_samples = 10000
    batch_size = 128


    # Generate synthetic images
    fake_images = []
    for _ in range(num_eval_samples // batch_size):
        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        with torch.no_grad():
            generated_images = generator(noise).detach().cpu()
        fake_images.append(generated_images)

    fake_images = torch.cat(fake_images, dim=0)

    # Calculate Inception Score
    preds = []
    for i in range(num_eval_samples // batch_size):
        preds_batch = get_features(fake_images[i * batch_size:(i + 1) * batch_size], inception_model)
        preds.append(preds_batch)

    preds = np.concatenate(preds, axis=0)
    inception_mean, inception_std = inception_score(preds)
    print(f"Inception Score: {inception_mean} ± {inception_std}")

    # Calculate FID score
    real_images = []
    for i, data in enumerate(dataloader):
        real_images.append(data[0])
        if (i + 1) * batch_size >= num_eval_samples:
            break

    real_images = torch.cat(real_images, dim=0)[:num_eval_samples]
    real_features, fake_features = [], []

    for i in range(num_eval_samples // batch_size):
        real_batch = real_images[i * batch_size:(i + 1) * batch_size].to(device)
        fake_batch = fake_images[i * batch_size:(i + 1) * batch_size].to(device)
        real_features_batch = get_features(real_batch, inception_model)
        fake_features_batch = get_features(fake_batch, inception_model)
        real_features.append(real_features_batch)
        fake_features.append(fake_features_batch)

    real_features = np.concatenate(real_features, axis=0)
    fake_features = np.concatenate(fake_features, axis=0)

    fid_score = compute_fid(real_features, fake_features)
    print(f"FID Score: {fid_score}")

    # Visual inspection
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(64):
        save_image(fake_images[i], os.path.join(output_dir, f"fake_image_{i + 1}.png"))


