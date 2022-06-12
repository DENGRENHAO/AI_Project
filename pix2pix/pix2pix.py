import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(" -- Using GPU -- ")


class Generator(nn.Module):
    def __init__(self, in_ch, out_ch, ngf=64):
        super(Generator, self).__init__()

        # U-Net encoder
        # Input: 256 * 256
        self.en1 = nn.Sequential(
            nn.Conv2d(in_ch, ngf, kernel_size=4, stride=2, padding=1),
        )
        # 128 * 128
        self.en2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2)
        )
        # 64 * 64
        self.en3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4)
        )
        # 32 * 32
        self.en4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        # 16 * 16
        self.en5 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        # 8 * 8
        self.en6 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        # 4 * 4
        self.en7 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        # 2 * 2
        self.en8 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
        )

        # U-Net decoder
        # 1 * 1（input）
        self.de1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(p=0.5)
        )
        # 2 * 2
        self.de2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(p=0.5)
        )
        # 4 * 4
        self.de3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(p=0.5)
        )
        # 8 * 8
        self.de4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(p=0.5)
        )
        # 16 * 16
        self.de5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4),
            nn.Dropout(p=0.5)
        )
        # 32 * 32
        self.de6 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.Dropout(p=0.5)
        )
        # 64 * 64
        self.de7 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2 * 2, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf),
            nn.Dropout(p=0.5)
        )
        # 128 * 128
        self.de8 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, out_ch, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, X):
        # Encoder
        en1_out = self.en1(X)
        en2_out = self.en2(en1_out)
        en3_out = self.en3(en2_out)
        en4_out = self.en4(en3_out)
        en5_out = self.en5(en4_out)
        en6_out = self.en6(en5_out)
        en7_out = self.en7(en6_out)
        en8_out = self.en8(en7_out)

        # Decoder
        de1_out = self.de1(en8_out)
        de1_cat = torch.cat([de1_out, en7_out], dim=1)  # cat by channel
        de2_out = self.de2(de1_cat)
        de2_cat = torch.cat([de2_out, en6_out], 1)
        de3_out = self.de3(de2_cat)
        de3_cat = torch.cat([de3_out, en5_out], 1)
        de4_out = self.de4(de3_cat)
        de4_cat = torch.cat([de4_out, en4_out], 1)
        de5_out = self.de5(de4_cat)
        de5_cat = torch.cat([de5_out, en3_out], 1)
        de6_out = self.de6(de5_cat)
        de6_cat = torch.cat([de6_out, en2_out], 1)
        de7_out = self.de7(de6_cat)
        de7_cat = torch.cat([de7_out, en1_out], 1)
        de8_out = self.de8(de7_cat)

        return de8_out


## Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_ch, out_ch, ndf=64):
        super(Discriminator, self).__init__()

        # Inout: 256 * 256
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch + out_ch, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 128 * 128
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 64 * 64
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 32 * 32
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 31 * 31
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )
        # Output: 30 * 30

    def forward(self, X):
        layer1_out = self.layer1(X)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)

        return layer5_out


## Data Loader
class MyDataset(Dataset):
    def __init__(self, root, subfolder, transform=None):
        super(MyDataset, self).__init__()
        self.path = os.path.join(root, subfolder)
        self.image_list = [x for x in os.listdir(self.path)]
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        image_path = os.path.join(self.path, self.image_list[item])
        image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR -> RGB
        if self.transform is not None:
            image = self.transform(image)

        lable = self.image_list[item]
        return image, lable

def loadData(root, subfolder, batch_size, shuffle=True):
    transform = transforms.Compose([
        transforms.ToTensor(),  # (H, W, C) -> (C, H, W) & (0, 255) -> (0, 1)
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # (0, 1) -> (-1, 1)
    ])
    dataset = MyDataset(root, subfolder, transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def D_train(D: Discriminator, G: Generator, X, BCELoss, optimizer_D):
    image_size = X.size(3) // 2
    x = X[:, :, :, image_size:].to(device)  # input
    y = X[:, :, :, :image_size].to(device)  # target
    xy = torch.cat([x, y], dim=1)
    # initialize
    D.zero_grad()
    # real data
    D_output_r = D(xy).squeeze()
    D_real_loss = BCELoss(D_output_r, torch.ones(D_output_r.size()).to(device))
    # fake data
    G_output = G(x)
    X_fake = torch.cat([x, G_output], dim=1)
    D_output_f = D(X_fake).squeeze()
    D_fake_loss = BCELoss(D_output_f, torch.zeros(D_output_f.size()).to(device))

    D_loss = (D_real_loss + D_fake_loss) * 0.5
    D_loss.backward()
    optimizer_D.step()

    return D_loss.data.item()


def G_train(D: Discriminator, G: Generator, X, BCELoss, L1, optimizer_G, lamb=100):
    image_size = X.size(3) // 2
    x = X[:, :, :, image_size:].to(device)  # input
    y = X[:, :, :, :image_size].to(device)  # target
    # initiallize
    G.zero_grad()
    G_output = G(x)
    X_fake = torch.cat([x, G_output], dim=1)
    D_output_f = D(X_fake).squeeze()
    G_BCE_loss = BCELoss(D_output_f, torch.ones(D_output_f.size()).to(device))
    G_L1_Loss = L1(G_output, y)
    G_loss = G_BCE_loss + lamb * G_L1_Loss
    G_loss.backward()
    optimizer_G.step()

    return G_loss.data.item()


def train(train_data, save_path):
    subfolder = ''
    batch_size = 4
    train_loader = loadData(train_data, subfolder, batch_size, shuffle=False)


    in_ch, out_ch = 3, 3
    ngf, ndf = 64, 64
    image_size = 256


    lr_G, lr_D = 0.0002, 0.0002
    beta1 = 0.5
    lamb = 100
    epochs = 200

    G = Generator(in_ch, out_ch, ngf).to(device)
    D = Discriminator(in_ch, out_ch, ndf).to(device)

    BCELoss = nn.BCELoss().to(device)
    L1 = nn.L1Loss().to(device) 
    optimizer_G = optim.Adam(G.parameters(), lr=lr_G, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=lr_D, betas=(beta1, 0.999))

    log = open(os.path.join(save_path, 'log.txt'), 'w')


    X, _ = next(iter(train_loader))
    g = G(X[:, :, :, image_size:].to(device))

    G.train()
    D.train()
    D_Loss, G_Loss, Epochs = [], [], range(1, epochs + 1)
    for epoch in range(epochs):
        D_losses, G_losses, batch, d_l, g_l = [], [], 0, 0, 0 
        for X, _ in train_loader:
            batch += 1

            D_losses.append(D_train(D, G, X, BCELoss, optimizer_D))

            G_losses.append(G_train(D, G, X, BCELoss, L1, optimizer_G, lamb))

            d_l, g_l = np.array(D_losses).mean(), np.array(G_losses).mean()
            msg = '[%d / %d]: batch#%d loss_d= %.3f  loss_g= %.3f' % (epoch + 1, epochs, batch, d_l, g_l)
            print(msg)
        log.write(msg + '\n')

        D_Loss.append(d_l)
        G_Loss.append(g_l)
    print("Done!")
    log.close()

    # save resault
    torch.save(G, os.path.join(save_path, 'generator.pkl'))
    torch.save(D, os.path.join(save_path, 'discriminator.pkl'))

    # plot loss
    plt.plot(Epochs, D_Loss, label='Discriminator Losses')
    plt.plot(Epochs, np.array(G_Loss) / 100, label='Generator Losses / 100')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss.png'))

def test(weight, input, output):
    batch_size = 1
    test_loader = loadData(input, '', batch_size, shuffle=False)
    in_ch = 3
    image_size = 256

    G = torch.load(os.path.join(weight, 'generator.pkl'))
    D = torch.load(os.path.join(weight, 'discriminator.pkl'))
    i = 0
    for X, filename in test_loader:
        g = G(X.to(device))
        save_image(g.view(batch_size, in_ch, image_size, image_size), os.path.join(output ,filename[0]))
        i+=1
