import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self,
                 in_chan,
                 out_chan,
                 stride,
                 kernel_size,
                 padding):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      stride=1,
                      kernel_size=3,
                      padding=1),
            #nn.BatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      stride=1,
                      kernel_size=3,
                      padding=1),
            #nn.BatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      stride=1,
                      kernel_size=3,
                      padding=1),
            #nn.BatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        self.decoder = nn.Sequential(
            #nn.BatchNorm2d(),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=128,
                               stride=2,
                               kernel_size=3,
                               padding=1,
                               output_padding=1),
            #nn.BatchNorm2d(),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               stride=2,
                               kernel_size=3,
                               padding=1,
                               output_padding=1),
            #nn.BatchNorm2d(),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=32,
                               stride=2,
                               kernel_size=3,
                               padding=1,
                               output_padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=1,
                      stride=1,
                      kernel_size=3,
                      padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
