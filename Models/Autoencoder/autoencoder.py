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
            nn.Conv2d(in_channels=in_chan,
                      out_channels=out_chan,
                      stride=stride,
                      kernel_size=kernel_size,
                      padding=padding),
            #nn.BatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder = nn.Sequential(
            nn.UpsamplingNearest2d(),#Does this do what I want it to do?
            #nn.BatchNorm2d(),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=out_chan,
                               out_channels=in_chan,
                               stride=stride,
                               kernel_size=kernel_size,
                               padding=padding),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
