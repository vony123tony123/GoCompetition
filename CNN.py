from torch import nn

class CNN(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 19*19):
        super(CNN, self).__init__()        
        self.conv1 = nn.Sequential(         
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=7),                              
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7), 
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5), 
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding='same'), 
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same'), 
            nn.ReLU(),
        )
        self.fc = nn.Linear(3*3*32, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.conv1:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
    

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)     
        return x