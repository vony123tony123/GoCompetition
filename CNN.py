from torch import nn

class CNN(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 19*19):
        super(CNN, self).__init__()        
        self.conv1 = nn.Sequential(         
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=7),                              
            nn.ReLU(), 
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5), 
            nn.ReLU(), 
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5), 
            nn.ReLU(), 
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3), 
            nn.ReLU(), 
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 3 * 3, out_channels),
        ) 
        self.softmax = nn.Softmax(-1) 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)     
        x = self.fc1(x)
        x = self.softmax(x)
        return x