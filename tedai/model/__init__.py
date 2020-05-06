from ..imports import *

class BasicHead(nn.Module):
    def __init__(self, in_features, hidden_size):
        super(BasicHead, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(nn.BatchNorm1d(in_features*2), nn.Dropout(0.25), 
                                nn.Linear(in_features*2, in_features), nn.LeakyReLU(),
                                nn.BatchNorm1d(in_features), nn.Dropout(0.5), 
                                nn.Linear(in_features, hidden_size))

    def forward(self, x):
        x = torch.cat([self.avgpool(x), self.maxpool(x)], dim=1)
        x = self.flatten(x)
        return self.fc(x)

class TedModel(nn.Module):
    """
    class wrapper for basic architecture used with `TedLearner`
    """
    def __init__(self, arch, hidden_size, num_classes, head=None):
        super(TedModel, self).__init__()
        self.base = arch
        self.head = self.create_head(hidden_size, num_classes) if head is None else head
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_in')
            elif isinstance(m, nn.Linear): nn.init.xavier_normal_(m.weight)

    def forward(self, x): return self.head(self.base(x))

    @staticmethod
    def create_head(in_features, hidden_size): return BasicHead(in_features, hidden_size)