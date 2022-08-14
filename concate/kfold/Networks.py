from torch.nn import functional as F
from torchvision import models
from torch import nn
import torch

'''ARM'''
class ResNet18(nn.Module):
    def __init__(self, pretrained=False, num_classes=7, drop_rate=0):
        super(ResNet18, self).__init__()
        self.drop_rate = drop_rate
        resnet = models.resnet18(pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, num_classes)


    def forward(self, x):
        x = self.features(x)
        if self.drop_rate > 0:
            x =  nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return out, out

class ResNet18_ARM___RAF(nn.Module):
    def __init__(self, pretrained=True, num_classes=12, drop_rate=0):
        super(ResNet18_ARM___RAF, self).__init__()
        self.drop_rate = drop_rate
        resnet = models.resnet18(pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # before avgpool 512x1
        self.arrangement = nn.PixelShuffle(16)
        self.stn = STN_Net()
        self.arm = Amend_raf()
        self.fc = nn.Linear(121, num_classes)


    def forward(self, x):
        x = self.stn(x)
        
        x = self.features(x)
        
        x = self.arrangement(x)

        x, alpha = self.arm(x)

        if self.drop_rate > 0:
            x = nn.Dropout(self.drop_rate)(x)

        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return out, alpha

class Amend_raf(nn.Module):  # moren
    def __init__(self, inplace=2):
        super(Amend_raf, self).__init__()
        self.de_albino = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=32, stride=8, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(inplace)
        self.alpha = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        mask = torch.tensor([]).cuda()
        createVar = locals()
        for i in range(x.size(1)):
            createVar['x' + str(i)] = torch.unsqueeze(x[:, i], 1)
            createVar['x' + str(i)] = self.de_albino(createVar['x' + str(i)])
            mask = torch.cat((mask, createVar['x' + str(i)]), 1)
        x = self.bn(mask)
        xmax, _ = torch.max(x, 1, keepdim=True)
        global_mean = x.mean(dim=[0, 1])
        xmean = torch.mean(x, 1, keepdim=True)
        xmin, _ = torch.min(x, 1, keepdim=True)
        x = xmean + self.alpha * global_mean

        return x, self.alpha

class STN_Net(nn.Module):
    def __init__(self, img_size=224, n_channel=3):
        super(STN_Net, self).__init__()

        '''Spatial transformer localization-network'''
        self.localization = nn.Sequential(
            nn.Conv2d(n_channel, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True))
		
        '''Regressor for the 3 * 2 affine matrix'''
        self.nx = ((img_size - 7 + 1) // 2 - 5 + 1) // 2
		#self.nx is for calculating the correct size of the input
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * self.nx * self.nx, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        '''Initialize the weights/bias with identity transformation'''
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    '''Spatial transformer network forward function'''
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * self.nx * self.nx)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        '''transform the input'''
        x = self.stn(x)
        return x


'''Fusing Body'''
class BodyFaceEmotionClassifier(nn.Module):
    def __init__(self, args):
        super(BodyFaceEmotionClassifier, self).__init__()
        self.args = args

        total_features_size = 0

        """ use simple dnn for modeling the skeleton """
        n = 42+42+50 # this is the number of openpose skeleton joints: 21 2D points for hands and 25 2D points for body

        self.static = nn.Sequential(nn.Linear(n, args.first_layer_size), nn.ReLU())

        total_features_size += args.first_layer_size

        self.bn2 = nn.BatchNorm1d(total_features_size)
        self.bn_body = nn.BatchNorm1d(128)
        self.bn_face = nn.BatchNorm1d(2048)
        self.classifier = nn.Sequential(nn.Linear(total_features_size, args.num_classes),)

    def forward(self, inp, get_features=False):
        body, hand_right, hand_left, length = inp

        feats = []

        features = torch.cat((body, hand_right, hand_left), dim=2)
        features = features.view(features.size(0), features.size(1), -1, 3)

        features_positions_x = features[:, :, :, 0].clone()
        features_positions_y = features[:, :, :, 1].clone()

        confidences = features[:, :, :, 2].clone()
        t = torch.Tensor([self.args.confidence_threshold]).cuda()  # threshold for confidence of joints
        confidences = (confidences > t).float() * 1

        # make all joints with threshold lower than 
        features_positions = torch.stack((features_positions_x*confidences, features_positions_y*confidences), dim=3)

        static_features = features_positions.view(features_positions.size(0), features_positions.size(1),-1)

        static_features = self.static(static_features)

        sum_ = torch.zeros(body.size(0),static_features.size(2)).float().cuda()

        if self.args.body_pooling == "max":
            for i in range(0, body.size(0)):
                sum_[i] = torch.max(static_features[i,:length[i],:], dim=0)[0]
        elif self.args.body_pooling == "avg":
            for i in range(0, body.size(0)):
                sum_[i] = torch.sum(static_features[i,:length[i],:], dim=0)/length[i].float()

        feats.append(sum_)
 
        features = torch.cat(feats, dim=1)

        if get_features:
            return self.bn2(features)

        out = self.classifier(self.bn2(features))
        return out


'''FC Layer'''
class FcLayer(nn.Module):
    def __init__(self, args):
        super(FcLayer, self).__init__()
        self.fc = nn.Linear(2 * args.num_classes, args.num_classes)

    def forward(self, x):
        out = self.fc(x)

        return out


if __name__=='__main__':
    model = ResNet18_ARM___RAF()
    input = torch.randn(64, 3, 224, 224)
    out, alpha = model(input)
    print(out.size())