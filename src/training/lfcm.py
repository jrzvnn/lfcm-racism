import torch.nn as nn 
import torch.nn.functional as F
import math 
import inceptionv3Cutie
import torch


class LFCM(nn.Module):
    def __init__(self, gpu=0):

        super().__init__()
        c = {}
        c['num_classes'] = 2
        c['lstm_hidden_state_dim'] = 150
        c['gpu'] = gpu
        self.cnn = inceptionv3Cutie.customInception3(pretrained=True, aux_logits=False)
        self.mm = FCM(c)
        self.initialize_weights()

    def forward(self, image, img_text, tweet, comment):

        i = self.cnn(image) # * 0 # CNN
        it = img_text * 0  # Img Text Input
        tt = tweet * 0   # Tweet Text Input
        tc = comment * 0 # Aggregated Tweet Comment Input
        x = self.mm(i, it, tt, tc) # Multimodal net
        return x

    def initialize_weights(self):
        for m in self.mm.modules(): # Initialize only mm weights
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



class FCM(nn.Module):

    def __init__(self, c):
        super().__init__()

        # Unimodal
        self.cnn_fc1 = BasicFC(2048, 1024)
        self.img_text_fc1 = BasicFC(c['lstm_hidden_state_dim'], 1024)
        self.tweet_text_fc1 = BasicFC(c['lstm_hidden_state_dim'], 1024)
        self.tweet_comment_fc1 = BasicFC(c['lstm_hidden_state_dim'], 1024)

        # Multimodal
        self.fc1 = BasicFC(1024*4, 2048)
        self.fc2 = BasicFC(2048, 1024)
        self.fc3 = BasicFC(1024, 512)
        self.fc4 = nn.Linear(512, c['num_classes'])

    def forward(self, i, it, tt, tc):

        # tt = F.dropout(tt, p=0.5, training=self.training)
        # it = F.dropout(it, p=0.5, training=self.training)
        # tc = F.dropout(it, p=0.5, training=self.training)

        # Separate process
        i = self.cnn_fc1(i)
        it = self.img_text_fc1(it)
        tt = self.tweet_text_fc1(tt)
        tc = self.tweet_comment_fc1(tc)

        # Concatenate
        x = torch.cat((it, tt), dim=1)
        x = torch.cat((i, x), dim=1)
        x = torch.cat((tc, x), dim=1)

        # ARCH-1 4fc
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x


class BasicFC(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
