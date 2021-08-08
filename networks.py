import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(w):
    """
    Initializes the weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('bn') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)

def init_hidden_state(params, device):
    return torch.randn(params['LSTM_layers'], params['bsize'], params['LSTM_hidden']).to(device), torch.randn(params['LSTM_layers'], params['bsize'], params['LSTM_hidden']).to(device)

class generator_actor(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Input Dimension: (nc) x 64 x 64
        self.conv1 = nn.Conv2d(params['nc'], params['ngf'],
            4, 2, 1, bias=False)

        # Input Dimension: (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(params['ngf'], params['ngf']*2,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params['ngf']*2)

        # Input Dimension: (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(params['ngf']*2, params['ngf']*4,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(params['ngf']*4)

        # Input Dimension: (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(params['ngf']*4, params['ngf']*8,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(params['ngf']*8)

        # Input Dimension: (ndf*8) x 4 x 4
        self.lin1 = nn.Linear(params['ngf']*8*4*4, params['nz'])

        # LSTM Input Dimension: 'nz'
        self.memory = nn.LSTM(input_size = params['nz'], hidden_size = params['LSTM_hidden'], 
                              num_layers = params['LSTM_layers'], batch_first = True)

        # Input Dimension: 'hidden'
        self.lin2 = nn.Linear(params['LSTM_hidden'], params['nz'])

    def forward(self, x, h, c):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)
        x = x.view(x.size(0), -1)
        x = self.lin1(x)
        x = x.view(x.size(0), 1, -1)

        m, (h, c) = self.memory(x, (h, c))
        m = m.view(m.size(0), -1)
        m = F.leaky_relu(m, 0.2, True)
        m = self.lin2(m)

        return x, m, h, c

class discriminator_actor(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Input Dimension: (nc) x 64 x 64
        self.conv1 = nn.Conv2d(params['nc'], params['ndf'],
            4, 2, 1, bias=False)

        # Input Dimension: (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(params['ndf'], params['ndf']*2,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params['ndf']*2)

        # Input Dimension: (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(params['ndf']*2, params['ndf']*4,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(params['ndf']*4)

        # Input Dimension: (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(params['ndf']*4, params['ndf']*8,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(params['ndf']*8)

        # Input Dimension: (ndf*8) x 4 x 4
        self.conv5 = nn.Conv2d(params['ndf']*8, 1, 4, 1, 0, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)

        x = F.sigmoid(self.conv5(x))

        return x

class generator_critic(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Input Dimension: (nc) x 64 x 64
        self.conv1 = nn.Conv2d(params['nc'], params['ngf'],
            4, 2, 1, bias=False)

        # Input Dimension: (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(params['ngf'], params['ngf']*2,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params['ngf']*2)

        # Input Dimension: (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(params['ngf']*2, params['ngf']*4,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(params['ngf']*4)

        # Input Dimension: (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(params['ngf']*4, params['ngf']*8,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(params['ngf']*8)

        # Input Dimension: (ndf*8) x 4 x 4
        self.lin1 = nn.Linear(params['ngf']*8*4*4, params['nz'])

        # Input Dimension: 'hidden'
        self.lin2 = nn.Linear(params['nz']*2, params['GC_hidden'])

        # Input Dimension: 'hidden'
        self.lin3 = nn.Linear(params['GC_hidden'], params['Generator_Q'])

    def forward(self, s, a):
        s = F.leaky_relu(self.conv1(s), 0.2, True)
        s = F.leaky_relu(self.bn2(self.conv2(s)), 0.2, True)
        s = F.leaky_relu(self.bn3(self.conv3(s)), 0.2, True)
        s = F.leaky_relu(self.bn4(self.conv4(s)), 0.2, True)
        
        s = s.view(s.size(0), -1)
        s = F.leaky_relu(self.lin1(s), 0.2, True)

        q = torch.cat((s, a), 1)
        q = F.relu(self.lin2(q))
        q = F.lin3(q)

        return q

class discriminator_critic(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Input Dimension: (nc) x 64 x 64
        self.conv1 = nn.Conv2d(params['nc'], params['ndf'],
            4, 2, 1, bias=False)

        # Input Dimension: (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(params['ndf'], params['ndf']*2,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params['ndf']*2)

        # Input Dimension: (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(params['ndf']*2, params['ndf']*4,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(params['ndf']*4)

        # Input Dimension: (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(params['ndf']*4, params['ndf']*8,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(params['ndf']*8)

        # Input Dimension: (ndf*8) x 4 x 4
        self.lin1 = nn.Linear(params['ndf']*8*4*4, params['nz'])

        # Input Dimension: 'hidden'
        self.lin2 = nn.Linear(params['nz']+1, params['DC_hidden'])

        # Input Dimension: 'hidden'
        self.lin3 = nn.Linear(params['DC_hidden'], params['Discriminator_Q'])

    def forward(self, s, a):
        s = F.leaky_relu(self.conv1(s), 0.2, True)
        s = F.leaky_relu(self.bn2(self.conv2(s)), 0.2, True)
        s = F.leaky_relu(self.bn3(self.conv3(s)), 0.2, True)
        s = F.leaky_relu(self.bn4(self.conv4(s)), 0.2, True)
        
        s = s.view(s.size(0), -1)
        s = F.leaky_relu(self.lin1(s), 0.2, True)

        q = torch.cat((s, a), 1)
        q = F.relu(self.lin2(q))
        q = F.lin3(q)

        return q

class generator_new_state(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Input is the latent vector Z.
        self.tconv1 = nn.ConvTranspose2d(params['nz'], params['ngf']*8,
            kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(params['ngf']*8)

        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(params['ngf']*8, params['ngf']*4,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params['ngf']*4)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(params['ngf']*4, params['ngf']*2,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(params['ngf']*2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(params['ngf']*2, params['ngf'],
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(params['ngf'])

        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose2d(params['ngf'], params['nc'],
            4, 2, 1, bias=False)
        #Output Dimension: (nc) x 64 x 64

    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))

        x = F.tanh(self.tconv5(x))

        return x