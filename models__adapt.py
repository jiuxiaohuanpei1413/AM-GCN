import torchvision.models as models
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
import torch.nn.functional as F



class NONLocalBlock1D(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(NONLocalBlock1D, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.theta = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        batch_size = x.size(0)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.squeeze(2)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        phi_x = phi_x.squeeze(2).transpose(0,1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        return f_div_C


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCNResnet(nn.Module):
    def __init__(self, model, num_classes, in_channel=300):
        super(GCNResnet, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)

        self.non_local_C1 = NONLocalBlock1D(in_channel)
        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.activate = nn.LeakyReLU(0.2)
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp):
        feature = self.features(feature)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)
        inp = inp[0]
        inp1 = inp.unsqueeze(2)
        adj = self.non_local_C1(inp1)
        A1 = torch.eye(80, 80).float().cuda()
        A1 = torch.autograd.Variable(A1)
        adj= adj+ A1
        adj = gen_adj_new(adj)
        x = self.gc1(inp, adj)
        x = self.activate(x)
        x = self.gc2(x, adj)
        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        A = A1.unsqueeze(0)
        adj = adj.unsqueeze(0)
        return x, adj, A

    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.features.parameters()))
        large_lr_layers = filter(lambda p: id(p) not in small_lr_layers, self.parameters())
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': large_lr_layers, 'lr': lr},
        ]


class DynamicGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, num_nodes, in_channel=300, bias=False):
        super(DynamicGraphConvolution, self).__init__()

        self.num_nodes = num_nodes

        self.static_weight = nn.Conv1d(in_features, out_features, 1)
        self.non_local_C1 = NONLocalBlock1D(in_channel)

        _adj = gen_A(num_nodes)
        self.A = Parameter(torch.from_numpy(_adj).float())

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_create_co_mat = nn.Conv1d(in_features * 2, in_features, 1)
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)

        self.activate = nn.LeakyReLU(0.2)
        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)

        self.long_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),
            nn.LeakyReLU(0.2))
        self.long_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2))

        self.con = nn.Conv1d(2048, num_nodes, 1, bias=False)

        self.fc_eq3_w = nn.Linear(num_nodes, num_nodes)
        self.fc_eq3_u = nn.Linear(num_nodes, num_nodes)
        self.fc_eq4_w = nn.Linear(in_features, in_features)
        self.fc_eq4_u = nn.Linear(in_features, in_features)

    def forward_construct_static_adj(self, inp):
        inp = inp[0]
        inp1 = inp.unsqueeze(2)
        adj = self.non_local_C1(inp1)
        A1 = torch.eye(self.num_nodes, self.num_nodes).float().cuda()
        A1 = torch.autograd.Variable(A1)
        adj = adj + A1
        static_adj = gen_adj_new(adj)
        return static_adj, A1

    def forward_static_gcn(self, feature, inp):
        static_adj_1, A1 = self.forward_construct_static_adj(inp)
        static_adj_2 = gen_adj(self.A).detach()
        static_adj = static_adj_1 + static_adj_2
        x = self.gc1(inp, static_adj)
        x = self.activate(x)
        x = self.gc2(x, static_adj)

        x = torch.matmul(feature, x)
        x = self.con(x.transpose(1, 2))
        x = x.transpose(1, 2)

        # x_ = x.transpose(0, 1)
        # x = torch.matmul(feature, x)
        # x = torch.matmul(x, x_)
        return x, static_adj, A1

    def forward_construct_short(self, x):
        ### Model global representations ###
        x_glb = self.gap(x)
        x_glb = self.conv_global(x_glb)
        x_glb = self.bn_global(x_glb)
        x_glb = self.relu(x_glb)
        x_glb = x_glb.expand(x_glb.size(0), x_glb.size(1), x.size(2))

        ### Construct the dynamic correlation matrix ###
        x = torch.cat((x_glb, x), dim=1)
        dynamic_adj = self.conv_create_co_mat(x)
        dynamic_adj = torch.sigmoid(dynamic_adj)
        return dynamic_adj

    def forward_construct_long(self, x, short_memory):
        with torch.no_grad():
            long_a = self.long_adj(x.transpose(1, 2))
            long_a = long_a.view(-1, x.size(2))
            long_w = self.long_weight(short_memory)
            long_w = long_w.view(x.size(0) * x.size(2), -1)
        x_w = short_memory.view(x.size(0) * x.size(2), -1)  # B*num_c,1024 短期记忆包含全局关系，提取相对权重关系。生成weight
        x_a = x.view(-1, x.size(2))  # B*1024, num_c, 注意力直接，注重提取个体之间出现的关系。生成adj
        # eq(3)
        av = torch.tanh(self.fc_eq3_w(x_a) + self.fc_eq3_u(long_a))
        # eq(4)
        wv = torch.tanh(self.fc_eq4_w(x_w) + self.fc_eq4_u(long_w))
        # eq(5)
        x_a = x_a + av * long_a
        x_a = x_a.view(x.size(0), x.size(2), -1)
        x_w = x_w + wv * long_w
        x_w = x_w.view(x.size(0), x.size(1), x.size(2))
        long_adj = self.long_adj(x_a)
        long_weight = self.long_weight(x_w)
        x = x + short_memory
        long_graph_feature1 = torch.mul(long_adj.transpose(1, 2), x)
        long_graph_feature2 = torch.mul(long_graph_feature1, long_weight)
        long_graph_feature2 = torch.sigmoid(long_graph_feature2)
        return long_graph_feature2

    def forward(self, x, inp):
        """ D-GCN module

        Shape:
        - Input: (B, C_in, N) # C_in: 1024, N: num_classes
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        out_static, static_adj, A1 = self.forward_static_gcn(x, inp)
        x = x + out_static  # residual
        short_memory = self.forward_construct_short(x)
        x = self.forward_construct_long(x, short_memory)
        A = A1.unsqueeze(0)
        adj = static_adj.unsqueeze(0)

        return x, A, adj


class AM_GCN(nn.Module):
    def __init__(self, model, num_classes, in_channel=300):
        super(AM_GCN, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_classes = num_classes

        self.fc = nn.Conv2d(model.fc.in_features, num_classes, (1, 1), bias=False)

        self.conv_transform = nn.Conv2d(2048, 1024, (1, 1))
        self.relu = nn.LeakyReLU(0.2)

        self.gcn = DynamicGraphConvolution(1024, 1024, num_classes)

        self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())
        self.last_linear = nn.Conv1d(1024, self.num_classes, 1)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

        # SE layers
        self.fc1 = nn.Conv2d(model.fc.in_features, model.fc.in_features // 16, kernel_size=1,
                             bias=False)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(model.fc.in_features // 16, model.fc.in_features, kernel_size=1, bias=False)

        self.pooling = nn.MaxPool2d(14, 14)

    def forward_feature(self, x):
        x = self.features(x)
        return x

    def forward_classification_sm(self, x):
        """ Get another confident scores {s_m}.

        Shape:
        - Input: (B, C_in, H, W) # C_in: 2048
        - Output: (B, C_out) # C_out: num_classes
        """
        x = self.fc(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.topk(1, dim=-1)[0].mean(dim=-1)
        return x

    def forward_sam(self, x):
        """ SAM module

        Shape:
        - Input: (B, C_in, H, W) # C_in: 2048
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        mask = self.fc(x)
        mask = mask.view(mask.size(0), mask.size(1), -1)
        mask = torch.sigmoid(mask)
        mask = mask.transpose(1, 2)

        x = self.conv_transform(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.matmul(x, mask)

        return x

    def forward_max_pool(self, x):
        x = self.pooling(x)
        return x

    def forward_SENet(self, x):
        """ SAM module

        Shape:
        - Input: (B, C_in, H, W) # C_in: 2048
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        # Squeeze
        mask = F.avg_pool2d(x, x.size(2))
        mask = F.relu(self.fc1(mask))
        mask = F.sigmoid(self.fc2(mask))

        # Excitation
        x = x * mask  # New broadcasting feature from v0.2!

        return x

    def forward_dgcn(self, x, static_adj):
        x = self.gcn(x, static_adj)
        return x

    def forward(self, x, inp):
        x = self.forward_feature(x)
        out1 = self.forward_classification_sm(x)

        v0 = self.forward_SENet(x)
        v = self.forward_sam(v0)  # B*1024*num_classes
        z, A, adj = self.forward_dgcn(v, inp)
        z = v + z
        out2 = self.last_linear(z)  # B*1*num_classes

        mask_mat = self.mask_mat.detach()
        out2 = (out2 * mask_mat).sum(-1)
        out = (out1 + out2) / 2

        return out, adj, A

    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.features.parameters()))
        large_lr_layers = filter(lambda p: id(p) not in small_lr_layers, self.parameters())
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': large_lr_layers, 'lr': lr},
        ]


def gcn_resnet101(num_classes, pretrained=True, in_channel=300):
    model = models.resnet101(pretrained=pretrained)
    return AM_GCN(model, num_classes, in_channel=in_channel)
