import torch.nn as nn
from feat.utils import euclidean_metric
import torch
class ProtoNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.model_type == 'ConvNet':
            from feat.networks.convnet import ConvNet
            self.encoder = ConvNet()
        elif args.model_type == 'ResNet':
            from feat.networks.resnet import ResNet
            self.encoder = ResNet()
        elif args.model_type == 'AmdimNet':
            from feat.networks.amdimnet import AmdimNet
            model = AmdimNet(ndf=args.ndf, n_rkhs=args.rkhs, n_depth=args.nd)
            self.encoder = torch.nn.DataParallel(model, device_ids=[ torch.device("cuda:0"), torch.device("cuda:1")])
        else:
            raise ValueError('')

        #GLVQ
        # 1. randaug in loader - data shot augmentieren = orig_datashot + randaug_datashot

    def forward(self, data_shot, data_query):
        proto = self.encoder(data_shot)
        proto = proto.reshape(self.args.shot, self.args.way, -1).mean(dim=0)
        logits = euclidean_metric(self.encoder(data_query), proto) / self.args.temperature
        return logits