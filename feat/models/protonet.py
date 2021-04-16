import torch.nn as nn
from feat.utils import euclidean_metric
import torch
from torch import nn

class GLVQ(nn.Module):
    def __init__(self,num_classes,num_protos,feat_dim=None,data=None):
        super().__init__()
        self.nall = num_protos *num_classes
        self.feat_dim,self.num_protos,self.num_classes =feat_dim,num_protos,num_classes
        self.protos = torch.nn.Parameter(torch.mean(data,dim=0)[None,:].repeat_interleave(num_protos*num_classes,0))
        self.num_classes = num_classes
        self.num_protos = num_protos

    def forward(self, x, encoder, metric="euclidean"):
        encoded_protos = encoder(self.protos)
        x = x.unsqueeze(1).expand(x.size(0), self.protos.size(0), x.size(-1))
        protos = encoded_protos.unsqueeze(0).expand(x.size(0), self.protos.size(0), x.size(-1))
        if metric ==  "euclidean": diff = (x - protos)
        # TODO: implement other metric
        diff = torch.norm(diff,2,dim=-1)
        diff = diff.reshape(x.size(0),self.num_protos,self.num_classes) # wta prepare
        diff,_ = torch.min(diff,dim=1) # wta 
        return diff 


class ProtoNet(nn.Module):

    def __init__(self, args, data_shape=None):
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
            self.encoder = model #torch.nn.DataParallel(model, device_ids=[ torch.device("cuda:0"), torch.device("cuda:1")])
            self.glvq = GLVQ(args.way, args.protosperclass,data=data_shape)
        else:
            raise ValueError('')

        #GLVQ
        # 1. randaug in loader - data shot augmentieren = orig_datashot + randaug_datashot

    def forward(self, data_shot, data_query, train=True):
        proto = self.encoder(data_shot)
        proto_q = self.encoder(data_query)
        
        if train:
            proto_glvq = self.glvq(proto, self.encoder)
            proto_q_glvq = self.glvq(proto_q, self.encoder)
            logits = euclidean_metric(proto_q_glvq, proto_glvq) / self.args.temperature
        else:
            proto = proto.reshape(self.args.shot, self.args.way, -1).mean(dim=0)
            logits = euclidean_metric(proto_q, proto) / self.args.temperature
        
        return logits