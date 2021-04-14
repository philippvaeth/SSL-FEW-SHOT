import argparse
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from feat.dataloader.samplers import CategoriesSampler
from feat.models.protonet import ProtoNet
from feat.utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, compute_confidence_interval
from tensorboardX import SummaryWriter
#from dda.data.transformations import randaugment
#from randaugment.randaugment import ImageNetPolicy
#from torchsummary import summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=128)
    parser.add_argument('--model_type', type=str, default='AmdimNet', choices=['ConvNet', 'ResNet', 'AmdimNet'])
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['MiniImageNet', 'CUB', 'TieredImageNet'])
    # MiniImageNet, ConvNet, './saves/initialization/miniimagenet/con-pre.pth'
    # MiniImageNet, ResNet, './saves/initialization/miniimagenet/res-pre.pth'
    # CUB, ConvNet, './saves/initialization/cub/con-pre.pth'
    parser.add_argument('--init_weights', type=str, default='./saves/initialization/miniimagenet/mini_imagenet_ndf192_rkhs1536_rd8_ssl_cpt.pth')
    parser.add_argument('--save_path', type=str, default="./saves")
    parser.add_argument('--gpu', default='2')
    parser.add_argument('--protosperclass', type=int, default=5)

    # AMDIM Modelrd
    parser.add_argument('--ndf', type=int, default=192)
    parser.add_argument('--rkhs', type=int, default=1536)
    parser.add_argument('--nd', type=int, default=8)


    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    save_path1 = '-'.join([args.dataset, args.model_type, 'ProtoNet'])
    save_path2 = '_'.join([str(args.shot), str(args.query), str(args.way), 
                               str(args.step_size), str(args.gamma), str(args.lr), str(args.temperature)])
    args.save_path = osp.join(args.save_path, osp.join(save_path1, save_path2))
    ensure_path(save_path1, remove=False)
    ensure_path(args.save_path)  

    if args.dataset == 'MiniImageNet':
        # Handle MiniImageNet
        from feat.dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'CUB':
        from feat.dataloader.cub import CUB as Dataset
    elif args.dataset == 'TieredImageNet':
        from feat.dataloader.tiered_imagenet import tieredImageNet as Dataset       
    else:
        raise ValueError('Non-supported Dataset.')

    # randaugment= randaugment(randaugment_type=ImageNetPolicy())
    #
    # def ParallelLoader(randaugment, **kwargs):
    #     def __init__():
    #         self.dataloader = DataLoader(**kwargs)
    #         self.randaugment = randaugment
    #
    #     def __iter__(self):
    #         return dataloader
    #
    #     def __next__(self):
    #         batch = next(self.dataloader)
    #         backup = copy.deepcopy(sample)
    #         if self.randaugment is not None:
    #             rand_sample = self.randaugment(backup)
    #             return batch, rand_sample
    #         return [0,1,2,3]

    def check_inf_nan(input):
        if torch.count_nonzero(torch.isinf(input).int()) > 0:
            print("Inf detected")
        if torch.count_nonzero(torch.isnan(input).int()) > 0:
            print("NaN detected")

    #check_inf_nan(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')])) # method test

    trainset = Dataset('train', args)
    train_sampler = CategoriesSampler(trainset.label, 50, args.way, args.shot + args.query) # batch original 100
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=0, pin_memory=True)



    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label, 250, args.way, args.shot + args.query) # batch original 500
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=0, pin_memory=True)
    
    model = ProtoNet(args,data_shape=next(iter(train_loader))[0])
    if args.model_type == 'ConvNet':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.model_type == 'ResNet':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    elif args.model_type == 'AmdimNet':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    else:
        raise ValueError('No Such Encoder')
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)        
    
    # load pre-trained model (no FC weights)

    model_dict = model.state_dict()

    if args.init_weights is not None:
        model_detail = torch.load(args.init_weights)
        if 'params' in model_detail:
            pretrained_dict = model_detail['params']
            # remove weights for FC
            pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            print(pretrained_dict.keys())
            model_dict.update(pretrained_dict)
        else:
            pretrained_dict = model_detail['model']
            #print(model_dict.keys())
            #print(pretrained_dict.keys())
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items() if k.replace('module.', '') in model_dict}
            #print(pretrained_dict.keys())

            model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)    
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
    
    def save_model(name):
        torch.save(dict(params=model.state_dict()), osp.join(args.save_path, name + '.pth'))
    
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['max_acc_epoch'] = 0

    timer = Timer()
    global_count = 0
    writer = SummaryWriter(logdir=args.save_path)
    
    for epoch in range(1, args.max_epoch + 1):
        lr_scheduler.step()
        model.train()
        tl = Averager()
        ta = Averager()

        label = torch.arange(args.way).repeat(args.query)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)
            
        for i, batch in enumerate(train_loader, 1):
            global_count = global_count + 1
            if torch.cuda.is_available():
                data, data_a, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            p = args.shot * args.way

            data_shot, data_query = data[:p], data[p:]
            #check_inf_nan(data_shot)
            #check_inf_nan(data_query)

            logits = model(data_shot, data_query, train=True)
            #check_inf_nan(logits)
            #summary(model, data_shot, data_query)

            data_shot_a, data_query_a = data_a[:p], data_a[p:]
            logits_a = model(data_shot_a, data_query_a, train=True)
            #check_inf_nan(logits_a)

            loss_c = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            #loss_a = F.cross_entropy(logits, logits_a)
            diss_no = torch.nn.Softmax(dim=1)(logits)
            #check_inf_nan(diss_no)
            diss_aug = torch.nn.Softmax(dim=1)(logits_a)
            #check_inf_nan(diss_aug)

           # loss_a = F.binary_cross_entropy_with_logits(logits,logits_a)/len(data)
            loss_a = F.binary_cross_entropy(diss_aug,diss_no.detach())/len(data)*100
            # Etl. F.MSE(diss_aug,diss_no.detach())/len(data)
            #loss_a = F.binary_cross_entropy(diss_no, diss_aug) / len(data)
            #loss_a = (-1 * (torch.sum(diss_no * torch.log(diss_aug))) - 1 * (
            #torch.sum(diss_no * torch.log(diss_aug)))) / len(data)
            #check_inf_nan(loss_a)

            writer.add_scalar('data/loss_c', float(loss_c), global_count)
            writer.add_scalar('data/loss_a', float(loss_a), global_count)
            writer.add_scalar('data/acc', float(acc), global_count)

            print('epoch {}, train {}/{}, loss={:.4f}, loss_a={:.4f}, acc={:.4f}'
                  .format(epoch, i, len(train_loader), loss_c.item(),loss_a, acc))

            loss = loss_a + loss_c
            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tl = tl.item()
        ta = ta.item()

        model.eval()

        vl = Averager()
        va = Averager()

        label = torch.arange(args.way).repeat(args.query)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)
            
        print('best epoch {}, best val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
        with torch.no_grad():
            for i, batch in enumerate(val_loader, 1):
                if torch.cuda.is_available():
                    data, _, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                p = args.shot * args.way
                data_shot, data_query = data[:p], data[p:]
    
                logits = model(data_shot, data_query, train=False)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)    
                vl.add(loss.item())
                va.add(acc)

        vl = vl.item()
        va = va.item()
        writer.add_scalar('data/val_loss', float(vl), epoch)
        writer.add_scalar('data/val_acc', float(va), epoch)        
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            trlog['max_acc_epoch'] = epoch
            save_model('max_acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))

        save_model('epoch-last')

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))
    writer.close()

    # Test Phase
    trlog = torch.load(osp.join(args.save_path, 'trlog'))
    test_set = Dataset('test', args)
    sampler = CategoriesSampler(test_set.label, 10000, args.way, args.shot + args.query)
    loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    test_acc_record = np.zeros((10000,))

    model.load_state_dict(torch.load(osp.join(args.save_path, 'max_acc' + '.pth'))['params'])
    model.eval()

    ave_acc = Averager()
    label = torch.arange(args.way).repeat(args.query)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
    else:
        label = label.type(torch.LongTensor)
        
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            k = args.way * args.shot
            data_shot, data_query = data[:k], data[k:]
    
            logits = model(data_shot, data_query, train=False)
            acc = count_acc(logits, label)
            ave_acc.add(acc)
            test_acc_record[i-1] = acc
            print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))
        
    m, pm = compute_confidence_interval(test_acc_record)
    print('Val Best Acc {:.4f}, Test Acc {:.4f}'.format(trlog['max_acc'], ave_acc.item()))
    print('Test Acc {:.4f} + {:.4f}'.format(m, pm))
