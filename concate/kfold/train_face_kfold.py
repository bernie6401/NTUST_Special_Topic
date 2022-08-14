import warnings
warnings.filterwarnings("ignore")
#from apex import amp
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import os, torch
import argparse
import Networks
from dataset import *
from tqdm import trange
import wandb
import time
from sklearn.model_selection import KFold

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='../datasets/', help='Root dataset path.')
    parser.add_argument('--data_type', type=str, default='rafdb', help='rafdb or affectnet or ferplus')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--val_batch_size', type=int, default=64, help='Batch size for validation.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=1, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=50, help='Total training epochs.')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--weight_d', type=float, default=1e-4, help='Adjust weight decay')
    parser.add_argument('--gamma', type=float, default=0.8, help='Initial gamma for scheduler and the default is 0.8.')
    parser.add_argument('--step', type=int, default=10, help='Initial step for scheduler and the default is 10.')
    parser.add_argument('--num_splits', type=int, default=5, help='The total cross validation folds.')
    return parser.parse_args()


def wandb_update():
    config = wandb.config
    config.learning_rate = args.lr
    config.epochs = args.epochs
    config.optimizer = args.optimizer
    config.momentum = args.momentum
    config.batch_size = args.batch_size
    config.val_batch_size = args.val_batch_size
    config.weight_d = args.weight_d
    config.checkpoint = args.checkpoint


def main():
    '''Wandb Prepare'''
    if args.wandb:
        wandb.init(project=args.data_type.upper() + ' On HFP by ARM')
        wandb_update()
    
    '''Trainig & Val Data Prepare'''
    data_type()
    kfold = KFold(n_splits=args.num_splits)
    data = get_all_face_data()
    all_file_path, all_labels = data
    
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.558, 0.437, 0.384], std=[0.277, 0.247, 0.241]),
        transforms.RandomErasing(scale=(0.02, 0.1))])
    # train_dataset = Dataset(args.dataset_path, args.data_type, phase='train', transform=data_transforms, basic_aug=True)
    # len_train = train_dataset.__len__()
    # print('Train set size:', len_train)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.558, 0.437, 0.384], std=[0.277, 0.247, 0.241])])
    # val_dataset = Dataset(args.dataset_path, args.data_type, phase='val', transform=data_transforms_val)
    # len_val = val_dataset.__len__()
    # print('Validation set size:', len_val)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
    
    
    cross_val_accuracy_meter_top_all = []
    for fold, (train_index, test_index) in enumerate(kfold.split(all_file_path)):   #, all_labels

        '''Model & Param Prepare'''
        model = Networks.ResNet18_ARM___RAF()
        if args.checkpoint:
            print("Loading pretrained weights...", args.checkpoint)
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)


        '''Optim Prepare'''
        params = model.parameters()
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, weight_decay=args.weight_d, lr=args.lr)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=args.weight_d)
        else:
            raise ValueError("Optimizer not supported.")
        print(optimizer)


        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)
        model = model.cuda()
        #model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
        CE_criterion = torch.nn.CrossEntropyLoss()


        train_dataset = Dataset(args.dataset_path, args.data_type, phase='train', transform=data_transforms, basic_aug=True, data = data, indices=train_index)
        val_dataset = Dataset(args.dataset_path, args.data_type, phase='val', transform=data_transforms_val, data = data, indices=test_index)#np.concatenate((test_index, train_index), axis=None)
        print('train index: ', train_index, '\n', 'test index: ', test_index, '\n\n')
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)
        # train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
        # val_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)


        '''Start Training'''
        best_acc = 0
        for i in trange(1, args.epochs + 1):
            train(i, train_loader, len(train_dataset), model, CE_criterion, optimizer, scheduler)
            val_acc = validate(i, val_loader, len(val_dataset), model, CE_criterion, optimizer)
            
            if args.data_type == 'affectnet' and val_acc > 0.56 and val_acc > best_acc:
                    store_weight(i, model.state_dict(), optimizer.state_dict(), val_acc)
            elif args.data_type == 'rafdb' and val_acc > 0.90 and val_acc > best_acc:
                    store_weight(i, model.state_dict(), optimizer.state_dict(), val_acc)
            elif args.data_type == 'GEMEP' and val_acc > 0.70 and val_acc > best_acc:
                    store_weight(i, model.state_dict(), optimizer.state_dict(), val_acc, fold)
                    
            if val_acc > best_acc:
                best_acc = val_acc
                print("best_acc:" + str(best_acc))


        cross_val_accuracy_meter_top_all.append(val_acc)
        print('[Split: %02d/%02d] Accuracy: %.3f'%(fold+1, args.num_splits, np.mean(cross_val_accuracy_meter_top_all)))


def train(epoch, train_loader, len_train, model, criterion, optimizer, scheduler):
    train_loss = 0.0
    correct_sum = 0
    iter_cnt = 0
    model.train()
    for batch_i, (imgs, targets, _) in enumerate(train_loader):
    
        iter_cnt += 1
        optimizer.zero_grad()
        imgs = imgs.cuda()
        outputs, alpha = model(imgs)
        targets = targets.cuda()
    
        CE_loss = criterion(outputs, targets)
        loss = CE_loss
        #with amp.scale_loss(loss, optimizer) as scaled_loss:
            #scaled_loss.backward()
        loss.backward()
        optimizer.step()
        
        train_loss += loss
        _, predicts = torch.max(outputs, 1)
        correct_num = torch.eq(predicts, targets).sum()
        correct_sum += correct_num
            
    
    train_acc = correct_sum.float() / float(len_train)
    train_loss = train_loss/iter_cnt
    print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f LR: %.6f' %
          (epoch, train_acc, train_loss, optimizer.param_groups[0]["lr"]))
    print('Training correct sum {:} and length is {:}'.format(correct_sum, len_train))
    if args.wandb:
        wandb.log({"lr": optimizer.param_groups[0]['lr'],
                   "train_loss": train_loss,
                   "train_acc": train_acc,})
    scheduler.step()


def validate(epoch, val_loader, len_val, model, criterion, optimizer):
    with torch.no_grad():
        val_loss = 0.0
        iter_cnt = 0
        bingo_cnt = 0
        model.eval()
        for batch_i, (imgs, targets, _) in enumerate(val_loader):
            outputs, _ = model(imgs.cuda())
            targets = targets.cuda()

            CE_loss = criterion(outputs, targets)
            loss = CE_loss
    
            val_loss += loss
            iter_cnt += 1
            _, predicts = torch.max(outputs, 1)
            # if batch_i == 0:
            #     print('train predicts: ', torch.nn.functional.softmax(outputs) * 100, 'train target', targets)
            correct_or_not = torch.eq(predicts, targets)
            # if batch_i == 0:
            #     print('train correct_or_not_face: ', correct_or_not)
            bingo_cnt += correct_or_not.sum().cpu()
            
        val_loss = val_loss/iter_cnt
        val_acc = bingo_cnt.float()/float(len_val)
        val_acc = np.around(val_acc.numpy(), 4)
        print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (epoch, val_acc, val_loss))
        print('Val correct sum {:} and length is {:}'.format(bingo_cnt, len_val))
    
        if args.wandb:
            wandb.log({"val_loss": val_loss, "val_acc": val_acc})
            
    return val_acc


class AverageMeter(object): 
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def store_weight(epoch, model_state_dict, optimizer_state_dict, val_acc, fold):
    torch.save({'iter': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict, },
                 os.path.join('../models', args.data_type, 'split' + str(fold) + "_epoch" + str(epoch) + "_acc" + str(val_acc) + ".pth"))
    print('Model saved.')

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def data_type():
    try:
        if args.data_type == 'rafdb' or args.data_type == 'affectnet' or args.data_type == 'ferplus' or args.data_type == 'GEMEP':
            args.dataset_path = args.dataset_path + args.data_type + '/'
            print('The dataset is {} and the path is {}'.format(args.data_type, args.dataset_path))
        else:
            raise Exception('This dataset is not available in the model.')
    except Exception as e:
        print('You need to check your input parameter: ' + str(e))
        quit()


if __name__ == "__main__":  
    args = parse_args()                  
    main()