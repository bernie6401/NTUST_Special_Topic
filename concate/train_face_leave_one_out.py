import warnings
warnings.filterwarnings("ignore")
#from apex import amp
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import os, torch
import argparse
import Networks
from dataset_leave_one_out import *
from tqdm import trange
import wandb
import time
from sklearn.model_selection import KFold

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='../datasets/', help='Root dataset path.')
    parser.add_argument('--data_type', type=str, default='rafdb', help='rafdb or affectnet or ferplus')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=180, help='Batch size.')
    parser.add_argument('--val_batch_size', type=int, default=128, help='Batch size for validation.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=1, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=50, help='Total training epochs.')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--weight_d', type=float, default=1e-2, help='Adjust weight decay')
    parser.add_argument('--gamma', type=float, default=0.8, help='Initial gamma for scheduler and the default is 0.8.')
    parser.add_argument('--step', type=int, default=10, help='Initial step for scheduler and the default is 10.')
    parser.add_argument('--num_splits', type=int, default=5, help='The total cross validation folds.')
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--num_classes', type=int, default=12)
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
    
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.558, 0.437, 0.384], std=[0.277, 0.247, 0.241]),
        transforms.RandomErasing(scale=(0.02, 0.1))])
    
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.558, 0.437, 0.384], std=[0.277, 0.247, 0.241])])
    
    
    # path_face = '../models/GEMEP/20210910_1200Performer_weight/'
    path_face = '../models/GEMEP_FACE_12/'
    for performer_number in range(1, 11):

        '''Model & Param Prepare'''
        model = Networks.ResNet18_ARM___RAF(num_classes=args.num_classes)
        # if args.checkpoint:
        #     print("Loading pretrained weights...", args.checkpoint)
        #     checkpoint = torch.load(args.checkpoint)
        #     model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        
        # performer_number = 8
        if args.mode == 'test':
            # performer_number = 10
            face_checkpoints = [name for name in os.listdir(path_face)]
            for i,face_checkpoint in enumerate(face_checkpoints):
                if performer_number == int(face_checkpoint.split('_')[0][5:]):
                    path_face_checkpoint = os.path.join(path_face,face_checkpoint)
            print("Loading face pretrained weights...", path_face_checkpoint)
            model = Networks.ResNet18_ARM___RAF()
            checkpoint_face = torch.load(path_face_checkpoint)
            model.load_state_dict(checkpoint_face["model_state_dict"], strict=False)
        model = model.cuda()

        '''Optim Prepare'''
        params = model.parameters()
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, weight_decay=args.weight_d, lr=args.lr)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=args.weight_d)
        else:
            raise ValueError("Optimizer not supported.")
        # print(optimizer)


        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)
        # model = model.cuda()
        CE_criterion = torch.nn.CrossEntropyLoss()


        '''Trainig & Test Data Prepare'''
        train_dataset = Dataset(args.dataset_path, args.data_type, phase='train', transform=data_transforms, basic_aug=True, data = None, performer_number=performer_number)
        val_dataset = Dataset(args.dataset_path, args.data_type, phase='val', transform=data_transforms_val, data = None, performer_number=performer_number)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)


        '''Start Training'''
        best_acc = 0
        cross_val_accuracy_meter_top_all = []

        for i in trange(1, args.epochs + 1):
            if args.mode == 'train':
                train(i, train_loader, len(train_dataset), model, CE_criterion, optimizer, scheduler)
            val_acc = validate(val_loader, len(val_dataset), model, CE_criterion, optimizer, val_dataset, i, performer_number)
            cross_val_accuracy_meter_top_all.append(val_acc)

            # if val_acc > best_acc:
            #     current_epoch = i
            #     best_acc = val_acc
            #     best_optim = optimizer.state_dict()
            #     best_weight = model.state_dict()
            #     print("best_acc:" + str(best_acc))
            
        # store_weight(current_epoch, best_weight, best_optim, best_acc, performer_number)
        print('[Split: %02d/%02d] Accuracy: %.3f \n\n'%(performer_number, 10, np.mean(cross_val_accuracy_meter_top_all)))


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
        # print(targets.dtype)
        targets = targets.cuda()
        
        CE_loss = criterion(outputs, targets)
        loss = CE_loss
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


def validate(val_loader, len_val, model, criterion, optimizer, val_dataset, current_epoch, performer_number):
    with torch.no_grad():
        val_loss = 0.0
        iter_cnt = 0
        bingo_cnt = 0
        model.eval()

        feature_map_face =[]
        video_img_count = 0
        content, file_path, label = val_dataset.__content__()
        storage_face = np.zeros((len(label),args.num_classes))
        bingo_temp = []

        for batch_i, (imgs, targets, _) in enumerate(val_loader):
            outputs, _ = model(imgs.cuda())
            targets = targets.cuda()
            face_percentage = torch.nn.functional.softmax(outputs, dim=1)
            for i in range(len(face_percentage.detach().cpu().numpy())):
                feature_map_face.append(face_percentage.detach().cpu().numpy()[i])
                
            CE_loss = criterion(outputs, targets)
            loss = CE_loss
    
            val_loss += loss
            iter_cnt += 1

        print(len(feature_map_face))
        for i in range(len(content)):                                           #videos in total
            for j in range(args.num_classes):                                   #emotions in total
                for k in range(video_img_count, video_img_count + content[i]):  #all imgs in each video have to add
                    storage_face[i][j] += feature_map_face[k][j]
                storage_face[i][j] /= content[i]
            video_img_count += content[i]

        print(storage_face)
        _, storage_face = torch.max(torch.from_numpy(storage_face), 1)
        correct_or_not = torch.eq(storage_face, torch.Tensor(label))#predicts, targets
        
        # print('correct_or_not:', correct_or_not)
        # print('label:', label)
        bingo_cnt += correct_or_not.sum().cpu()

        for i in range(args.num_classes):
            if correct_or_not[i] == True:
                bingo_temp.append(label[i])
            
        val_loss = val_loss/iter_cnt
        val_acc = bingo_cnt.float()/float(len(label))
        val_acc = np.around(val_acc.numpy(), 4)
        print("Validation accuracy:%.4f. Loss:%.3f" % (val_acc, val_loss))
        print('Val correct sum {:} and length is {:}'.format(bingo_cnt, len(label)))
        # print('Img count: ', len_val)
    
        if args.wandb:
            wandb.log({"val_loss": val_loss, "val_acc": val_acc})
        if val_acc > 0.3 and args.mode != 'test':
            store_weight(current_epoch, model.state_dict(), optimizer.state_dict(), val_acc, performer_number, bingo_temp)
            print(str(bingo_temp))
            
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


def store_weight(epoch, model_state_dict, optimizer_state_dict, val_acc, fold, bingo_temp):
    torch.save({'iter': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict},
                 os.path.join('../models', args.data_type, 'split' + str(fold) + "_epoch" + str(epoch) + "_acc" + str(val_acc) + str(bingo_temp) + ".pth"))
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