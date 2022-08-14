from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold,KFold
from sklearn.preprocessing import MinMaxScaler

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch

from tqdm import trange
import argparse
import wandb
import time

from dataset import *
from Networks import *
from image_utils import *

torch.manual_seed(7)
np.random.seed(7)


class EmotionRecognitionSystem():
    def __init__(self, args):
        print(args)
        self.args = args

    def run(self):
        args = self.args
        '''Wandb Prepare'''
        if args.wandb:
            wandb.init(project= args.dataset.upper() + ' On HFP by Fusing Body')
            wandb_update(args)
        
        '''test mode'''
        if args.mode == 'test':
            args.epochs = 1
            args.num_splits = 1
            args.num_total_iterations = 1

        # array to store accuracies from all 10 iterations
        self.all_iteration_accuracies = []

        self.confusion_matrix = np.empty(0)

        all_iterations_accuracy_meter_top_all = []
        all_iterations_p = []
        all_iterations_r = []
        all_iterations_f = []

        for i in range(self.args.num_total_iterations):
            self.current_iteration = i
            val_top_all, p, r, f = self.cross_validation(num_splits=args.num_splits)

            all_iterations_accuracy_meter_top_all.append(val_top_all)
            all_iterations_p.append(p)
            all_iterations_r.append(r)
            all_iterations_f.append(f)

            print('[Iteration: %02d/%02d] Top1 Accuracy: %.3f SK Prec: %.3f, SK Rec: %.3f F-Score: %.3f'
            % (i+1, args.num_total_iterations, np.mean(all_iterations_accuracy_meter_top_all), np.mean(all_iterations_p), np.mean(all_iterations_r), np.mean(all_iterations_f)))

    def get_scaler(self):
        scaler = {}
        feats = ["bodies", "hands_right", "hands_left", ]

        for x in feats:
            all_data = np.vstack(getattr(self.train_dataset, x))            

            scaler[x] = MinMaxScaler()
            scaler[x].fit(all_data)

        return scaler

    def cross_validation(self, num_splits):
        cross_val_accuracy_meter_top_all = []

        cross_val_p = []
        cross_val_r = []
        cross_val_f = []

        if args.mode == 'test':
            num_splits = 1
        data = get_babyrobot_data(phase = 'train')
        bodies, hands_right, hands_left, lengths, Y,  paths = data       

        self.kfold =KFold(n_splits = num_splits)

        # for n in trange(num_splits):
        for  n,(train_index, test_index) in enumerate(self.kfold.split(bodies)):
            self.current_split = n

            '''Dataset'''
            self.train_dataset = BodyFaceDataset(args=args, data=data, indices=train_index, subjects=list(range(0,31)), phase="train")
            self.test_dataset = BodyFaceDataset(args=args, data=data, indices=test_index, subjects=list(range(0,31)), phase="test")

            print("train samples: %d" % len(self.train_dataset))
            print(np.bincount(self.train_dataset.Y))

            print("test samples: %d" % len(self.test_dataset))
            print(np.bincount(self.test_dataset.Y))
          
            scaler = self.get_scaler()

            self.train_dataset.set_scaler(scaler)
            self.test_dataset.set_scaler(scaler)

            self.train_dataset.to_tensors()
            self.test_dataset.to_tensors()

            self.train_dataset.prepad()
            self.test_dataset.prepad()

            print("scaled data")

            if self.args.batch_size == -1:
                batch_size = len(self.train_dataset)
            else:
                batch_size = self.args.batch_size

            self.dataloader_train = torch.utils.data.DataLoader(self.train_dataset, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=4)
            self.dataloader_test = torch.utils.data.DataLoader(self.test_dataset, batch_size=len(self.test_dataset), num_workers=4, shuffle=True)


            '''Model'''
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = BodyFaceEmotionClassifier(self.args)
            # if torch.cuda.device_count() > 1:
            #     print("Let's use", torch.cuda.device_count(), "GPUs!")
            #     self.model = nn.DataParallel(self.model)
            self.model.to(self.device)
            if self.args.checkpoint:
                print("Loading pretrained weights...", self.args.checkpoint)
                checkpoint = torch.load(self.args.checkpoint)
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)


            '''Start Training & Testing'''
            start = time.time()
            val_top_all, p, r, f = self.fit(self.model, self.current_split)
            end = time.time()


            '''Append The ACC'''
            cross_val_accuracy_meter_top_all.append(val_top_all)
            cross_val_p.append(p)
            cross_val_r.append(r)
            cross_val_f.append(f)
            print(val_top_all, p, r, f)
            print('[Split: %02d/%02d] Accuracy: %.3f SK Prec: %.3f SK Rec: %.3f F-Score: %.3f Time: %.3f'
                % (n+1, num_splits, np.mean(cross_val_accuracy_meter_top_all), np.mean(cross_val_p), np.mean(cross_val_r), np.mean(cross_val_f), end-start))

            if self.args.wandb:
                wandb.log({'cross_val_acc_top_all': np.mean(cross_val_accuracy_meter_top_all),
                           'cross_val_p':np.mean(cross_val_p),
                           'cross_val_r':np.mean(cross_val_r),
                           'cross_val_f':np.mean(cross_val_f)})


        return np.mean(cross_val_accuracy_meter_top_all), np.mean(cross_val_p), np.mean(cross_val_r), np.mean(cross_val_f)

    def fit(self, model, current_split):

        self.criterion = nn.CrossEntropyLoss().to(self.device)

        if self.args.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, momentum=self.args.momentum)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=args.gamma)

        best_acc = 0

        self.mat = np.empty(0)

        for self.current_epoch in trange(0, self.args.epochs):
            if self.args.mode == 'train':
                train_acc, train_loss = self.train_epoch()
            else:
                train_acc, train_loss =0, 0


            val_top_all, val_loss, p, r, f = self.eval()

            for param_group in self.optimizer.param_groups:
                lr = param_group['lr']

            print('[Epoch: %3d/%3d] Training Loss: %.3f, Validation Loss: %.3f, Training Acc: %.3f, Validation Acc: %.3f, Learning Rate:%.8f'
                % (self.current_epoch, self.args.epochs, train_loss, val_loss, train_acc, val_top_all, lr))

            if val_top_all > 40 and val_top_all > best_acc and self.args.mode != 'test':
                torch.save({'iter': self.current_epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                            os.path.join('../models/GEMEP_BODY/split' + str(self.current_split) + '_ep' + str(self.current_epoch) +  "_acc" + str(round(val_top_all, 2)) + ".pth"))
                best_acc = val_top_all
                print('Model saved.')

        return val_top_all, p, r, f

    def train_epoch(self):
        self.model.train()

        accuracy_meter_top_all = AverageMeter()
        loss_meter = AverageMeter()

        for i, batch in enumerate(self.dataloader_train):
            body, hand_right, hand_left, length, y = batch['body'].to(self.device),\
                                                     batch['hand_right'].to(self.device),\
                                                     batch['hand_left'].to(self.device),\
                                                     batch['length'].to(self.device),\
                                                     batch['label'].to(self.device)

            self.optimizer.zero_grad()

            out = self.model.forward((body, hand_right, hand_left, length))

            loss = self.criterion(out, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

            self.optimizer.step()

            accs = accuracy(out, y, topk=(1,))
            accuracy_meter_top_all.update(accs[0], body.size(0))
            loss_meter.update(loss.item(), body.size(0))

        if args.wandb:
            wandb.log({"lr": self.optimizer.param_groups[0]['lr'],
                       "train_loss_avg": loss_meter.avg,
                       "train_acc_avg": accuracy_meter_top_all.avg})
        
        self.scheduler.step()

        return accuracy_meter_top_all.avg, loss_meter.avg

    def eval(self):
        accuracy_meter_top_all = AverageMeter()
        loss_meter = AverageMeter()
        self.gt_labels = []
        self.pre_labels_body = []
        self.correct_count_body = 0

        with torch.no_grad():
            self.model.eval()
            for i, batch in enumerate(self.dataloader_test):
                body, hand_right, hand_left, length, y = batch['body'].to(self.device), \
                                                         batch['hand_right'].to(self.device), \
                                                         batch['hand_left'].to(self.device), \
                                                         batch['length'].to(self.device), \
                                                         batch['label'].to(self.device)

                out = self.model.forward((body, hand_right, hand_left, length))
                accs = accuracy(out, y, topk=(1,))

                """ change average to the desired (macro for balanced) """
                p, r, f, s = precision_recall_fscore_support(y.cpu(), out.detach().cpu().argmax(dim=1), average="macro")

                accuracy_meter_top_all.update(accs[0].item(), length.size(0))

                loss = self.criterion(out, y)
                loss_meter.update(loss.item(), body.size(0))

                if self.args.plot_cm:
                    '''confusion matrix preprocess'''
                    self.gt_labels += y.cpu().tolist()
                    _, body_predicts = torch.max(out, 1)
                    self.pre_labels_body += body_predicts.cpu().tolist()

                    '''calculate the acc'''
                    correct_or_not_body = torch.eq(body_predicts, y)
                    self.correct_count_body += correct_or_not_body.sum().cpu()

        if (self.current_epoch == self.args.epochs or self.args.mode == 'test') and self.args.plot_cm:
            acc_body = self.correct_count_body.float() / float(self.test_dataset.__len__())
            acc_body = np.around(acc_body.numpy(), 4)
            plot_cm(self.gt_labels, self.pre_labels_body, acc_body)
            
        if args.wandb:
            wandb.log({'val_acc_top_all': accuracy_meter_top_all.avg,
                       'val_loss': loss_meter.avg,
                       'val_p':p*100,
                       'val_r':r*100,
                       'val_f':f*100})

        return accuracy_meter_top_all.avg, loss_meter.avg, p*100,r*100,f*100


def plot_cm(gt_labels, pre_labels_body, acc_body):
    cm_body = confusion_matrix(gt_labels, pre_labels_body)
    cm_body = np.array(cm_body)
    labels_name = ['SU', 'FE', 'DI', 'HA', 'SA', 'AN', "CO"]#the label of the confusion_matrix
    plot_confusion_matrix(cm_body, labels_name, args.dataset, acc_body)
    print('Plotting is Done.')


def parse_opts():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--wandb', action='store_true')
    # ========================= Optimizer Parameters ==========================
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--step_size', default=50, type=int)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--optimizer', type=str, default="Adam")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--gamma', type=float, default=0.8)

    # ========================= Usual Hyper Parameters ==========================
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--dataset', type=str, default='rafdb')
    parser.add_argument('--db', default="babyrobot")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--grad_clip', type=float, default=0.1)

    # ========================= Network Parameters ==========================
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    parser.add_argument('--mode',default='train',help='choose the mode train or test')
    parser.add_argument('--confidence_threshold', type=float, default=0.1)
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--num_total_iterations', type=int, default=1)
    parser.add_argument('--num_splits', type=int, default=5)
    parser.add_argument('--first_layer_size', default=256, type=int)
    parser.add_argument('--plot_cm', action="store_true", help="plot the confusion matrix")

    # ========================= Training Parameters ==========================
    parser.add_argument('--face_pooling', default="max", help="how to aggregate the face features sequence")
    parser.add_argument('--body_pooling', default="avg", help="how to aggregate the body features sequence")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_opts()
    b = EmotionRecognitionSystem(args)
    b.run()