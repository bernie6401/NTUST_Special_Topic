import warnings
warnings.filterwarnings("ignore")
#from apex import amp
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import os, torch
import argparse
from dataset_leave_one_out import *

from tqdm import trange
import wandb
from sklearn.preprocessing import MinMaxScaler
from Networks import * 
import Networks


def parse_args():
    parser = argparse.ArgumentParser()
    '''ARM'''
    parser.add_argument('--dataset_path', type=str, default='../datasets/', help='Root dataset path.')
    parser.add_argument('--data_type', type=str, default='GEMEP', help='rafdb or affectnet or ferplus or new_test(the new data you want to test)')
    parser.add_argument('--new_test_path', type=str, default=None, help='New test dataset path.')
    parser.add_argument('--phase', type=str, default='test', help='The phase you want to choose.(default:test or new_test)')
    parser.add_argument('--checkpoint_face', type=str, default='../models/GEMEP_FACE/split1_epoch1_acc0.9139.pth', help='Pytorch checkpoint file path')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--workers', default=1, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('-p', '--plot_cm', action='store_true', help='Ploting confusion matrix.')
    parser.add_argument('--mtcnn', action='store_true', help='Using MTCNN to align image.')
    parser.add_argument('--mt_img_path', type=str, default=None, help='Path of the dataset you want to align.')
    parser.add_argument('--mt_save_path', type=str, default=None, help='Path you want to save your dataset.')
    parser.add_argument('--video', action="store_true", help="Using MTCNN to align image.")
    parser.add_argument('--video_path', type=str, default='../datasets/video/', help='Video data path.')
    
    '''Fusing Body'''
    parser.add_argument('--checkpoint_body', type=str, default='../models/GEMEP_BODY/split4_ep59_acc77.27.pth', help='Pytorch checkpoint file path')
    parser.add_argument('--first_layer_size', default=256, type=int)
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--confidence_threshold', type=float, default=0.1)
    parser.add_argument('--body_pooling', default="avg", help="how to aggregate the body features sequence")
    parser.add_argument('--db', default="babyrobot")

    parser.add_argument('--epochs', type=int, default=200, help='Total training epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for sgd.')
    parser.add_argument('--mode', type=str, default="train")

    return parser.parse_args()


def test():
    path_face = '../models/GEMEP_FACE_7/'
    path_body = '../models/GEMEP_BODY_7/'
    path_fc = '../models/GEMEP_FC_7/'
    
    for n in range(1, 11):
        n = 2
        #----------------------Face Model----------------------#
        face_checkpoints = [name for name in os.listdir(path_face)]
        for i, face_checkpoint in enumerate(face_checkpoints):
            if n == int(face_checkpoint.split('_')[0][5:]):
                path_face_checkpoint = os.path.join(path_face,face_checkpoint)
        print("Loading face pretrained weights...", path_face_checkpoint)
        face_model = Networks.ResNet18_ARM___RAF(num_classes=args.num_classes)
        checkpoint_face = torch.load(path_face_checkpoint)
        face_model.load_state_dict(checkpoint_face["model_state_dict"], strict=False)
        face_model = face_model.cuda()

        #----------------------Body Model----------------------#
        body_checkpoints = [name for name in os.listdir(path_body)]
        for i, body_checkpoint in enumerate(body_checkpoints):
            if n == int(body_checkpoint.split('_')[0][5:]):
                path_body_checkpoint = os.path.join(path_body,body_checkpoint)
        print("Loading body pretrained weights...", path_body_checkpoint)       
        body_model = Networks.BodyFaceEmotionClassifier(args).cuda()
        checkpoint_body = torch.load(path_body_checkpoint)
        body_model.load_state_dict(checkpoint_body["model_state_dict"], strict=False)
        body_model = body_model.cuda()

        #----------------------FC Layer----------------------#
        if args.mode == 'train':
            fc_model = Networks.FcLayer(args).cuda()
            fc_model = fc_model.cuda()

        # elif args.mode == 'test':
            fc_checkpoints = [name for name in os.listdir(path_fc)]
            for i, fc_checkpoint in enumerate(fc_checkpoints):
                if n == int(fc_checkpoint.split('_')[0][5:]):
                    path_fc_checkpoint = os.path.join(path_fc, fc_checkpoint)
            print("Loading body pretrained weights...", path_fc_checkpoint)
            fc_model = Networks.FcLayer(args).cuda()
            checkpoint_fc = torch.load(path_fc_checkpoint)
            fc_model.load_state_dict(checkpoint_fc["model_state_dict"], strict=False)
            fc_model = fc_model.cuda()

        CE_criterion = torch.nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adam(params = fc_model.parameters(), weight_decay=1e-2, lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)

        #----------------------Face Data----------------------#
        # data_type()
        data_transforms = transforms.Compose([transforms.ToPILImage(),
                                              transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.558, 0.437, 0.384], std=[0.277, 0.247, 0.241]),
                                              transforms.RandomErasing(scale=(0.02, 0.1))])
        face_train_dataset = Dataset(args.dataset_path, args.data_type, phase='train', transform=data_transforms, basic_aug=True, data = None, performer_number = n)
        face_train_size = face_train_dataset.__len__()
        print('Face testing size: ', face_train_size, '\n')

        face_train_loader = torch.utils.data.DataLoader(face_train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)


        data_transforms_test = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.558, 0.437, 0.384], std=[0.277, 0.247, 0.241])])

        face_test_dataset = Dataset(args.dataset_path, args.data_type, phase='val', transform=data_transforms_test, data = None, performer_number=n)
        face_test_size = face_test_dataset.__len__()
        print('Face testing size: ', face_test_size, '\n')

        face_test_loader = torch.utils.data.DataLoader(face_test_dataset,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.workers,
                                                    shuffle=False,
                                                    pin_memory=True)

        #----------------------Body Data----------------------#
        body_test_dataset = BodyFaceDataset(args=args,  subjects=list(range(0,31)), phase="test", number = n)
        scaler = get_scaler(body_test_dataset)#find the min and max value that will be used later to scale
        body_test_dataset.set_scaler(scaler)#actually scaled the whole data
        body_test_dataset.to_tensors()
        body_test_dataset.prepad()
        body_test_size = body_test_dataset.__len__()
        body_test_loader = torch.utils.data.DataLoader(body_test_dataset, 
                                                        batch_size=body_test_size, 
                                                        num_workers=args.workers,
                                                        shuffle=False, 
                                                        pin_memory=True)

        body_train_dataset = BodyFaceDataset(args=args,  subjects=list(range(0,31)), phase="train", number = n)
        body_train_dataset.set_scaler(scaler)#actually scaled the whole data
        body_train_dataset.to_tensors()
        body_train_dataset.prepad()
        body_train_size = body_train_dataset.__len__()
        body_train_loader = torch.utils.data.DataLoader(body_train_dataset, 
                                                        shuffle=False, 
                                                        batch_size=body_train_size, 
                                                        drop_last=True, 
                                                        num_workers=args.workers)
        print('Body training size: ', body_train_size, '\n')
        print('Body testing size: ', body_test_size, '\n')

        '''Start Testing'''
        print('Start Testing...\n')
        

        train(fc_model, CE_criterion, optimizer, body_train_size, face_model, body_model, face_train_loader, face_train_dataset, body_train_loader, body_test_size, face_test_loader, face_test_dataset, body_test_loader, n, scheduler)



def train(fc_model, CE_criterion, optimizer, body_train_size, face_model, body_model, face_train_loader, face_train_dataset, body_train_loader, body_test_size, face_test_loader, face_test_dataset, body_test_loader, n, scheduler):
    if args.mode == 'train':
        fc_model.train()
        storage_face, storage_body, all_label = concate(body_train_size, face_model, body_model, face_train_loader, face_train_dataset, body_train_loader)
        concate_data = torch.cat((torch.from_numpy(storage_face), torch.from_numpy(storage_body)), 1).float().cuda()
    concate_data_val, all_label_val = 0, 0

    for current_epoch in trange(1, args.epochs + 1):    
        if args.mode == 'train':
            concate_output = fc_model(concate_data)
            
            loss = CE_criterion(concate_output, all_label.long())
            loss.backward()
            optimizer.step()
            
            _, predicts = torch.max(concate_output, 1)
            correct_count_fc = torch.eq(predicts, all_label.cuda())
            correct_num = correct_count_fc.sum()
            train_acc = correct_num.float()/float(len(all_label))
            print('Train loss = ', loss.item())
            print('Train ACC = ', train_acc.item())


        val_acc, concate_data_val, all_label_val = validate(fc_model, CE_criterion, body_test_size, face_model, body_model, face_test_loader, face_test_dataset, body_test_loader, current_epoch, n, concate_data_val, all_label_val)
        print('[Split: %02d/%02d]\n\n'%(n, 10))
        scheduler.step()


def validate(fc_model, CE_criterion, body_test_size, face_model, body_model, face_test_loader, face_test_dataset, body_test_loader, current_epoch, performer_number, concate_data, all_label):
    fc_model.eval()
    if current_epoch == 1:
        storage_face, storage_body, all_label = concate(body_test_size, face_model, body_model, face_test_loader, face_test_dataset, body_test_loader)
        concate_data = torch.cat((torch.from_numpy(storage_face), torch.from_numpy(storage_body)), 1).float().cuda()
    

    with torch.no_grad():
        
        concate_output = fc_model(concate_data)
        loss = CE_criterion(concate_output, all_label.long())
        _, predicts = torch.max(concate_output, 1)
        correct_count_fc = torch.eq(predicts, all_label)
        print('Val loss = ', loss.item())
        correct_num = correct_count_fc.sum()
        val_acc = correct_num.float()/float(len(all_label))
        val_acc = np.around(val_acc.cpu().numpy(), 4)
        print('[Val acc {}/{}] = {}'.format(correct_num.float(), len(all_label), val_acc))

        if val_acc > 0.8 and args.mode != 'test':
            store_weight(current_epoch, fc_model.state_dict(), val_acc, performer_number)
    
    return val_acc, concate_data, all_label


def concate(body_size, face_model, body_model, face_loader, face_dataset, body_loader):
    feature_map_face =[]
    storage_face = np.zeros((body_size, args.num_classes))
    storage_body = np.zeros((body_size, args.num_classes))
    video_img_count = 0

    with torch.no_grad():
        correct_count_face, correct_count_body, correct_count_whole = 0, 0, 0
        face_model.eval()
        body_model.eval()
        bingo_temp = []
        
        '''Face Output'''
        for batch_i, (face_imgs, face_targets, _) in enumerate(face_loader):
            face_outputs, _ = face_model(face_imgs.cuda())
            face_targets = face_targets.cuda()
            face_percentage = face_outputs
            # face_percentage = torch.nn.functional.softmax(face_outputs, dim=1)

            for i in range(len(face_percentage.detach().cpu().numpy())):
                feature_map_face.append(face_percentage[i])#.detach().cpu().numpy()
        
        content, file_path, label = face_dataset.__content__()
        for i in range(len(content)):                                           #videos in total
            for j in range(args.num_classes):                                   #7 emotions in total
                for k in range(video_img_count, video_img_count + content[i]):  #all imgs in each video have to add
                    storage_face[i][j] += feature_map_face[k][j]
                storage_face[i][j] /= content[i]
            video_img_count += content[i]
        

        '''Body Output'''
        video_img_count = 0
        all_label = torch.tensor([]).cuda()

        for batch_i, (batch) in enumerate(body_loader):
            body, hand_right, hand_left, length, Y = batch['body'].cuda(), batch['hand_right'].cuda(), batch['hand_left'].cuda(), batch['length'].cuda(), batch['label'].cuda()
            body_outputs = body_model.forward((body, hand_right, hand_left, length))
            _, body_predicts = torch.max(body_outputs, 1)
            # body_percentage = torch.nn.functional.softmax(body_outputs, dim=1)
            # print('Y =',Y)
            all_label = torch.cat((all_label, Y), 0)
            body_percentage = body_outputs
            
            for i in range(len(body_percentage.detach().cpu().numpy())):
                for j in range(args.num_classes):
                    storage_body[i + video_img_count][j] = body_percentage[i][j]#.detach().cpu().numpy()
            video_img_count += len(body_percentage.detach().cpu().numpy())

            
        print('len_face = ',len(storage_face))
        print('len_body = ',len(storage_body))
        print('length_concate = ',len(all_label))
        

    return storage_face, storage_body, all_label
            

def data_type():
    try:
        if (args.data_type == 'rafdb' or args.data_type == 'affectnet' or args.data_type == 'ferplus' or args.data_type == 'GEMEP_FACE') and args.mtcnn == False:
            args.dataset_path = args.dataset_path + args.data_type + '/'
            print('The dataset is {} and the path is {}'.format(args.data_type, args.dataset_path))
        elif args.mtcnn or args.data_type == 'new_test':
            if args.mtcnn and (args.mt_img_path == None or args.mt_save_path == None):
                raise Exception('Please enter the img path or the save path.')
            elif args.data_type == 'new_test' and args.new_test_path == None:
                raise Exception('Please enter the new test img path')
            else:
                args.dataset_path = mtcnn(args.mt_img_path, args.mt_save_path, args.new_test_path, args.mtcnn)
                args.data_type = args.phase = 'new_test'#這是一個保險，當使用者想要用mtcnn的時候忘記輸入phase和data_type
                print('Now the dataset path is {}'.format(args.mt_save_path if args.mtcnn else args.new_test_path))
        else:
            raise Exception('This dataset is not available in the model.')
    except Exception as e:
        print('You need to check your input parameter: ' + str(e))
        quit()
        
    
def get_scaler(test_dataset):
        scaler = {}
        feats = ["bodies", "hands_right", "hands_left", ]

        for x in feats:
            all_data = np.vstack(getattr(test_dataset, x))            

            scaler[x] = MinMaxScaler()
            scaler[x].fit(all_data)

        return scaler
    

def store_weight(epoch, model_state_dict, val_acc, fold):
    torch.save({'iter': epoch,
                'model_state_dict': model_state_dict,},
                 os.path.join('../models', args.data_type, 'split' + str(fold) + "_epoch" + str(epoch) + "_acc" + str(val_acc) + ".pth"))
    print('Model saved.')


if __name__ == "__main__":
    args = parse_args()                
    test()