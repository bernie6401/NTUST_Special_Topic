from image_utils import mtcnn, plot_confusion_matrix,accuracy
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from dataset_leave_one_out import Dataset
import numpy as np
import argparse
import Networks
import warnings
warnings.filterwarnings("ignore")
import torch
from tqdm import trange

import pandas as pd
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=1)


'''Fusing Body'''
from sklearn.preprocessing import MinMaxScaler
from dataset_leave_one_out import BodyFaceDataset, get_babyrobot_data
from Networks import *
import os 

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
    parser.add_argument('--num_classes', type=int, default=12)
    parser.add_argument('--confidence_threshold', type=float, default=0.1)
    parser.add_argument('--body_pooling', default="avg", help="how to aggregate the body features sequence")
    parser.add_argument('--db', default="babyrobot")

    return parser.parse_args()


def test():
    pre_labels_face = []
    pre_labels_body = []
    pre_labels_whole = []
    gt_labels = []
    avg_face =[]
    avg_body =[]
    avg_whole =[]
    best_whole_acc = 0

    if args.num_classes == 12:
        path_face = '../models/GEMEP_FACE_12/'
        path_body = '../models/GEMEP_BODY_12/'
        path_fc = '../models/GEMEP_FC_12/'
    elif args.num_classes == 7:
        path_face = '../models/GEMEP_FACE_7/'
        path_body = '../models/GEMEP_BODY_7/'
        path_fc = '../models/GEMEP_FC_7/'
    
    for n in range(1, 11):

        #----------------------Face Model----------------------#
        face_checkpoints = [name for name in os.listdir(path_face)]
        for i,face_checkpoint in enumerate(face_checkpoints):
            if n == int(face_checkpoint.split('_')[0][5:]):
                path_face_checkpoint = os.path.join(path_face,face_checkpoint)
        print("Loading face pretrained weights...", path_face_checkpoint)
        face_model = Networks.ResNet18_ARM___RAF(num_classes=args.num_classes)
        checkpoint_face = torch.load(path_face_checkpoint)
        face_model.load_state_dict(checkpoint_face["model_state_dict"], strict=False)
        face_model = face_model.cuda()

        #----------------------Body Model----------------------#
        body_checkpoints = [name for name in os.listdir(path_body)]
        for i,body_checkpoint in enumerate(body_checkpoints):
            if n == int(body_checkpoint.split('_')[0][5:]):
                path_body_checkpoint = os.path.join(path_body,body_checkpoint)
        print("Loading body pretrained weights...", path_body_checkpoint)       
        body_model = Networks.BodyFaceEmotionClassifier(args).cuda()
        checkpoint_body = torch.load(path_body_checkpoint)
        body_model.load_state_dict(checkpoint_body["model_state_dict"], strict=False)
        body_model = body_model.cuda()

        #----------------------FC Layer----------------------#
        fc_checkpoints = [name for name in os.listdir(path_fc)]
        for i, fc_checkpoint in enumerate(fc_checkpoints):
            if n == int(fc_checkpoint.split('_')[0][5:]):
                path_fc_checkpoint = os.path.join(path_fc, fc_checkpoint)
        print("Loading body pretrained weights...", path_fc_checkpoint)
        fc_model = Networks.FcLayer(args).cuda()
        checkpoint_fc = torch.load(path_fc_checkpoint)
        fc_model.load_state_dict(checkpoint_fc["model_state_dict"], strict=False)
        fc_model = fc_model.cuda()

        #----------------------Face Data----------------------#
        # data_type()
        data_transforms_test = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.558, 0.437, 0.384], std=[0.277, 0.247, 0.241])])

        face_test_dataset = Dataset(args.dataset_path, args.data_type, phase='val', transform = data_transforms_test, data = None, performer_number = n, num_classes = args.num_classes)
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
        print('Body testing size: ', body_test_size, '\n')

        body_test_loader = torch.utils.data.DataLoader(body_test_dataset, 
                                                    batch_size=args.batch_size, 
                                                    num_workers=args.workers,
                                                    shuffle=False, 
                                                    pin_memory=True)

        '''
        if args.num_classes == 7:
            face_matrix = np.array([0.5, 0.5, 0.9, 0.8, 0.9, 0.1, 0.5])
            body_matrix = np.array([0.5, 0.5, 0.1, 0.2, 0.1, 0.9, 0.5])
            # face_matrix = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
            # body_matrix = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        if args.num_classes == 12:
            # face_matrix = np.ones(12) * test * 0.1
            # body_matrix = np.ones(12) * (10 - test) * 0.1
            ones_matrix = np.ones(12)
            # face_matrix = np.array([0.9, 0.4, 0.4, 0.1, 0.8, 0.5, 0.2, 0.9, 0.8, 0.3, 0.5, 0.5])
            # body_matrix = ones_matrix - face_matrix
            # face_matrix = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
            # body_matrix = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
            # face_matrix = np.log2([7, 6, 5, 5, 5, 3, 6, 7, 4, 4, 6, 4])/np.log2([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
            # body_matrix = np.log2([4, 7, 7, 9, 3, 4, 8, 4, 2, 6, 4, 5])/np.log2([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
            face_matrix = np.array([0.09022648, 0.08598018, 0.06935265, 0.0823464 , 0.07255445, 0.105259, 0.0862889, 0.1142189, 0.0860274, 0.04978191, 0.09207553, 0.0658882])
            body_matrix = np.array([0.06197982, 0.08406493, 0.14086772, 0.14503295, 0.06976949, 0.09839345, 0.09265199, 0.05517694, 0.04631127, 0.06194569, 0.06130058, 0.08250517])
            # face_matrix = np.array([0.07610315, 0.08502256, 0.10511018, 0.11368968, 0.07116197, 0.10182623, 0.08947045, 0.08469792, 0.06616934, 0.0558638,  0.07668805, 0.07419669])
            # body_matrix = ones_matrix - face_matrix
        '''

        '''Start Testing'''
        print('Start Testing...\n')
        if args.phase != 'new_test':

            feature_map_face =[]
            storage_face = np.zeros((body_test_size, args.num_classes))
            storage_body = np.zeros((body_test_size, args.num_classes))
            video_img_count = 0

            with torch.no_grad():
                correct_count_face, correct_count_body, correct_count_whole = 0, 0, 0
                face_model.eval()
                body_model.eval()
                fc_model.eval()
                bingo_temp = []
                
                '''Face Output'''
                for batch_i, (face_imgs, face_targets, _) in enumerate(face_test_loader):
                    face_outputs, _ = face_model(face_imgs.cuda())
                    face_targets = face_targets.cuda()
                    face_percentage = face_outputs
                    # face_percentage = torch.nn.functional.softmax(face_outputs, dim=1)

                    for i in range(len(face_percentage.detach().cpu().numpy())):
                        feature_map_face.append(face_percentage[i])#.detach().cpu().numpy()
                
                content, file_path, label = face_test_dataset.__content__()
                for i in range(len(content)):                                           #videos in total
                    for j in range(args.num_classes):                                   #7 emotions in total
                        for k in range(video_img_count, video_img_count + content[i]):  #all imgs in each video have to add
                            storage_face[i][j] += feature_map_face[k][j]
                        storage_face[i][j] /= content[i]
                    video_img_count += content[i]
                
                # storage_face_mx = storage_face * face_matrix


                '''Body Output'''
                video_img_count = 0
                all_label = torch.tensor([]).cuda()

                for batch_i, (batch) in enumerate(body_test_loader):
                    body, hand_right, hand_left, length, Y = batch['body'].cuda(), batch['hand_right'].cuda(), batch['hand_left'].cuda(), batch['length'].cuda(), batch['label'].cuda()
                    body_outputs = body_model.forward((body, hand_right, hand_left, length))
                    _, body_predicts = torch.max(body_outputs, 1)
                    # body_percentage = torch.nn.functional.softmax(body_outputs, dim=1)
                    body_percentage = body_outputs
                    
                    for i in range(len(body_percentage.detach().cpu().numpy())):
                        for j in range(args.num_classes):
                            storage_body[i + video_img_count][j] = body_percentage[i][j]#.detach().cpu().numpy()
                    video_img_count += len(body_percentage.detach().cpu().numpy())

                    all_label = torch.cat((all_label, Y), 0)
                    

                # storage_body_mx = storage_body * body_matrix
                

                '''FC Output'''
                concate_data = torch.cat((torch.from_numpy(storage_face), torch.from_numpy(storage_body)), 1).float().cuda()
                concate_output = fc_model(concate_data)
                _, whole_predicts = torch.max(concate_output, 1)
                correct_count_whole = torch.eq(whole_predicts, all_label.cuda())

                
                '''Concate'''
                # _, whole_predicts = torch.max(torch.from_numpy(storage_face_mx + storage_body_mx), 1)
                _, storage_face = torch.max(torch.from_numpy(storage_face), 1)
                _, storage_body = torch.max(torch.from_numpy(storage_body), 1)

                correct_count_face = torch.eq(storage_face, torch.Tensor(label))
                correct_count_body = torch.eq(storage_body, torch.Tensor(label))
                # correct_count_whole = torch.eq(whole_predicts, torch.Tensor(label))

                '''confusion matrix preprocess'''
                pre_labels_face += storage_face.cpu().tolist()
                pre_labels_body += storage_body.cpu().tolist()
                pre_labels_whole += whole_predicts.cpu().tolist()
                gt_labels += label

                # for i in range(args.num_classes):
                #     if correct_count_face[i] == True:
                #         bingo_temp.append(label[i])
                # print(bingo_temp)

                acc_face = correct_count_face.float().sum().cpu() / float(body_test_size)
                acc_face = np.around(acc_face.numpy(), 4)
                avg_face.append(acc_face)
                print(f"Face Test ACC: {acc_face:.4f}.")

                acc_body = correct_count_body.float().sum().cpu() / float(body_test_size)
                acc_body = np.around(acc_body.numpy(), 4)
                avg_body.append(acc_body)
                print(f"Body Test ACC: {acc_body:.4f}.")

                acc_whole = correct_count_whole.float().sum().cpu() / float(body_test_size)
                acc_whole = np.around(acc_whole.numpy(), 4)
                avg_whole.append(acc_whole)
                print(f"Whole Test ACC: {acc_whole:.4f}.")

        else:
            with torch.no_grad():
                model.eval()
                for batch_i, (imgs) in enumerate(face_test_loader):
                    outputs, _ = model(imgs.cuda())
                    _, predicts = torch.max(outputs, 1)
                    print(predicts)
            
    acc_face = np.around(sum(avg_face)/len(avg_face), 4)  
    acc_body = np.around(sum(avg_body)/len(avg_body), 4) 
    acc_whole = np.around(sum(avg_whole)/len(avg_whole), 4)

    print('All face acc: ', acc_face, '\nAll body acc: ', acc_body, '\nAll whole acc: ', acc_whole)


    if args.plot_cm:
        if args.phase == 'new_test':
            print('You can not use it function without non-lables datasets.')
        else:
            if args.num_classes == 7:
                labels_name = ['SU', 'FE', 'DI', 'PL', 'SA', 'AN', "CO"]#the label of the confusion_matrix
            elif args.num_classes == 12:
                labels_name = ['amu', 'pri', 'joy', 'fea', 'anx', 'des', "sad",'ang','irr','int','ple','rel']

            cm_face = confusion_matrix(gt_labels, pre_labels_face)
            cm_face = np.array(cm_face)
            plot_confusion_matrix(cm_face, labels_name, args.data_type + '_Face', acc_face, args.num_classes)

            cm_body = confusion_matrix(gt_labels, pre_labels_body)
            cm_body = np.array(cm_body)
            plot_confusion_matrix(cm_body, labels_name, args.data_type + '_Body', acc_body, args.num_classes)

            cm_whole = confusion_matrix(gt_labels, pre_labels_whole)
            cm_whole = np.array(cm_whole)
            plot_confusion_matrix(cm_whole, labels_name, args.data_type + '_Whole', acc_whole, args.num_classes)


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
    

def store_weight(epoch, model_state_dict, optimizer_state_dict, val_acc, fold, bingo_temp):
    torch.save({'iter': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict},
                 os.path.join('../models', args.data_type, 'split' + str(fold) + "_epoch" + str(epoch) + "_acc" + str(val_acc) + str(bingo_temp) + ".pth"))
    print('Model saved.')


if __name__ == "__main__":
    args = parse_args()                
    test()