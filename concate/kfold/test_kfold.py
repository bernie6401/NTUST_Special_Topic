from image_utils import mtcnn, plot_confusion_matrix,accuracy
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from dataset import Dataset
import numpy as np
import argparse
import Networks
import warnings
warnings.filterwarnings("ignore")
import torch


'''Fusing Body'''
from sklearn.preprocessing import MinMaxScaler
from dataset import BodyFaceDataset, get_babyrobot_data
from Networks import *


def parse_args():
    parser = argparse.ArgumentParser()
    '''ARM'''
    parser.add_argument('--dataset_path', type=str, default='../datasets/', help='Root dataset path.')
    parser.add_argument('--data_type', type=str, default='GEMEP_FACE', help='rafdb or affectnet or ferplus or new_test(the new data you want to test)')
    parser.add_argument('--new_test_path', type=str, default=None, help='New test dataset path.')
    parser.add_argument('--phase', type=str, default='test', help='The phase you want to choose.(default:test or new_test)')
    parser.add_argument('--checkpoint_face', type=str, default='../models/GEMEP_FACE/NEW_split0_epoch46_acc0.774.pth', help='Pytorch checkpoint file path')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--workers', default=1, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('-p', '--plot_cm', action='store_true', help='Ploting confusion matrix.')
    parser.add_argument('--mtcnn', action='store_true', help='Using MTCNN to align image.')
    parser.add_argument('--mt_img_path', type=str, default=None, help='Path of the dataset you want to align.')
    parser.add_argument('--mt_save_path', type=str, default=None, help='Path you want to save your dataset.')
    parser.add_argument('--video', action="store_true", help="Using MTCNN to align image.")
    parser.add_argument('--video_path', type=str, default='../datasets/video/', help='Video data path.')
    
    '''Fusing Body'''
    parser.add_argument('--checkpoint_body', type=str, default='../models/GEMEP_BODY/NEW_split0_ep31_acc75.0.pth', help='Pytorch checkpoint file path')
    parser.add_argument('--first_layer_size', default=256, type=int)
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--confidence_threshold', type=float, default=0.1)
    parser.add_argument('--body_pooling', default="avg", help="how to aggregate the body features sequence")
    parser.add_argument('--db', default="babyrobot")

    return parser.parse_args()


def test():
    
    #----------------------Face Model----------------------#
    print("Loading face pretrained weights...", args.checkpoint_face)
    face_model = Networks.ResNet18_ARM___RAF()
    checkpoint_face = torch.load(args.checkpoint_face)
    face_model.load_state_dict(checkpoint_face["model_state_dict"], strict=False)
    face_model = face_model.cuda()

    #----------------------Body Model----------------------#
    print("Loading body pretrained weights...", args.checkpoint_body)
    body_model = Networks.BodyFaceEmotionClassifier(args).cuda()
    checkpoint_body = torch.load(args.checkpoint_body)
    body_model.load_state_dict(checkpoint_body["model_state_dict"], strict=False)
    body_model = body_model.cuda()
	

    #----------------------Face Data----------------------#
    # data_type()
    data_transforms_test = transforms.Compose([transforms.ToPILImage(),
                                               transforms.Resize((224, 224)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.558, 0.437, 0.384], std=[0.277, 0.247, 0.241])])

    face_test_dataset = Dataset(args.dataset_path, args.data_type, phase=args.phase, transform=data_transforms_test, data=None, indices=None)
    face_test_size = face_test_dataset.__len__()
    print('Face testing size: ', face_test_size, '\n')

    face_test_loader = torch.utils.data.DataLoader(face_test_dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.workers,
                                                   shuffle=False,
                                                   pin_memory=True)

    #----------------------Body Data----------------------#
    body_test_dataset = BodyFaceDataset(args=args, subjects=list(range(0,31)))
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

    face_matrix = torch.tensor([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]).cuda()
    body_matrix = torch.tensor([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]).cuda()


    '''Start Testing'''
    print('Start Testing...\n')
    if args.phase != 'new_test':
        pre_labels_face = []
        pre_labels_body = []
        pre_labels_whole = []
        gt_labels = []

        feature_map_face =[]
        storage_face = np.zeros((body_test_size,7))
        storage_body = np.zeros((body_test_size,7))
        video_img_count = 0

        with torch.no_grad():
            correct_count_face, correct_count_body, correct_count_whole = 0, 0, 0
            face_model.eval()
            body_model.eval()
            
            '''Face Output'''
            for batch_i, (face_imgs, face_targets, _) in enumerate(face_test_loader):
                face_outputs, _ = face_model(face_imgs.cuda())
                face_targets = face_targets.cuda()
                _, face_predicts = torch.max(face_outputs, 1)
                face_percentage = torch.nn.functional.softmax(face_outputs, dim=1)
                # correct_or_not_face = torch.eq(face_predicts, face_targets)
                # pre_labels_face += face_predicts.cpu().tolist()
                # gt_labels += face_targets.cpu().tolist()
                # correct_count_face += correct_or_not_face.sum().cpu()


                face_per_mx = face_percentage * face_matrix
                for i in range(len(face_per_mx.detach().cpu().numpy())):
                    feature_map_face.append(face_per_mx.detach().cpu().numpy()[i])
            
            content, file_path, label = face_test_dataset.__content__()
            for i in range(len(content)):                                           #110 videos in total
                for j in range(7):                                                  #7 emotions in total
                    for k in range(video_img_count, video_img_count + content[i]):  #all imgs in each video have to add
                        storage_face[i][j] += feature_map_face[k][j]
                    storage_face[i][j] /= content[i]
                video_img_count += content[i]

            '''Body Output'''
            video_img_count = 0
            for batch_i, (batch) in enumerate(body_test_loader):
                body, hand_right, hand_left, length, Y = batch['body'].cuda(), batch['hand_right'].cuda(), batch['hand_left'].cuda(), batch['length'].cuda(), batch['label'].cuda()
                body_outputs = body_model.forward((body, hand_right, hand_left, length))
                _, body_predicts = torch.max(body_outputs, 1)
                body_percentage = torch.nn.functional.softmax(body_outputs, dim=1)
                body_per_mx = body_percentage * body_matrix
                for i in range(len(body_per_mx.detach().cpu().numpy())):
                    for j in range(7):
                        storage_body[i + video_img_count][j] = body_per_mx.detach().cpu().numpy()[i][j]
                video_img_count += len(body_per_mx.detach().cpu().numpy())
               
                # accs = accuracy(body_predicts, Y, topk=(1,))
                # print('accs = ',accs)

            # print('storage_face',storage_face)
            # print('length ',len(storage_face))
            # print('storage_body',storage_body)
            # print('length',len(storage_body))


            '''Concate'''
            _, whole_predicts = torch.max(torch.from_numpy(storage_face + storage_body), 1)
            _, storage_face = torch.max(torch.from_numpy(storage_face), 1)
            _, storage_body = torch.max(torch.from_numpy(storage_body), 1)

            correct_count_face = torch.eq(storage_face, torch.Tensor(label))
            correct_count_body = torch.eq(storage_body, torch.Tensor(label))
            correct_count_whole = torch.eq(whole_predicts, torch.Tensor(label))

            '''confusion matrix preprocess'''
            # pre_labels_face += face_predicts.cpu().tolist()
            # pre_labels_body += body_predicts.cpu().tolist()
            pre_labels_face = storage_face.cpu().tolist()
            pre_labels_body = storage_body.cpu().tolist()
            pre_labels_whole = whole_predicts.cpu().tolist()
            gt_labels = label

            # '''calculate the acc'''
            # correct_count_face += correct_or_not_face.sum().cpu()
            # correct_count_body += correct_or_not_body.sum().cpu()
            # correct_count_whole += correct_or_not_whole.sum().cpu()

            # print('face GT: ', label, '\n', 'body GT: ', label)
    
            acc_face = correct_count_face.float().sum().cpu() / float(body_test_size)
            acc_face = np.around(acc_face.numpy(), 4)
            print(f"Face Test ACC: {acc_face:.4f}.")

            acc_body = correct_count_body.float().sum().cpu() / float(body_test_size)
            acc_body = np.around(acc_body.numpy(), 4)
            print(f"Body Test ACC: {acc_body:.4f}.")

            acc_whole = correct_count_whole.float().sum().cpu() / float(body_test_size)
            acc_whole = np.around(acc_whole.numpy(), 4)
            print(f"Whole Test ACC: {acc_whole:.4f}.")

    else:
        with torch.no_grad():
            model.eval()
            for batch_i, (imgs) in enumerate(face_test_loader):
                outputs, _ = model(imgs.cuda())
                _, predicts = torch.max(outputs, 1)
                print(predicts)

    if args.plot_cm:
        if args.phase == 'new_test':
            print('You can not use it function without non-lables datasets.')
        else:
            cm_face = confusion_matrix(gt_labels, pre_labels_face)
            cm_face = np.array(cm_face)
            labels_name = ['SU', 'FE', 'DI', 'HA', 'SA', 'AN', "NE"]#the label of the confusion_matrix
            plot_confusion_matrix(cm_face, labels_name, args.data_type, acc_face)

            cm_body = confusion_matrix(gt_labels, pre_labels_body)
            cm_body = np.array(cm_body)
            labels_name = ['SU', 'FE', 'DI', 'HA', 'SA', 'AN', "NE"]#the label of the confusion_matrix
            plot_confusion_matrix(cm_body, labels_name, args.data_type, acc_body)

            cm_whole = confusion_matrix(gt_labels, pre_labels_whole)
            cm_whole = np.array(cm_whole)
            labels_name = ['SU', 'FE', 'DI', 'HA', 'SA', 'AN', "NE"]#the label of the confusion_matrix
            plot_confusion_matrix(cm_whole, labels_name, args.data_type, acc_whole)


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
    

if __name__ == "__main__":
    args = parse_args()                
    test()