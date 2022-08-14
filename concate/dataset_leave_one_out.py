from pandas.io import parsers
import torch.utils.data as data
from image_utils import *
import image_utils
import pandas as pd
import numpy as np
import random
import torch
import json
import cv2
import os
from torchvision import transforms

'''General'''
def emo_tranform(n):
    
    if n == 7:
        dic_emotion_transform = {
            'surprise':0,
            'fear':1,
            'disgust':2,
            'pleasure':3,
            'sadness':4,
            'anger':5,
            'contempt':6
        }

    elif n == 12:
        dic_emotion_transform = {
            'amusement':0,
            'pride':1,
            'joy':2,
            'fear':3,
            'anxiety':4,
            'despair':5,
            'sadness':6,
            'anger':7,
            'irritation':8,
            'interest':9,
            'pleasure':10,
            'relief':11
        }
    
    return dic_emotion_transform


'''ARM'''
class Dataset(data.Dataset):
    def __init__(self, data_path, data_type, phase, data, performer_number, num_classes, transform=None, basic_aug=False, ):
        self.phase = phase
        self.transform = transform
        self.data_path = data_path
        self.data_type = data_type
        self.dic_emotion_transform = emo_tranform(num_classes)

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        
        if data_type == 'affectnet':
            
            if phase == 'train':
                df = pd.read_csv(os.path.join(self.data_path, 'EmoLabel/train_lab7.txt'), sep=' ', header=None)
                dataset = df[df[NAME_COLUMN].str.endswith('.jpg')]
                
            elif phase == 'val' or phase == 'test':
                df = pd.read_csv(os.path.join(self.data_path, 'EmoLabel/val_lab7.txt'), sep=' ', header=None)
                dataset = df[df[NAME_COLUMN].str.endswith('.jpg')]
            
            file_names = dataset.iloc[:, NAME_COLUMN].values
            self.label = dataset.iloc[:, LABEL_COLUMN].values
            # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
            self.file_paths = []
    
            for f in file_names:
                if phase == 'train':
                    path = os.path.join(self.data_path, 'Image/train_set/images/', f)
                else:
                    path = os.path.join(self.data_path, 'Image/val_set/images/', f)
                self.file_paths.append(path)
            

        elif data_type == 'rafdb':
            
            if phase == 'train':
                df = pd.read_csv(os.path.join(self.data_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
                dataset = df[df[NAME_COLUMN].str.startswith('train')]
            elif phase == 'val' or phase == 'test':
                df = pd.read_csv(os.path.join(self.data_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
                dataset = df[df[NAME_COLUMN].str.startswith('test')]
            
            file_names = dataset.iloc[:, NAME_COLUMN].values
            self.label = dataset.iloc[:, LABEL_COLUMN].values - 1
            # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
    
            self.file_paths = []
            for f in file_names:
                f = f.split(".")[0]
                f = f + "_aligned.jpg"
                path = os.path.join(self.data_path, 'Image/aligned', f)
                self.file_paths.append(path)


        elif data_type == 'GEMEP':
            data = []
            self.label = []
            self.file_paths = []
            self.content = []
            self.emotion_label = []
            with open("../datasets/GEMEP/all_face_label.csv") as data_file:#face_test
                for x in data_file.readlines()[1:]:
                    v = {
                        "path": x.split(",")[0].split(".")[0],
                        "performer_num": int(x.split(",")[0].split("/")[2][0:2]),
                        "emotion":x.split(",")[1].strip() # map emotion to number
                    }
                    data.append(v)

            if phase == 'train':
                for video in data:
                    if video['emotion'] in self.dic_emotion_transform.keys() and video['performer_num'] != performer_number:
                        self.get_data_what_u_want(video, phase)
            elif phase == 'val':
                for video in data:
                    if video['emotion'] in self.dic_emotion_transform.keys() and video['performer_num'] == performer_number:
                        self.get_data_what_u_want(video, phase)


        elif data_type == 'new_test':
            self.file_paths = data_path
        
        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image, image_utils.add_gaussian_noise]

    def get_data_what_u_want(self, video, phase):
        join_path = os.path.join('../datasets/' + video['path'] + "/")
        img_paths = [name for name in os.listdir(join_path)]     
        for img_path in img_paths:
            join_img_path = os.path.join(join_path,img_path)
            self.file_paths.append(join_img_path)
            self.label.append(self.dic_emotion_transform[video['emotion']]) #U must put it here and be sure that every file has label
        self.content.append(len(img_paths))
        self.emotion_label.append(self.dic_emotion_transform[video['emotion']])
        if phase == 'val':
            print(phase, ': ', join_path)

    def __content__(self):
        # return (self.content, self.label)
        return self.content, self.file_paths, self.emotion_label

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]# BGR to RGB
        if self.phase != 'new_test':
            label = self.label[idx]
            # label = self.label
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)
            
        # if self.data_type =='GEMEP_FACE' :
        #     return image, label,self.content ,idx   
        if self.phase != 'new_test':
            return image, label ,idx
        else:
            return image


# 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
'''Fusing Body'''
'''babyrobot_mapper = {
    "Happiness": 3,
    "Sadness": 4,
    "Surprise": 0,
    "Fear": 1,
    "Disgust": 2,
    "Anger": 5,
    "Neutral": 6,
}

inv_babyrobot_mapper = {
    5: "Anger",
    2: "Disgust",
    1: "Fear",
    3: "Happiness",
    4: "Sadness",
    0: "Surprise",
    6: "Neutral"
}'''


def get_babyrobot_annotations(phase):
    """ load the annotations of the babyrobot dataset """
    data = []
    with open("../datasets/GEMEP/all_body_label.csv") as data_file:
    # with open("rafdb_openpose/train_labels.csv" if phase == 'train' else 'rafdb_openpose/test_labels.csv') as data_file:
        for x in data_file.readlines()[1:]:
            v = {
                "path": x.split(",")[0].split(".")[0],
                "group_num": int(x.split(",")[0].split("/")[2][0:2]),
                "emotion": x.split(",")[1].strip(),
            }
  
            data.append(v)
        
    return data


def get_babyrobot_data(phase, number, num_classes, subjects=list(range(0,31))):
    
    data = get_babyrobot_annotations(phase)

    bodies, lengths, hands_right, hands_left, Y = [], [], [], [], []
    paths = []

    for video in data:
        if video['emotion'] in emo_tranform(num_classes).keys() :
            if phase =='train' and video['group_num'] != number:

                label = emo_tranform(num_classes)[video['emotion']]
                # ========================= Load OpenPose Features ==========================
                json_dir = os.path.join('../datasets/' + video['path'] + "/")
                # print('train_direction = ',json_dir)
                if not os.path.exists(json_dir):
                    print(json_dir)
                    raise

                json_list = sorted(os.listdir(json_dir))

                keypoints_array, hand_left_keypoints_array, hand_right_keypoints_array = get_keypoints_from_json_list(
                    json_list, json_dir, video['emotion'], visualize=False)

                keypoints_array = np.stack(keypoints_array).astype(np.float32)
                hand_right_keypoints_array = np.stack(hand_right_keypoints_array).astype(np.float32)
                hand_left_keypoints_array = np.stack(hand_left_keypoints_array).astype(np.float32)

                hands_right.append(hand_right_keypoints_array)
                hands_left.append(hand_left_keypoints_array)
                bodies.append(keypoints_array)
                lengths.append(keypoints_array.shape[0])
                Y.append(label)
                paths.append(video['path'])

        if video['emotion'] in emo_tranform(num_classes).keys() :
            if phase =='test' and video['group_num'] == number:
                label = emo_tranform(num_classes)[video['emotion']]

                # ========================= Load OpenPose Features ==========================

                json_dir = os.path.join('../datasets/' + video['path'] + "/")
                print('test_direction = ',json_dir)
                if not os.path.exists(json_dir):
                    print(json_dir)
                    raise

                json_list = sorted(os.listdir(json_dir))

                keypoints_array, hand_left_keypoints_array, hand_right_keypoints_array = get_keypoints_from_json_list(
                    json_list, json_dir, video['emotion'], visualize=False)

                keypoints_array = np.stack(keypoints_array).astype(np.float32)
                hand_right_keypoints_array = np.stack(hand_right_keypoints_array).astype(np.float32)
                hand_left_keypoints_array = np.stack(hand_left_keypoints_array).astype(np.float32)

                hands_right.append(hand_right_keypoints_array)
                hands_left.append(hand_left_keypoints_array)
                bodies.append(keypoints_array)
                lengths.append(keypoints_array.shape[0])
                Y.append(label)
                paths.append(video['path'])
    
    return bodies, hands_right, hands_left, lengths, Y, paths


def get_keypoints_from_json_list(json_list, json_dir, subject=None,emotion=None, visualize=False):
    global k1,k2
    keypoints_array, hand_left_keypoints_array, hand_right_keypoints_array = [], [], []

    visualization_counter = 1

    for json_file in json_list:
        if not json_file.endswith(".json"):
            raise
        js = os.path.join(json_dir, json_file)

        with open(js) as f:
            json_data = json.load(f)

        # ========================= Load OpenPose Features ==========================

        if len(json_data['people']) == 0:
            keypoints = np.zeros(75, dtype=np.float32)
            hand_left_keypoints = np.zeros(63, dtype=np.float32)
            hand_right_keypoints = np.zeros(63, dtype=np.float32)
        else:
            keypoints = np.asarray(json_data['people'][0]['pose_keypoints_2d'], dtype=np.float32)
            hand_left_keypoints = np.asarray(json_data['people'][0]['hand_left_keypoints_2d'], dtype=np.float32)
            hand_right_keypoints = np.asarray(json_data['people'][0]['hand_right_keypoints_2d'], dtype=np.float32)

        keypoints = np.reshape(keypoints, (-1, 3))  # reshape to num_points x dimension
        hand_left_keypoints = np.reshape(hand_left_keypoints, (-1, 3))  # reshape to num_points x dimension
        hand_right_keypoints = np.reshape(hand_right_keypoints, (-1, 3))  # reshape to num_points x dimension

        # ========================= Spatial Normalization ==========================
        if visualize:
            visualize_skeleton_openpose(keypoints,hand_left_keypoints, hand_right_keypoints, filename="figs/%04d.jpg"%visualization_counter)
            visualization_counter+=1

        normalize_point_x = keypoints[8, 0]
        normalize_point_y = keypoints[8, 1]

        keypoints[:, 0] -= normalize_point_x
        keypoints[:, 1] -= normalize_point_y

        hand_left_keypoints[:, 0] = hand_left_keypoints[:, 0]  - hand_left_keypoints[0, 0]
        hand_left_keypoints[:, 1] = hand_left_keypoints[:, 1] - hand_left_keypoints[0, 1]

        hand_right_keypoints[:, 0] = hand_right_keypoints[:, 0] - hand_right_keypoints[0,0]
        hand_right_keypoints[:, 1] = hand_right_keypoints[:, 1] - hand_right_keypoints[0,1]

        keypoints_array.append(np.reshape(keypoints, (-1)))
        hand_left_keypoints_array.append(np.reshape(hand_left_keypoints, (-1)))
        hand_right_keypoints_array.append(np.reshape(hand_right_keypoints, (-1)))

    # if visualize:
    #     os.system("ffmpeg -framerate 30 -i figs_tmp/%%04d.jpg -c:v libx264 -pix_fmt yuv420p figs_tmp/%s_%s.mp4" % (subject,emotion))
    #     os.system("find figs_tmp/ -maxdepth 1 -type f -iname \*.jpg -delete")

    return keypoints_array, hand_left_keypoints_array, hand_right_keypoints_array


class BodyFaceDataset(data.Dataset):
    def __init__(self, args, data = None, indices = None, subjects = None, phase = None, number = None):
        self.args = args
        self.phase = phase

        if args.db == "babyrobot":
            if data != None:
                bodies, hands_right, hands_left, lengths, Y, paths = data
                self.bodies = [bodies[x] for x in indices]
                self.hands_right = [hands_right[x] for x in indices]
                self.hands_left = [hands_left[x] for x in indices]
                self.lengths = [lengths[x] for x in indices]
                self.Y = [Y[x] for x in indices]
                self.paths = [paths[x] for x in indices]

            elif subjects !=None:
                self.bodies, self.hands_right, self.hands_left, self.lengths, self.Y, self.paths = get_babyrobot_data(subjects = subjects, phase = phase, number = number, num_classes = self.args.num_classes)

        self.lengths = []
        for index in range(len(self.bodies)):
            self.lengths.append(self.bodies[index].shape[0])

        self.features = []

    def set_scaler(self, scaler):
        self.scaler = scaler
        self.hands_right = [scaler['hands_right'].transform(x) for x in self.hands_right]
        self.hands_left = [scaler['hands_left'].transform(x) for x in self.hands_left]
        self.bodies = [scaler['bodies'].transform(x) for x in self.bodies]

    def to_tensors(self):
        self.hands_right = [torch.from_numpy(x).float() for x in self.hands_right]
        self.hands_left = [torch.from_numpy(x).float() for x in self.hands_left]
        self.bodies = [torch.from_numpy(x).float() for x in self.bodies]

    def prepad(self):
        """ prepad sequences to the max length sequence of each database """    
        max_len=323
        self.bodies = pad_sequence(self.bodies, batch_first=True, max_len = max_len)
        self.hands_right = pad_sequence(self.hands_right, batch_first=True, max_len = max_len)
        self.hands_left = pad_sequence(self.hands_left, batch_first=True, max_len = max_len)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        body = self.bodies[index]
        hand_right = self.hands_right[index]
        hand_left = self.hands_left[index]
        length = self.lengths[index]

        features = torch.Tensor(1)

        return {
            "body": body,
            "hand_left": hand_left,
            "hand_right": hand_right,
            "label": self.Y[index],
            "length": length,
            "paths": self.paths[index]
        }


# if __name__ == "__main__":
#     mt_path = mtcnn('../datasets/new_test/', '../datasets/test_alignm/', '../datasets/new_test/', False)
#     print(mt_path)
#     train_dataset = Dataset(data_path=mt_path, data_type='new_test', phase='new_test', transform=None, basic_aug=True)
#     print(train_dataset)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4,  num_workers=1, shuffle=False, pin_memory=True)
#     print(train_loader)

# if __name__ == '__main__':
#     data_transforms_test = transforms.Compose([transforms.ToPILImage(),
#                                                transforms.Resize((224, 224)),
#                                                transforms.ToTensor(),
#                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

#     # face_test_dataset = Dataset('../datasets/GEMEP/', 'GEMEP', 'test', transform=data_transforms_test, data=None, indices=None)
#     # from sklearn.model_selection import KFold
#     # data = get_all_face_data()
#     # all_file_path, all_labels = data
#     # kfold = KFold(n_splits=2)
#     # for fold, (train_index, test_index) in enumerate(kfold.split(all_file_path, all_labels)):
#     #     face_train_dataset = Dataset('../datasets/GEMEP/', 'GEMEP', phase='train', transform=data_transforms_test, basic_aug=True, data = data, indices=train_index)
#     #     val_dataset = Dataset('../datasets/GEMEP/', 'GEMEP', phase='val', transform=data_transforms_test, data = data, indices=np.concatenate((test_index, train_index), axis=None))
#     #     #np.concatenate((test_index, train_index), axis=None)
#     get_babyrobot_annotations(phase= 'train')
