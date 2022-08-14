import matplotlib.pyplot as plt
#from mtcnn import MTCNN
from tqdm import trange
from PIL import Image
import numpy as np
import itertools
import cv2
import os

'''Fusing Body'''
import os
import wandb
import torch
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [16, 9]


'''ARM'''
def add_gaussian_noise(image_array, mean=0.0, var=30):
    std = var**0.5
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped

def flip_image(image_array):
    return cv2.flip(image_array, 1)

def color2gray(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    gray_img_3d = image_array.copy()
    gray_img_3d[:, :, 0] = gray
    gray_img_3d[:, :, 1] = gray
    gray_img_3d[:, :, 2] = gray
    return gray_img_3d

def mtcnn(img_path, save_path, new_test_path, mtcnn):
    img_path_return = []
    if mtcnn:
        all_img = os.listdir(img_path)
        
        single_img_path = all_img
        for i in trange(len(all_img)):
            single_img_path[i] = img_path + all_img[i]
            print(all_img[i])
            
            
            im_open = Image.open(single_img_path[i])
            img = cv2.cvtColor(cv2.imread(single_img_path[i]), cv2.COLOR_BGR2RGB)
            detector = MTCNN()
            im=detector.detect_faces(img)
            if im :
                acurracy = im[0]['confidence']
                if acurracy>= 0.9:
                    bouding = im[0]['box']
                    im1=im_open.crop((bouding[0],bouding[1],(bouding[0]+bouding[2]),(bouding[1]+bouding[3])))
                    im1=im1.resize((224,224))
                    
                    file_path = save_path + all_img[i].split('/')[-1]
                    im1.save(file_path)
                img_path_return.append(file_path)
    else:
        for i in range(len(os.listdir(new_test_path))):
            img_path_return.append(new_test_path + os.listdir(new_test_path)[i])
            
    return img_path_return

def plot_confusion_matrix(cm, labels_name, title, acc, num_classes):
    cm = cm / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.imshow(cm, interpolation='nearest')  # 在特定的窗口上显示图像
    plt.title(title)  # 图像标题
    plt.colorbar()
    num_class = np.array(range(len(labels_name)))  # 获取标签的间隔数
    plt.xticks(num_class, labels_name, rotation=90)  # 将标签印在x轴坐标上
    plt.yticks(num_class, labels_name)  # 将标签印在y轴坐标上
    plt.ylabel('Target')
    plt.xlabel('Prediction')
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.tight_layout()
    plt.savefig(os.path.join('../Confusion_matrix/GEMEP', title + str(num_classes) + "_acc" + str(acc) + ".png"), format='png')
    plt.show()


'''Fusing Body'''
def accuracy(output, target, topk=(1,), weighted = False):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def visualize_skeleton_openpose(joints, hand_left, hand_right, filename="fig.png"):
    joints_edges = [[15, 17], [15, 0], [16, 0], [16, 18], [1, 0], [1, 2],
                  [3, 2], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
                  [10, 11], [11, 24], [23, 22], [8, 12], [13, 12], [13, 14], [14, 21], [19, 21],
                  [19, 20]]

    hands_edges = [[0, 1], [1, 2], [2, 3], [3, 4],
                       [0, 5], [5, 6], [6, 7], [7, 8],
                       [0, 9], [9, 10], [10, 11], [11, 12],
                       [0, 13], [13, 14], [14, 15], [15, 16],
                       [0, 17], [17, 18], [18, 19], [19, 20]]

    import matplotlib.animation as animation
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # def update_plot(i, data, scat):
    #     scat.set_offsets(data[i].reshape(-1,2))
    #     return scat
    joints[joints[:,2]<0.01] = np.nan
    joints[np.isnan(joints[:,2])] = np.nan

    hand_right[hand_right[:,2]<0.01] = np.nan
    hand_right[np.isnan(hand_right[:,2])] = np.nan

    hand_left[hand_left[:,2]<0.01] = np.nan
    hand_left[np.isnan(hand_left[:,2])] = np.nan

    # hand_right[hand_right<0.3] = 'nan'
    # hand_left[hand_left[:,2]<0.3] = 'nan'
    # skeleton = sequence[frame].reshape(-1, 2)
    # joints[:,0] = 1-joints[:,0]
    scat = ax.scatter(joints[:, 0], joints[:, 1])
    for edge in joints_edges:
        ax.plot((joints[edge[0], 0], joints[edge[1], 0]),
                (joints[edge[0], 1], joints[edge[1], 1]))

    joints = hand_right
    # joints[:,0] = 1-joints[:,0]
    scat = ax.scatter(joints[:, 0], joints[:, 1])
    for edge in hands_edges:
        ax.plot((joints[edge[0], 0], joints[edge[1], 0]),
                (joints[edge[0], 1], joints[edge[1], 1]))


    joints = hand_left
    # joints[:,0] = 1-joints[:,0]
    scat = ax.scatter(joints[:, 0], joints[:, 1])
    for edge in hands_edges:
        ax.plot((joints[edge[0], 0], joints[edge[1], 0]),
                (joints[edge[0], 1], joints[edge[1], 1]))

    # ax.set_xlim(right=1, left=0)
    # ax.set_ylim(top=1, bottom=0)
    plt.gca().invert_yaxis()


    plt.savefig(filename)
    plt.close()

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

def get_weighted_loss_weights(dataset, num_classes):
    print("Calculating sampler weights...")
    # labels_array = [x['emotion'] for x in dataset.data]
    labels_array = dataset#.Y_body

    from sklearn.utils import class_weight
    import numpy as np
    class_weights = class_weight.compute_class_weight('balanced', np.unique(labels_array), labels_array)
    assert(class_weights.size == num_classes)
    # class_weights = 1/class_weights
    print("Class Weights: ", class_weights)
    return class_weights

def pad_sequence(sequences, batch_first=False, padding_value=0, max_len=100):
    r"""Pad a list of variable length Tensors with zero

    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise.

    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *`` where `T` is the
            length of the longest sequence.
        Function assumes trailing dimensions and type of all the Tensors
            in sequences are same.

    Arguments:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``T x B x *`` if batch_first is False
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    # max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor

def mkdir_p(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def date_path(dataset):
    date_path = './checkpoints/' + dataset + '/'
    for i in range(3):
        date_path = date_path + str(date.today()).split('-')[i]
    mkdir_p(date_path)

    return date_path

def wandb_update(args):
    config = wandb.config
    '''Optimizer Parameters'''
    config.learning_rate = args.lr
    config.step_size = args.step_size
    config.weight_d = args.weight_decay
    config.optimizer = args.optimizer
    config.momentum = args.momentum

    '''Usual Hyper Parameters'''
    config.batch_size = args.batch_size
    config.database = args.dataset
    config.exp_name = args.exp_name
    config.epochs = args.epochs

    '''Network Parameters'''
    config.checkpoint = args.checkpoint
    config.mode = args.mode
    config.confidence_threshold = args.confidence_threshold
    config.num_classes = args.num_classes
    config.num_total_iterations = args.num_total_iterations
    config.num_splits = args.num_splits
    config.add_body_dnn = args.add_body_dnn
    config.first_layer_size = args.first_layer_size

    '''Training Parameters'''
    config.body_pooling = args.body_pooling


if __name__ == "__main__":
    img_path = '../datasets/test/'
    save_path = '../datasets/test_alignm/'
    print(mtcnn(img_path, save_path))