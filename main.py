# reference : https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/Colorful/src/model/models_colorful.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils_data
import torchvision.transforms as transforms
from torchviz import make_dot
import hiddenlayer as hl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torchvision.datasets.utils import check_integrity, download_url
from os.path import join, exists, abspath, basename
from os import makedirs, listdir, getcwd, chdir
from PIL import Image
from time import time
import sys, os, cv2
from inspect import currentframe, getframeinfo


import torch.optim as optim
import torch.utils.data as utils_data

import matplotlib.pyplot as plt
import numpy as np
from lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable



class Net(nn.Module):
    #def __init__(self, n_class, batch_size):
    def __init__(self, batch_size):
        super(Net, self).__init__()
        #self.nb_class = n_class
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(1, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #print('\ntype(x.shape) :', type(x.shape)) 
        #print('x.shape :', x.shape) 
        batch_size = x.shape[0]
        #if torch.is_tensor(batch_size):
        #    batch_size = batch_size.item()

        #print('type(batch_size) :', type(batch_size))
        #print('batch_size :', batch_size)
        dev = torch.device("cuda") if next(self.parameters()).is_cuda else torch.device("cpu") 

        list_input_size = [1, 64, 128, 256, 512]
        list_output_size = [64, 128, 256, 512, 512]
        list_block_size = [2, 2, 3, 3, 3]
        subsample = [(2, 2), (2, 2), (2, 2), (1, 1), (1, 1)]

        # A trous blocks parameters
        list_input_size_atrous = [512, 512]
        list_output_size_atrous = [512, 512]
        list_block_size_atrous = [3, 3]

        #t1 = x.size()
        h, w = x.size()[2], x.size()[3]
        current_h, current_w = h, w
        block_idx = 1

        '''
        # First block
        f, b, s = list_filter_size[0], list_block_size[0], subsample[0]
        t1 = x.size()
        x = self.convolutional_block(x, block_idx, 1, f, b, s)
        block_idx += 1
        current_h, current_w = current_h / s[0], current_w / s[1]
        '''
        # Next blocks
        #for f, b, s in zip(list_filter_size[1:-1], list_block_size[1:-1], subsample[1:-1]):
        for i, f, b, s in zip(list_input_size[:-1], list_output_size[:-1], list_block_size[:-1], subsample[:-1]):
            x = convolutional_block(x, block_idx, i, f, b, s, dev)
            block_idx += 1
            current_h, current_w = current_h / s[0], current_w / s[1]

        # Atrous blocks
        for idx, (i, f, b) in enumerate(zip(list_input_size_atrous, list_output_size_atrous, list_block_size_atrous)):
            x = atrous_block(x, block_idx, i, f, b, dev)
            block_idx += 1

        # Block 7
        i, f, b, s = list_input_size[-1], list_output_size[-1], list_block_size[-1], subsample[-1]
        x = convolutional_block(x, block_idx, i, f, b, s, dev)
        block_idx += 1
        current_h, current_w = current_h / s[0], current_w / s[1]

        # Block 8
        # Not using Deconvolution at the moment
        # x = Deconvolution2D(256, 2, 2,
        #                     output_shape=(None, 256, current_h * 2, current_w * 2),
        #                     subsample=(2, 2),
        #                     border_mode="valid")(x)
        #x = UpSampling2D(size=(2, 2), name="upsampling2d")(x)
        x = nn.UpsamplingBilinear2d(scale_factor=2).to(dev)(x)
        current_h, current_w = current_h * 2, current_w * 2

        x = convolutional_block(x, block_idx, list_output_size[-1], 256, 2, (1, 1), dev)
        block_idx += 1
        #current_h, current_w = current_h * 2, current_w * 2
        current_h, current_w = current_h / s[0], current_w / s[1]

        # Final conv
        #x = Convolution2D(nb_classes, 1, 1, name="conv2d_final", border_mode="same")(x)
        #print('self.nb_class :', self.nb_class);    exit(0);
        #x = nn.Conv2d(256, self.nb_class, 1)(x)
        x = nn.Conv2d(256, 1, 1).to(dev)(x)

        #x = K.permute_dimensions(x, [0, 2, 3, 1])  # last dimension in number of filters
        #x = x.permute(0, 2, 3, 1)

        #x = K.reshape(x, (self.batch_size * current_h * current_w, nb_classes))
        #x = x.contiguous().view(self.batch_size * current_h * current_w, self.nb_class)
        x, input_size, trans_size, axis = softmax2D(x)
        # Add a zero column so that x has the same dimension as the target (313 classes + 1 weight)
        #xc = K.zeros((self.batch_size * current_h * current_w, 1))
        #xc = Variable(torch.zeros((self.batch_size * current_h * current_w, 1)))
        #print(self.batch_size);        print(current_h);        print(current_w);        exit(0)
        #xc = Variable(torch.zeros(int(self.batch_size * current_h * current_w), 1)).to(dev)
        '''
        if torch.is_tensor(current_w):
            current_w = current_w.item()
        if torch.is_tensor(current_h):
            current_h = current_h.item()
        print('type(current_h) :', type(current_h))
        print('type(current_w) :', type(current_w))
        '''
        if torch.is_tensor(batch_size) and torch.is_tensor(current_h) and torch.is_tensor(current_w):
            t_0 = int(batch_size.item() * current_h.item() * current_w.item())
        else:
            t_0 = int(batch_size * current_h * current_w)
        t_1 = torch.zeros(t_0, 1).to(dev)
        xc = Variable(t_1)
        #x = K.concatenate([x, xc], axis=1)
        #print('x.shape :', x.shape);    print('xc.shape :', xc.shape);  #exit(0)
        x = torch.cat([x, xc], 1)
        # Reshape back to (batch_size, h, w, nb_classes + 1) to satisfy keras' shape checks
        #x = K.reshape(x, (self.batch_size, current_h, current_w, self.nb_class + 1))
        #x = x.view(*trans_size)
        x = x.view(trans_size[0], trans_size[1], trans_size[2], trans_size[3] + 1)
        x = x.transpose(axis, len(input_size) - 1)
        #x = K.resize_images(x, h / current_h, w / current_w, "tf")
        x = nn.UpsamplingBilinear2d(size=(h, w)).to(dev)(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class ImageNetCustomFile(utils_data.Dataset):
    #def __init__(self, dataset_path, data_size, data_transform, li_label, ext_img):
    def __init__(self, dataset_path, size, ext_img, is_tiny_and_val = False):

        self.dataset_path = dataset_path
        #print('self.dataset_path :', self.dataset_path);    exit(0);
        self.size_img = (size, size)
        #self.num_samples = data_size
        #self.transform = data_transform
        self.li_fn_img = []
        for dirpath, dirnames, filenames in os.walk(dataset_path):
            print('Reading image names under %s' % (dirpath))
            '''
            self.li_fn_img += \
                [join(dirpath, f) for f in filenames
                 if f.lower().endswith(ext_img.lower())]
            print('self.__len__() :', self.__len__())
            if self.__len__() >= 200:
                break
            '''
            for f in filenames:
                if f.lower().endswith(ext_img.lower()):
                    self.li_fn_img.append(join(dirpath, f))
                    #if self.__len__() >= 4000:
                    #    break
            #if self.__len__() >= 4000:
            #    break
        '''
        print('self.li_fn_img :', self.li_fn_img)
        print('len(self.li_fn_img) :', len(self.li_fn_img))
        if is_tiny_and_val:
            exit(0)
        '''
        return

    def __getitem__(self, index):

        fn_img = self.li_fn_img[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img)
        #im_rgb = Image.open(fn_img).convert('RGB')
        im_bgr = cv2.imread(fn_img)
        im_bgr = cv2.resize(im_bgr, self.size_img, interpolation=cv2.INTER_AREA)
        #if self.transform is not None:
            #img = self.transform(img)
        im_lab = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2LAB)
        #t1, t2 = torch.from_numpy(im_lab[:, :, 0:1]), torch.from_numpy(im_lab[:, :, 1:])
        #return torch.from_numpy(im_lab[:, :, 0:1]), torch.from_numpy(im_lab[:, :, 1:])
        #return im_lab[:, :, 0:1].astype(np.float32), torch.from_numpy(im_lab[:, :, 1:].astype(np.float32))
        '''
        print('im_lab.dype :', im_lab.dtype);
        t1, t2 = ToTensor2(im_lab[:, :, 0:1]), ToTensor2(im_lab[:, :, 1:])
        t1_np = np.transpose(t1.cpu().detach().numpy(), (1, 2, 0));
        t2_np = np.transpose(t2.cpu().detach().numpy(), (1, 2, 0));
        print('t1_np.dtype :', t1_np.dtype);
        im_bgr_bbb, im_gray = reconstruct_color_image(t1_np, t2_np);
        print('fn_img :', fn_img);
        print('im_bgr_bbb.bmp saved');
        cv2.imwrite('im_bgr_bbb.bmp', im_bgr_bbb);  exit(0);      
        '''

        '''
        t_l = im_lab[:, :, 0:1];    t_ab = im_lab[:, :, 1:]
        print('t_l[1, 1] :', t_l[1, 1]);    print('t_ab[1, 1] :', t_ab[1, 1]);
        im_bgr_tmp, im_gray_tmp = reconstruct_color_image(t_l, t_ab);
        cv2.imwrite('im_bgr_tmp.bmp', im_bgr_tmp);  
        print('fn :', fn_img);        exit(0);
        print('saved im_bgr_tmp.bmp');        exit(0);
        '''
        return ToTensor2(im_lab[:, :, 0:1]), ToTensor2(im_lab[:, :, 1:])

    def __len__(self):
        return len(self.li_fn_img)


def softmaxND(input, axis=1):

    input_size = input.size()
    t1 = len(input_size) - 1
    trans_input = input.transpose(axis, t1)
    trans_size = trans_input.size()
    input_2d = trans_input.contiguous().view(-1, trans_size[-1])
    soft_max_2d = F.softmax(input_2d)
    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(axis, len(input_size) - 1)

def softmax2D(input, axis=1):
    input_size = input.size()
    t1 = len(input_size) - 1
    trans_input = input.transpose(axis, t1)
    trans_size = trans_input.size()
    input_2d = trans_input.contiguous().view(-1, trans_size[-1])
    soft_max_2d = F.softmax(input_2d, 1)
    #print('input_2d.shape :', input_2d.shape);  print('soft_max_2d.shape :', soft_max_2d.shape);    exit(0)
    return soft_max_2d, input_size, trans_size, axis


def convolutional_block(x, block_idx, n_input_channel, nb_filter,
                        nb_conv, subsample, dev):
    # 1st conv
    for i in range(nb_conv):
        #name = "block%s_conv2D_%s" % (block_idx, i);    print(name)
        if i < nb_conv - 1:
            # x = Convolution2D(nb_filter, 3, 3, name=name, border_mode="same")(x)
            x = nn.Conv2d(n_input_channel, nb_filter, 3, padding=1).to(dev)(x)
        else:
            # x = Convolution2D(nb_filter, 3, 3, name=name, subsample=subsample, border_mode="same")(x)
            x = nn.Conv2d(n_input_channel, nb_filter, 3, padding=1, stride=subsample).to(dev)(x)
        n_input_channel = nb_filter
        # x = BatchNormalization(mode=2, axis=1)(x)
        x = nn.BatchNorm2d(nb_filter).to(dev)(x)
        # x = Activation("relu")(x)
        x = F.relu(x)
    return x

def atrous_block(x, block_idx, n_input_channel, nb_filter, nb_conv, dev):

    # 1st conv
    for i in range(nb_conv):
        #name = "block%s_conv2D_%s" % (block_idx, i);   print(name)
        #x = AtrousConvolution2D(nb_filter, 3, 3, name=name, border_mode="same")(x)
        #x = nn.Conv2d(n_input_channel, nb_filter, 3, dilation=?)(x)
        x = nn.Conv2d(n_input_channel, nb_filter, 3, padding=1).to(dev)(x)

        #x = BatchNormalization(mode=2, axis=1)(x)
        x = nn.BatchNorm2d(nb_filter).to(dev)(x)
        #x = Activation("relu")(x)
        x = F.relu(x)
        n_input_channel = nb_filter
    return x

def ToTensor2(pic):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    to a torch.FloatTensor of shape (C x H x W)
    """
    if isinstance(pic, np.ndarray):
        #print('aaa');   exit(0);
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backard compability
        #return img.float().div(255)
        return img.float()
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        #return img.float().div(255)
        return img.float()
    else:
        return img


def check_if_uncompression_done(dir_save, foldername_train, foldername_test):

    #base_folder = 'imagenet'
    #fpath = join(dir_save, base_folder)
    #print(dir_save);    #exit(0);
    if not exists(dir_save):
        return False
    fpath_train = join(dir_save, foldername_train)
    #print(fpath_train);    exit(0);
    if not exists(fpath_train):
        return False
    fpath_test = join(dir_save, foldername_test)
    if not exists(fpath_test):
        return False
    return True



def check_if_download_done(dir_save, foldername_train, foldername_test):

    if not check_if_uncompression_done(dir_save, foldername_train, foldername_test):
        filename = "imagenet.tar.gz"
        fn = join(dir_save, filename)
        return exists(fn)
    return True

def check_if_image_set_exists(dir_save, li_label, n_im_per_label, ext_img):
    does_exist = True
    for label in li_label:
        dir_label = join(dir_save, label)
        if exists(dir_label):
            li_img_file = [file for file in listdir(dir_label) if file.endswith(ext_img)]
            n_img = len(li_img_file)
            if n_im_per_label != n_img:
                does_exist = False
                break
        else:
            does_exist = False
            break
    return does_exist



def saveTrainImages(dir_save, li_label, n_im_per_batch, foldername, ext_img):#byte_per_image, ):

    #data = {}
    #dataMean = np.zeros((3, ImgSize, ImgSize))
    dir_train = join(dir_save, foldername)
    i_total = 0
    for ifile in range(1, 6):
        fn_batch = join(join(dir_save, 'cifar-10-batches-py'), 'data_batch_' + str(ifile))
        with open(fn_batch, 'rb') as f:
            if sys.version_info[0] < 3:
                data = cp.load(f)
            else:
                data = cp.load(f, encoding='latin1')
            for i in range(n_im_per_batch):
                i_total += 1
                idx_label = data['labels'][i]
                name_label = li_label[idx_label]
                dir_label = abspath(join(dir_train, name_label))
                if not exists(dir_label):
                    makedirs(dir_label)
                fname = join(dir_label, ('%05d.%s' % (i + (ifile - 1) * 10000, ext_img)))
                saveImage(fname, data['data'][i, :])
                print('Saved %d th image of %s at %s' % (i_total, name_label, fname))

                #saveImage(fname, data['data'][i, :], data['labels'][i], mapFile, regrFile, 4, mean=dataMean)
    #dataMean = dataMean / (50 * 1000)
    #saveMean('CIFAR-10_mean.xml', dataMean)
    return

def saveTestImages(dir_save, li_label, n_im_per_batch, foldername, ext_img):

    #if not os.path.exists(foldername):
        #os.makedirs(foldername)
    dir_test = join(dir_save, foldername)
    fn_batch = join(join(dir_save, 'cifar-10-batches-py'), 'test_batch')
    i_total = 0
    with open(fn_batch, 'rb') as f:
        if sys.version_info[0] < 3:
            data = cp.load(f)
        else:
            data = cp.load(f, encoding='latin1')
        for i in range(n_im_per_batch):
            i_total += 1
            idx_label = data['labels'][i]
            name_label = li_label[idx_label]
            dir_label = abspath(join(dir_test, name_label))
            if not exists(dir_label):
                makedirs(dir_label)
            fname = join(dir_label, ('%05d.%s' % (i, ext_img)))
            saveImage(fname, data['data'][i, :])
            print('Saved %d th image of %s at %s' % (i_total, name_label, fname))



def prepare_imagenet_dataset(dir_save, foldername_train,
                             foldername_test):

    #dir_save = './data'
    #n_im_per_label_train, n_im_per_label_test = 5000, 1000
    #foldername_train, foldername_test = 'train', 'test'
    if not check_if_download_done(dir_save, foldername_train, foldername_test):
        print('The imagenet file has not been downloaded yet')
        sys.exit(1)
    if not check_if_uncompression_done(dir_save, foldername_train, foldername_test):
        print('The imagenet file has not been uncompressed yet')
        sys.exit(1)


#def make_dataloader_custom_file(dir_data, data_transforms, ext_img,                                n_img_per_batch, n_worker):
def make_dataloader_custom_file(dir_data, size_img, ext_img,
                                n_img_per_batch, n_worker, li_idx_sample_ratio):
    #is_tiny = 'tiny' in dir_data
    foldername_train, foldername_test = 'train', 'val'
    prepare_imagenet_dataset(dir_data, foldername_train, foldername_test)
    li_set = [foldername_train, foldername_test]
    #data_size = {'train' : 50000, 'test' : 10000}
    dsets = {x: ImageNetCustomFile(
        #join(dir_data, x), data_size[x], data_transforms[x], li_class, ext_img)
        join(dir_data, x), size_img, ext_img)
        #join(dir_data, x), size_img, ext_img, is_tiny and 'val' == x)
             for x in li_set}
    dset_loaders = {x: utils_data.DataLoader(
        dsets[x], batch_size = n_img_per_batch, shuffle = 'train' == x, num_workers = n_worker) for x in li_set}
    trainloader, testloader = dset_loaders[li_set[0]], dset_loaders[li_set[1]]
    n_val = len(dsets['val']) 
    li_idx_sample = [int(idx_sample_ratio * n_val) for idx_sample_ratio in li_idx_sample_ratio]
    li_fn_sample = [dsets['val'].li_fn_img[idx] for idx in li_idx_sample]
    print('li_idx_sample :', li_idx_sample);    #exit(0);
    print('li_fn_sample :', li_fn_sample);    #exit(0);
    return trainloader, testloader, li_idx_sample, li_fn_sample


def reconstruct_color_image(ll, aa_bb):
    #print('hw :', hw);  #exit(0);
    '''
    print('ll.shape :', ll.shape);  #exit(0);
    print('aa_bb.shape :', aa_bb.shape);  #exit(0);
    print('ll[1, 1] :', ll[1, 1]);  #exit(0);
    print('aa_bb[1, 1] :', aa_bb[1, 1]);  #exit(0);
    '''
    im_lab = np.dstack((ll, aa_bb));
    im_lab = im_lab.astype(np.uint8)
    #print('im_lab.shape :', im_lab.shape);  #exit(0);
    im_bgr = cv2.cvtColor(im_lab, cv2.COLOR_LAB2BGR)
    im_gray = ll
    return im_bgr, im_gray
    
def categorical_crossentropy_color(y_true, y_pred):

    print(y_true.size());   print(y_pred.size())
    print(type(y_true));    print(type(y_pred));    #exit(0)
    # Flatten
    n, q, h, w = y_true.shape
    print('b4 n :', n, ', h :', h, ', w :', w, ', q :', q);    #exit(0);
    #y_true = K.reshape(y_true, (n * h * w, q))
    #y_pred = K.reshape(y_pred, (n * h * w, q))
    y_true = y_true.permute(0, 2, 3, 1);    y_pred = y_pred.permute(0, 2, 3, 1);
    n, h, w, q = y_true.shape
    print('after n :', n, ', h :', h, ', w :', w, ', q :', q);    exit(0);
    y_true.reshape(n * h * w, q);   y_pred.reshape(n * h * w, q)
    weights = y_true[:, 313:]  # extract weight from y_true
    weights = K.concatenate([weights] * 313, axis=1)
    y_true = y_true[:, :-1]  # remove last column
    y_pred = y_pred[:, :-1]  # remove last column

    # multiply y_true by weights
    y_true = y_true * weights
    frameinfo = getframeinfo(currentframe());   print(frameinfo.filename, frameinfo.lineno)
    cross_ent = K.categorical_crossentropy(y_pred, y_true)
    cross_ent = K.mean(cross_ent, axis=-1)
    return cross_ent

def initialize(dev, dir_data, size_img, #di_set_transform,
               ext_img, n_img_per_batch, n_worker, li_idx_sample_ratio = None
               #, n_class
               ):
               

    trainloader, testloader, li_idx_sample, li_fn_sample =\
        make_dataloader_custom_file(
            dir_data, size_img, #di_set_transform,
            ext_img, n_img_per_batch, n_worker, li_idx_sample_ratio)

    #net = Net().cuda()
    #net = Net(n_class, n_img_per_batch)
    net = Net(n_img_per_batch)
    #t1 = net.cuda()
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    #print('is_gpu :', is_gpu);  exit(0);
    #if is_gpu:
    #    net.cuda()
    #    criterion.cuda()
    net = net.to(dev);  criterion.to(dev);
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=1, patience = 8, epsilon=0.00001, min_lr=0.000001) # set up scheduler

    return trainloader, testloader, net, criterion, optimizer, scheduler, li_idx_sample, li_fn_sample



def validate_epoch(net, n_loss_rising, loss_avg_pre, ax,
                   li_n_img_val, li_loss_avg_val,
                   testloader, criterion, th_n_loss_rising,
                   kolor, n_img_train, sec, is_gpu, idx_eopch, 
                   li_idx_sample, li_fn_sample):
    #print('idx_eopch :', idx_eopch);    exit(0);
    net.eval()
    shall_stop = False
    sum_loss = 0
    n_img_val = 0
    i_sample = 0;   n_sample = len(li_idx_sample)
    start_val = time()
    with torch.no_grad():
        for i, data in enumerate(testloader):
            inputs, labels = data
            n_img_4_batch = labels.size()[0]
            if is_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)
            #images, labels = images.cuda(), labels.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            #sum_loss += loss.data[0]
            sum_loss += loss.item()
            n_img_val += n_img_4_batch
            while i_sample < n_sample:
                if n_img_val <= li_idx_sample[i_sample]:
                    break
                idx_sample = n_img_4_batch - (n_img_val - li_idx_sample[i_sample])
                #print('idx_sample :', idx_sample);  #exit(0)
                im_bgr_sample, im_gray_sample = reconstruct_color_image(
                    np.transpose(inputs[idx_sample].cpu().detach().numpy(), (1, 2, 0)),
                    np.transpose(outputs[idx_sample].cpu().detach().numpy(), (1, 2, 0)))
                    #         list(inputs.shape[-2:]))
                print('li_fn_sample[i_sample] :', li_fn_sample[i_sample])
                #print('idx_eopch :', idx_eopch);    #exit(0);
                fn_bgr_sample = 'im_bgr_sample_{0:06d}_{1:02d}.bmp'.format(li_idx_sample[i_sample], idx_eopch)
                cv2.imwrite(fn_bgr_sample, im_bgr_sample);
                print('Validation sample result image is saved at ', fn_bgr_sample);    #exit(0)
                #fn_gray_sample = 'im_gray_sample_{0:06d}_{0:03d}.bmp'.format(li_idx_sample[i_sample], idx_eopch)
                #cv2.imwrite(fn_gray_sample, im_gray_sample);
                #print('Input gray image is saved at', fn_gray_sample);    #exit(0)
                i_sample += 1

    lap_val = time() - start_val
    loss_avg = sum_loss / n_img_val
    if loss_avg_pre <= loss_avg:
        n_loss_rising += 1
        if n_loss_rising >= th_n_loss_rising:
            shall_stop = True
    else:
        n_loss_rising = max(0, n_loss_rising - 1)
    li_n_img_val.append(n_img_train)
    li_loss_avg_val.append(loss_avg)
    ax.plot(li_n_img_val, li_loss_avg_val, c=kolor)
    plt.pause(sec)
    loss_avg_pre = loss_avg
    return shall_stop, net, n_loss_rising, loss_avg_pre, ax, \
           li_n_img_val, li_loss_avg_val, lap_val, n_img_val


def train_epoch(
        net, trainloader, optimizer, criterion, scheduler, n_img_total,
        n_img_interval, n_img_milestone, running_loss, is_lr_just_decayed,
        li_n_img, li_loss_avg_train, ax_loss, sec, epoch,
        kolor, interval_train_loss, dev):#, loss_l2):
    shall_stop = False
    net.train()
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        '''
        im_bgr_aaa, im_gray = reconstruct_color_image(
            np.transpose(inputs[0].cpu().detach().numpy(), (1, 2, 0)),
            np.transpose(labels[0].cpu().detach().numpy(), (1, 2, 0)))
        print('im_bgr_aaa.bmp saved');
        cv2.imwrite('im_bgr_aaa.bmp', im_bgr_aaa);  exit(0);      
        '''
        n_img_4_batch = labels.size()[0]
        #print('\n', i, ' / ', len(trainloader), ', n_img_4_batch :', n_img_4_batch)
        # wrap them in Variable
        # inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        #if is_gpu:
        #    inputs, labels = inputs.cuda(), labels.cuda()
        inputs = inputs.to(dev);    labels = labels.to(dev)
        inputs, labels = Variable(inputs), Variable(labels)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        #print('next(input.parameters()).is_cuda :', next(input.parameters()).is_cuda);  #exit(0);
        #print('next(net.parameters()).is_cuda :', next(net.parameters()).is_cuda);  #exit(0);
        inputs.requires_grad_(True)
        outputs = net(inputs)
        '''
        graph = hl.build_graph(net, torch.zeros([1, 1, 64, 64]).to(dev));    graph.them = hl.graph.THEMES['blue'].copy();    
        graph.save('colorization_hiddenlayer', format='png')
        '''
        make_dot(outputs, params = dict(list(net.named_parameters()) + [('inputs', inputs)])).render('colorization_torchviz', format='png');    exit(0)
        #print(inputs.size());   print(labels.size());   print(outputs.size());  exit(0);
        # labels += 10
        loss = criterion(outputs, labels)
        #frameinfo = getframeinfo(currentframe());   print(frameinfo.filename, frameinfo.lineno)
        #loss = categorical_crossentropy_color(outputs, labels)
        #loss = loss_l2(outputs, labels)
        #print('loss :', loss);  exit(0);
        loss.backward()
        optimizer.step()
        # n_image_total += labels.size()[0]
        # print statistics
        #print('type(loss) :', type(loss));
        #print('loss.shape :', loss.shape);  exit(0)
        #running_loss += loss.data[0]
        running_loss += loss.item()
        #n_image_total += n_img_per_batch
        n_img_total += n_img_4_batch
        n_img_interval += n_img_4_batch

        #if n_image_total % interval_train_loss == interval_train_loss - 1:  # print every 2000 mini-batches
        #if n_image_total % interval_train_loss == 0:  # print every 2000 mini-batches
        if n_img_total > n_img_milestone:  # print every 2000 mini-batches

            # if i % 2000 == 1999:    # print every 2000 mini-batches
            running_loss_avg = running_loss / n_img_interval
            li_n_img.append(n_img_total)
            li_loss_avg_train.append(running_loss_avg)
            ax_loss.plot(li_n_img, li_loss_avg_train, c=kolor)
            plt.pause(sec)
            #i_batch += 1
            print('[%d, %5d] avg. loss per image : %.5f' %
                  (epoch + 1, i + 1, running_loss_avg))
            is_best_changed, is_lr_decayed = scheduler.step(
                running_loss_avg, n_img_total)  # update lr if needed
            running_loss = 0.0
            n_img_interval = 0
            n_img_milestone = n_img_total + interval_train_loss
            #'''
            #if is_lr_just_decayed and (not is_best_changed):
            if is_lr_just_decayed and is_lr_decayed:
                shall_stop = True
                break
            #'''
            is_lr_just_decayed = is_lr_decayed
    #return shall_stop, net, optimizer, scheduler, n_img_total, n_img_interval, \
    #       n_img_milestone, running_loss, li_n_img, li_loss_avg_train, #ax_loss_train,
    #       is_lr_just_decayed, i + 1
    return shall_stop, net, optimizer, scheduler, n_img_total, n_img_interval, \
           n_img_milestone, running_loss, li_n_img, li_loss_avg_train, is_lr_just_decayed, i + 1









def prepare_display(interval_train_loss, lap_init, color_time, sec):
    #fig = plt.figure(num=None, figsize=(1, 2), dpi=500)
    fig = plt.figure(num=None, figsize=(12, 18), dpi=100)
    plt.ion()
    ax_time = fig.add_subplot(2, 1, 1)
    ax_time.set_title(
        'Elapsed time (sec.) of validation on 10k images vs. epoch. Note that value for epoch 0 is the elapsed time of init.')
    ax_time.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_loss = fig.add_subplot(2, 1, 2)
    ax_loss.set_title('Avg. train and val. loss per image vs. # train input images')
    ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))

    li_n_img_train, li_n_img_val, li_loss_avg_train, li_loss_avg_val = [], [], [], []
    li_lap, li_epoch = [lap_init], [0]
    n_img_milestone = interval_train_loss

    ax_time.plot(li_epoch, li_lap, c = color_time)
    plt.pause(sec)

    return ax_time, ax_loss, \
           li_n_img_train, li_n_img_val, li_loss_avg_train, li_loss_avg_val, \
           li_lap, li_epoch, n_img_milestone




def train(dev, trainloader, testloader, net, criterion, optimizer, scheduler, #li_class,
          n_epoch, lap_init, n_img_per_batch, interval_train_loss, li_idx_sample, li_fn_sample):#, loss_l2):

    sec = 0.01
    is_lr_just_decayed = False
    n_image_total, n_img_interval, running_loss = 0, 0, 0.0
    n_loss_rising, th_n_loss_rising, loss_avg_pre = 0, 3, 100000000000
    di_ax_color = {'time' : np.random.rand(3), 'train' : np.random.rand(3),
                   'val' : np.random.rand(3)}
    ax_time, ax_loss, li_n_img_train, li_n_img_val, \
    li_loss_avg_train, li_loss_avg_val, li_lap, li_epoch, n_img_milestone = \
        prepare_display(interval_train_loss, lap_init, di_ax_color['time'], sec)
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        print('epoch : %d' % (epoch + 1))
        shall_stop_train, net, optimizer, scheduler, n_image_total, n_img_interval, \
        n_img_milestone, running_loss, li_n_img_train, li_loss_avg_train, is_lr_just_decayed, n_batch \
        = train_epoch(
            net, trainloader, optimizer, criterion, scheduler, n_image_total,
            n_img_interval, n_img_milestone, running_loss, is_lr_just_decayed,
            li_n_img_train, li_loss_avg_train, ax_loss, sec, epoch,
            di_ax_color['train'], interval_train_loss, dev)#, loss_l2)
        shall_stop_val, net, n_loss_rising, loss_avg_pre, ax_loss_val, \
        li_n_img_val, li_loss_avg_val, lap_val, n_img_val = \
            validate_epoch(
                net, n_loss_rising, loss_avg_pre, ax_loss,
                li_n_img_val, li_loss_avg_val,
                testloader, criterion, th_n_loss_rising, di_ax_color['val'],
                n_image_total, sec, is_gpu, epoch, li_idx_sample, li_fn_sample)
        #lap_train = time() - start_train
        n_batch_val = n_img_val / n_img_per_batch
        lap_batch = lap_val / n_batch_val
        li_lap.append(lap_val)
        li_epoch.append(epoch + 1)
        #ax_time.plot(li_epoch, li_lap, c=kolor)
        ax_time.plot(li_epoch, li_lap, c = di_ax_color['val'])
        #ax_time.legend()
        #ax_time.show()
        plt.show()
        plt.pause(sec)
        if shall_stop_train or shall_stop_val:
            break
    '''
    ax_time.plot(li_epoch, li_lap, c=kolor, label=legend)
    ax_time.legend()
    ax_loss_train.plot(li_n_img_train, li_loss_avg_train, c=kolor, label=legend)
    ax_loss_train.legend()
    ax_loss_val.plot(li_n_img_val, li_loss_avg_val, c=kolor, label=legend)
    ax_loss_val.legend()
    plt.pause(sec)
    '''
    print('Finished Training')

    return

def main():

    #is_gpu = False
    #is_gpu = torch.cuda.device_count() > 0
    dev = torch.device("cuda") if torch.cuda.device_count() > 0 else torch.device("cpu") 
    #print('is_gpu :', is_gpu);  exit(0);
    #dir_data = './data'
    #dir_data = '/mnt/data/data/imagenet'
    dir_data = '/workspace/tiny-imagenet-200'
    ext_img = 'jpeg'
    #n_epoch = 100
    n_epoch = 50
    #n_img_per_batch = 40
    #n_img_per_batch = 60
    n_img_per_batch = 1500
    #n_img_per_batch = 2
    n_worker = 4
    #n_worker = 1
    #n_worker = 40
    #size_img = 256
    size_img = 64
    n_class = 300
    interval_train_loss = int(round(20000 / n_img_per_batch)) * n_img_per_batch
    li_idx_sample_ratio = [0.2, 0.4, 0.6, 0.8]

    '''
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    di_set_transform = {'train' : transform, 'test' : transform}
    '''
    start = time()
    trainloader, testloader, net, criterion, optimizer, scheduler, li_idx_sample, li_fn_sample =\
        initialize(
            dev, dir_data, size_img, #di_set_transform,
            ext_img, n_img_per_batch, n_worker, li_idx_sample_ratio
            #, n_class
            )
    #loss_l2 = nn.MSELoss()         
    lap_init = time() - start
    train(dev, trainloader, testloader, net, criterion, optimizer, scheduler,  # li_class,
          n_epoch, lap_init, n_img_per_batch, interval_train_loss, li_idx_sample, li_fn_sample)#, loss_l2)
    return

if __name__ == "__main__":
    main()

