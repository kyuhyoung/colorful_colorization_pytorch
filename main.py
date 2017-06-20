import torchvision.transforms as transforms
from os.path import join, exists, abspath, basename
from torchvision.datasets.utils import check_integrity, download_url

def check_if_uncompression_done(dir_save):

    base_folder = 'cifar-10-batches-py'

    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    root = dir_save
    for fentry in (train_list + test_list):
        filename, md5 = fentry[0], fentry[1]
        fpath = join(root, base_folder, filename)
        if not check_integrity(fpath, md5):
            return False
    return True


def check_if_download_done(dir_save):

    if not check_if_uncompression_done(dir_save):
        filename = "cifar-10-python.tar.gz"
        fn = join(dir_save, filename)
        return exists(fn)
    return True


def prepare_imagenet_dataset(dir_save, ext_img):

    #dir_save = './data'
    n_im_per_label_train, n_im_per_label_test = 5000, 1000
    foldername_train, foldername_test = 'train', 'test'
    if not check_if_download_done(dir_save):
        import tarfile
        url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        filename = "cifar-10-python.tar.gz"
        tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
        root = dir_save
        download_url(url, root, filename, tgz_md5)
    if not check_if_uncompression_done(dir_save):
        # extract file
        cwd = getcwd()
        tar = tarfile.open(join(root, filename), "r:gz")
        chdir(root)
        tar.extractall()
        tar.close()
        chdir(cwd)
        #loadData(url, dir_save)

    li_label, byte_per_image, n_im_per_batch = get_label_names(dir_save)
    dir_train, dir_test = join(dir_save, foldername_train), join(dir_save, foldername_test)
    if check_if_image_set_exists(
            dir_train, li_label, n_im_per_label_train, ext_img):
        print(dir_train + ' are already existing.')
    else:
        saveTrainImages(dir_save, li_label, n_im_per_batch, foldername_train, ext_img)
    if check_if_image_set_exists(
            dir_test, li_label, n_im_per_label_test, ext_img):
        print(dir_test + ' are already existing.')
    else:
        saveTestImages(dir_save, li_label, n_im_per_batch, foldername_test, ext_img)
    return li_label


def make_dataloader_custom_file(dir_data, data_transforms, ext_img):

    li_class = prepare_imagenet_dataset(dir_data, ext_img)
    li_set = ['train', 'test']
    data_size = {'train' : 50000, 'test' : 10000}
    dsets = {x: Cifar10CustomFile(
        join(dir_data, x), data_size[x], data_transforms[x], li_class, ext_img)
             for x in li_set}
    dset_loaders = {x: utils_data.DataLoader(
        dsets[x], batch_size=4, shuffle=True, num_workers=4) for x in li_set}
    trainloader, testloader = dset_loaders[li_set[0]], dset_loaders[li_set[1]]

    return trainloader, testloader, li_class

def initialize(dir_data, di_set_transform, ext_img):

    trainloader, testloader, li_class = make_dataloader_custom_file(
        dir_data, di_set_transform, ext_img)

    #net = Net().cuda()
    net = Net()
    #t1 = net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=1) # set up scheduler

    return trainloader, testloader, net, criterion, optimizer, scheduler, li_class

def main():
    dir_data = './data'
    ext_img = 'png'
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    di_set_transform = {'train' : transform, 'test' : transform}

    trainloader, testloader, net, criterion, optimizer, scheduler, li_class = \
        initialize(dir_data, di_set_transform, ext_img)

    return

if __name__ == "__main__":
    main()

