import random
import numpy as np
import os

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

from test import test
from model import CNNModel
from dataset import GetLoader
import constants as consts
from torch.utils.data import Dataset, DataLoader
from data import DigitDataset
import parser

args = parser.arg_parse()

# You can change here for change source and target
source_dataset_name = 'mnistm'
target_dataset_name = 'svhn'
source_image_root = os.path.join('../hw2_data/digits/', source_dataset_name)
target_image_root = os.path.join('../hw2_data/digits/', target_dataset_name)
cudnn.benchmark = True

# Create models folder if needed
os.makedirs("models", exist_ok=True)

# Decide which device we want to run on
cuda = True if torch.cuda.is_available() else False

# ---------------------Dataset and DataLoader---------------------
dataset_source = GetLoader(
    # You can change here for change source and target
    img_root=os.path.join(source_image_root,"data"),
    label_path=os.path.join(source_image_root, 'train.csv'),
    transform=transforms.Compose([
        # transforms.Resize((56, 56)),
        # transforms.RandomHorizontalFlip(p=0.5), <- not good :(
        transforms.ToTensor(),  # to 0 ~ 1
        # We can notice that using same 0.5 is better than different paser
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # to -0.5 ~ 0.5
    ])
)
print('# images in dataset_source:', len(dataset_source))
dataloader_source = torch.utils.data.DataLoader(
    dataset=dataset_source,
    batch_size=consts.batch_size,
    shuffle=True,
    num_workers=consts.workers)

# source_data = DigitDataset(root=args.train_data, type=args.source, mode="train")
# dataloader_source = DataLoader(source_data, batch_size=args.train_batch, num_workers=args.num_workers, shuffle=True)

sample_batch = next(iter(dataloader_source))
print('Image tensor in each batch:', sample_batch[0].shape, sample_batch[0].dtype)
print('Label tensor in each batch:', sample_batch[1].shape, sample_batch[1].dtype)

# target_data = DigitDataset(root=args.train_data, type=args.target, mode="train")
# dataloader_target = DataLoader(target_data, batch_size=args.train_batch, num_workers=args.num_workers, shuffle=True)


dataset_target = GetLoader(
    img_root=os.path.join(target_image_root,"data"),
    label_path=os.path.join(target_image_root, 'train.csv'),
    transform=transforms.Compose([
        # transforms.Resize((56, 56)),

        transforms.ToTensor(),  # to 0~1
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # to -0.5~0.5
    ])
)
print('# images in dataset_target:', len(dataset_target))
dataloader_target = torch.utils.data.DataLoader(
    dataset=dataset_target,
    batch_size=consts.batch_size,
    shuffle=True,
    num_workers=consts.workers)

sample_batch = next(iter(dataloader_target))
print('Image tensor in each batch:', sample_batch[0].shape, sample_batch[0].dtype)
print('Label tensor in each batch:', sample_batch[1].shape, sample_batch[1].dtype)

# ---------------------Models------------------------
my_net = CNNModel()
print(my_net)


# ---------------------Loss Functions and Optimizers---------------------
optimizer = optim.Adam(my_net.parameters(), lr=consts.lr)
loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()


# ---------------------training loop---------------------
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
if cuda:
    my_net.cuda()
    loss_class.cuda()
    loss_domain.cuda()
for p in my_net.parameters():
    p.requires_grad = True

for epoch in range(consts.num_epochs):
    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    for i in range(len_dataloader):
        p = float(i + epoch * len_dataloader) / consts.num_epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        #  ---------------------Training model using source data---------------------
        data_source = data_source_iter.next()
        s_img, s_label = data_source
        batch_size = len(s_label)
        input_img = FloatTensor(batch_size, 3, consts.image_size, consts.image_size)
        class_label = LongTensor(batch_size)
        domain_label = LongTensor(np.zeros(batch_size))

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(s_img).copy_(s_img)
        class_label.resize_as_(s_label).copy_(s_label)

        my_net.zero_grad()

        class_output, domain_output = my_net(input_data=input_img, alpha=alpha)
        err_s_label = loss_class(class_output, class_label)
        err_s_domain = loss_domain(domain_output, domain_label)

        # ---------------------Training model using target data---------------------
        # if we don't want to use DANN like Lower or Upper bound, we just command here
        data_target = data_target_iter.next()
        t_img, _ = data_target  # we would not see the label
        batch_size = len(t_img)
        input_img = FloatTensor(batch_size, 3, consts.image_size, consts.image_size)
        domain_label = LongTensor(np.ones(batch_size))

        if cuda:
            t_img = t_img.cuda()
            input_img = input_img.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)

        _, domain_output = my_net(input_data=input_img, alpha=alpha)
        err_t_domain = loss_domain(domain_output, domain_label)

        err = err_t_domain + err_s_domain + err_s_label
        err.backward()
        optimizer.step()

    torch.save(my_net, 'models/model_temp.pth')
    test(target_dataset_name, epoch)