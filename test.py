import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms

from dataset import GetLoader
import constants as consts
from torch.utils.data import Dataset, DataLoader
import parser

args = parser.arg_parse()


def test(dataset_name, epoch):
    image_root = os.path.join('../hw2_data/digits/', dataset_name)
    cuda = True if torch.cuda.is_available() else False

    cudnn.benchmark = True
    alpha = 0

    """load data"""
    dataset = GetLoader(
        img_root=os.path.join(image_root, 'data'),
        label_path=os.path.join(image_root, 'val.csv'),
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=consts.batch_size,
        shuffle=False,
        num_workers=8
    )
    # source_data = DigitDataset(root=args.train_data, type=args.source, mode="train")
    # dataloader_source = DataLoader(source_data, batch_size=args.train_batch, num_workers=args.num_workers, shuffle=True)
    #
    # sample_batch = next(iter(dataloader_source))
    # print('Image tensor in each batch:', sample_batch[0].shape, sample_batch[0].dtype)
    # print('Label tensor in each batch:', sample_batch[1].shape, sample_batch[1].dtype)
    #
    # target_data = DigitDataset(root=args.train_data, type=args.target, mode="train")
    # dataloader_target = DataLoader(target_data, batch_size=args.train_batch, num_workers=args.num_workers, shuffle=True)

    """ training """
    my_net = torch.load(os.path.join('models','model_temp.pth'))
    my_net = my_net.eval()
    if cuda:
        my_net = my_net.cuda()

    n_total = 0
    n_correct = 0

    for i, (t_img, t_label) in enumerate(dataloader):
        batch_size = len(t_label)
        input_img = torch.FloatTensor(batch_size, 3, consts.image_size, consts.image_size)
        class_label = torch.LongTensor(batch_size)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        class_label.resize_as_(t_label).copy_(t_label)

        class_output, _ = my_net(input_data=input_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size
    accu = n_correct.data.numpy() * 1.0 / n_total
    torch.save(my_net, 'models/epoch_{}_{}.pth'.format(epoch, accu))

    print ('epoch: %d, accuracy of the %s dataset: %f' % (epoch, dataset_name, accu))
