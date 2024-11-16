import sys
import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from torchvision import datasets

from pre_dataset import GetLoader
import constants as consts
import parser

args = parser.arg_parse()

cuda = True if torch.cuda.is_available() else False
cudnn.benchmark = True
alpha = 0

######################################################################
# load data
######################################################################
dataset = GetLoader(
    img_root=args.image_dir,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
)

print('# images in dataset:', len(dataset))
dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=consts.batch_size,
    shuffle=False,
    num_workers=consts.workers
)
sample_batch = next(iter(dataloader))
print('Image tensor in each batch:', sample_batch[0].shape, sample_batch[0].dtype)

######################################################################
# load model
######################################################################
if args.model_path == "bestMtoU.pth":
    print("bestMtoU.pth")
elif args.model_path == "bestMtoS.pth":
    print("bestMtoS.pth")
my_net = torch.load(args.model_path)
my_net = my_net.eval()

######################################################################
# predict
######################################################################
if cuda:
    my_net = my_net.cuda()
n_total = 0
n_correct = 0
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

output_lines=[]
for i, (t_img, t_img_fns) in enumerate(dataloader):
    batch_size = len(t_img)
    input_img = FloatTensor(batch_size, 3, consts.image_size, consts.image_size)
    if cuda:
        t_img = t_img.cuda()
        input_img = input_img.cuda()
    input_img.resize_as_(t_img).copy_(t_img)

    class_output, _ = my_net(input_data=input_img, alpha=alpha)
    pred = class_output.data.max(1, keepdim=True)[1]

    for j,img_fn in enumerate(t_img_fns):
        output_lines.append([img_fn, pred[j][0].item()])

######################################################################
# output as csv file
######################################################################

with open(os.path.join(args.pre_label_path),'w') as f:
    f.write("image_name" + ',' + "label")
    f.write('\n')
    for i,line in enumerate(output_lines):
        f.write(line[0]+','+str(line[1]))
        if (i==len(output_lines)-1): break
        f.write('\n')
print("save predicted file at",os.path.join(args.pre_label_path))
