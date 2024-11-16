import os
import torchvision
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import parameters
from dataDefine import DataForm
from numpy.random import choice
from modelB import Generator, Discriminator, weights_init
import tqdm
from parameters import fix_random_seed

def load_checkpoint(pathG, pathD, model_G, optimizer_G, model_D, optimizer_D):
    state_G = torch.load(pathG)
    state_D = torch.load(pathD)
    model_G.load_state_dict(state_G['state_dict'])
    model_D.load_state_dict(state_D['state_dict'])
    optimizer_G.load_state_dict(state_G['optimizer'])
    optimizer_D.load_state_dict(state_D['optimizer'])

    # load from pre_train model
    checkpoint = torch.load(pathG)
    states_to_load = {}
    for name, param in checkpoint['state_dict'].items():
        if name.startswith('main'):
            states_to_load[name] = param
    model_state = model_G.state_dict()
    model_state.update(states_to_load)
    model_G.load_state_dict(model_state)

    print('model loaded from {} and {}'.format(pathG, pathD))


# randomly flip some labels to make data become more diff
def noisy_labels(y, p_flip):
    n_select = int(p_flip * y.shape[0])
    flip_ix = choice([i for i in range(y.shape[0])], size=n_select)
    y[flip_ix] = 1 - y[flip_ix]
    return y

def smooth_negative_labels(y):
    return np.random.uniform(low=0.0, high=0.2, size=y.shape)

def main():
    args = parameters.arg_parse()

    if not os.path.exists(args.ckpts_dir):
        os.makedirs(args.ckpts_dir)
    if not os.path.exists(args.save_train_result_dir):
        os.makedirs(args.save_train_result_dir)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device : " + device)
    # notice that training can't fix, it will be garbage and waste your time for training QQ
    # fix_random_seed(args.random_seed)

    print("===> Preparing dataloader...")
    train_data = DataForm(root=args.train_data, mode="train")
    train_loader = DataLoader(train_data, batch_size=args.train_batch, num_workers=args.num_workers, shuffle=True)
    print("===> Loading model...")
    model_G = Generator().to(device)
    model_G.apply(weights_init)
    model_D = Discriminator().to(device)
    model_D.apply(weights_init)

    print(model_G)
    print(model_D)

    criterion = nn.BCELoss()

    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=args.lr_d, weight_decay=args.weight_decay,
                                   betas=(0.5, 0.999))

    if args.ckpt_g and args.ckpt_d:
        load_checkpoint(args.ckpt_g, args.ckpt_d, model_G, optimizer_G, model_D, optimizer_D)
    real_sample = torch.randn((100, 100, 1, 1)).to(device)
    
    g_iter = args.g_iter
    d_iter = args.d_iter

    print("===> Start training...")
    for epoch in tqdm.trange(1, args.epochs + 1):
        if epoch > 100:
            g_iter = 1
            d_iter = 1

        iter = 0
        for idx, data in enumerate(train_loader):
            real_img = data[0]
            iter += 1
            model_D.train()
            model_G.train()
            batch_size = real_img.size(0)

            # ----------Start train model_D----------

            for _ in range(d_iter):
                model_D.zero_grad()

                real_img = real_img.to(device)
                label = torch.full((batch_size,), 1, dtype=torch.float, device=device)
                label = label.to(device)
                output = model_D(real_img).view(-1)
                label = label.float()
                D_loss_real = criterion(output, label)
                D_loss_real.backward()

                noise = torch.randn(batch_size, 100, 1, 1, device=device)
                # make fake image by model_G and be training data for model_D
                fake = model_G(noise)
                label.fill_(0)
                label = smooth_negative_labels(label.cuda())
                label = torch.from_numpy(label).float().to(device)
                output = model_D(fake.detach()).view(-1)
                D_loss_fake = criterion(output, label)
                D_loss_fake.backward()
                D_loss = D_loss_real + D_loss_fake
                optimizer_D.step()

            # ----------Start train model_G----------

            for _ in range(g_iter):
                model_G.zero_grad()
                label.fill_(1)
                label = label.float().to(device)
                output = model_D(fake).view(-1).float()
                G_loss = criterion(output, label)
                G_loss.backward()
                optimizer_G.step()

            if (iter + 1) % args.log_interval == 0:
                print(
                    "Epoch: {} [{}/{}] | G Loss:{:.4f} | D Loss: {:.4f}".format(
                        epoch, idx + 1, len(train_loader), G_loss.item(), D_loss.item()))

        with torch.no_grad():
            model_G.eval()
            fake_imgs_sample = (model_G(real_sample)).detach().cuda()
            model_G.train()
            # print(fake_imgs_sample) # be used to check
            # print(fake_imgs_sample.shape) # should be (100, 3, 64, 64)
            filename = os.path.join(args.save_train_result_dir, "epoch_{:03d}.jpg".format(epoch))
            torchvision.utils.save_image(fake_imgs_sample[:32], filename, nrow=8)

        # save model every 50 epochs
        if epoch % 50 == 0:
            print("Saving model G(Generator)...")
            state = {'state_dict': model_G.state_dict(),
                     'optimizer': optimizer_G.state_dict()}
            torch.save(state, os.path.join(args.ckpts_dir, "{:03d}_G.pth".format(epoch)))
            print("Saving model D(Discriminator)...")
            state = {'state_dict': model_D.state_dict(),
                     'optimizer': optimizer_D.state_dict()}
            torch.save(state, os.path.join(args.ckpts_dir, "{:03d}_D.pth".format(epoch)))


if __name__ == '__main__':
    main()
