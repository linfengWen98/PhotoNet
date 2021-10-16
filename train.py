import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
from torch.optim import lr_scheduler
from models import VGGEncoder, PhotoNetDecoder, NASDecoder


class TransferDataset(Dataset):
    def __init__(self, img_dir, size):
        super(TransferDataset, self).__init__()
        self.img_dir = img_dir
        self.size = size
        self.content_name_list = self.get_name_list(self.img_dir)
        self.transforms = self.transform()

    def get_name_list(self, name):
        name_list = os.listdir(name)
        name_list = [os.path.join(name, i) for i in name_list]
        np.random.shuffle(name_list)
        return name_list

    def transform(self):
        data_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.RandomCrop((self.size, self.size)),
            transforms.ToTensor()
        ])
        return data_transform

    def __len__(self):
        a = len(self.content_name_list)
        return a

    def __getitem__(self, item):
        img = Image.open(self.content_name_list[item]).convert('RGB')
        img_out = self.transforms(img)
        return img_out


def get_loss(net_e, net_d, content):
    fc = net_e(content)
    content_new = net_d(*fc)
    fc_new = net_e(content_new)
    mse_loss = nn.MSELoss()
    loss_r = mse_loss(content_new, content)
    loss_p_list = []
    for i in range(5):
        loss_p_list.append(mse_loss(fc_new[i], fc[i]))
    loss_p = sum(loss_p_list) / len(loss_p_list)
    loss = 0.5 * loss_r + 0.5 * loss_p
    return loss


def train(args):
    encoder = VGGEncoder()
    encoder.load_state_dict(torch.load(args.encoder_path))
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    if args.model == 'PhotoNet':
        decoder = PhotoNetDecoder()
    elif args.model == 'PhotoNAS':
        decoder = NASDecoder()
    else:
        assert 0, "Unsupported model: {}".format(args.model)

    decoder.train()
    if args.cuda:
        encoder.cuda()
        decoder.cuda()

    bs = args.batch_size
    transferset = TransferDataset(args.dataset, args.size)
    loader = DataLoader(transferset, batch_size=bs, shuffle=True, num_workers=bs, drop_last=True)
    opt = torch.optim.Adam(decoder.parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=0.0001)
    scheduler = lr_scheduler.StepLR(opt, step_size=10000, gamma=0.6, last_epoch=-1)

    bs_total = len(transferset) / bs
    for epoch in range(args.epoch):
        total_loss = 0
        for i, batch in enumerate(loader):
            opt.zero_grad()
            batch.requires_grad = False
            if args.cuda:
                batch = batch.cuda()

            loss = get_loss(encoder, decoder, batch)
            loss.backward(retain_graph=False)
            opt.step()

            if i % 100 == 0:
                print('epoch: %d | batch: %d / %d | loss:%.6f' % (epoch+1, i+1, bs_total, loss.item()))
            total_loss += loss.item()

            scheduler.step()
        print('--epoch: ', epoch+1, '  total_loss: ', total_loss)

        torch.save({
            'epoch': epoch+1,
            'model_state_dict': decoder.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
        }, '{:s}/{:s}_decoder_ckpoint_epoch_{:d}.pth.tar'.format(args.save_dir, args.model, epoch + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help='dir to the training images.')
    parser.add_argument('-m', '--model', type=str, default='PhotoNet', choices=['PhotoNet', 'PhotoNAS'])
    parser.add_argument('-e', '--epoch', type=int, default=2)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-s', '--size', type=int, default=512)
    parser.add_argument('-c', '--cuda', type=bool, default=True)
    parser.add_argument('--save_dir', type=str, default='ckpoint')
    parser.add_argument('--encoder_path', type=str, default='ckpoint/vgg_normalised_conv5_1.pth')
    args = parser.parse_args()

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    train(args)

