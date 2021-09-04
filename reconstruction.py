import os
import argparse
import torch
from models import VGGEncoder, PhotoNetDecoder, NASDecoder
from torchvision import transforms, utils
import cv2
import numpy as np


def reconstruction(args):
    img = cv2.imread(args.image)
    h, w, _ = img.shape
    m = max(h, w)
    if m > args.max_size:
        w = round(w * 1.0 / m * args.max_size)
        h = round(h * 1.0 / m * args.max_size)
    w = w // 16 * 16
    h = h // 16 * 16
    img_resize = cv2.resize(img, (w, h), cv2.INTER_AREA)

    img = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)

    net_e = VGGEncoder()
    net_e.load_state_dict(torch.load(args.encoder_path))
    if args.model == 'PhotoNet':
        net_d = PhotoNetDecoder()
    elif args.model == 'PhotoNAS':
        net_d = NASDecoder()
    else:
        assert 0, "Unsupported model: {}".format(args.model)

    checkpoint = torch.load(args.decoder_path)
    net_d.load_state_dict(checkpoint['model_state_dict'])

    if args.cuda:
        img = img.cuda()
        net_e = net_e.cuda()
        net_d = net_d.cuda()

    net_e.eval()
    net_d.eval()
    features = list(net_e(img))
    out = net_d(*features)

    utils.save_image(out, os.path.join(args.output_dir, 'reconstruction.jpg'))
    out = cv2.imread(os.path.join(args.output_dir, 'reconstruction.jpg'))
    out_compare = np.concatenate((img_resize, out), 1)
    cv2.imwrite(os.path.join(args.output_dir, 'compare.jpg'), out_compare)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PhotoNet')

    parser.add_argument('--image', type=str, default='content/1.jpg')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--max_size', type=int, default=640)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--model', type=str, default='PhotoNet', choices=['PhotoNet', 'PhotoNAS'])
    parser.add_argument('--encoder_path', type=str, default='ckpoint/vgg_normalised_conv5_1.pth')
    parser.add_argument('--decoder_path', type=str, default='ckpoint/PhotoNet_decoder_ckpoint_epoch_2.pth.tar')

    args = parser.parse_args()
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    reconstruction(args)
