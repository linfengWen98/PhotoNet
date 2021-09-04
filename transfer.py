import argparse
import torch
from torchvision import transforms, utils
from models import VGGEncoder, PhotoNetDecoder, NASDecoder
from wct import transform
import cv2


def open_image(path, max_size=640):
    img = cv2.imread(path)
    h, w, _ = img.shape
    m = max(h, w)
    if m > max_size:
        w = round(w * 1.0 / m * max_size)
        h = round(h * 1.0 / m * max_size)
    w = w // 16 * 16
    h = h // 16 * 16
    content_resize = cv2.resize(img, (w, h), cv2.INTER_AREA)    # AREA
    content = cv2.cvtColor(content_resize, cv2.COLOR_BGR2RGB)
    content = transforms.ToTensor()(content)
    img = content.unsqueeze(0)
    return img


def stylize(args):
    content = open_image(args.content, args.max_size)
    style = open_image(args.style, args.max_size)

    # control the wct transform of layers
    en_transform = []
    de_transform = []

    net_e = VGGEncoder()
    net_e.load_state_dict(torch.load(args.encoder_path))

    if args.model == 'PhotoNet':
        net_d = PhotoNetDecoder()
        en_transform = [1, 1, 1, 1, 1]
        de_transform = [1, 1, 1, 1, 1, 0]
    elif args.model == 'PhotoNAS':
        net_d = NASDecoder()
        en_transform = [0, 0, 0, 0, 1]
        de_transform = [0, 0, 0, 0, 1, 0]
    else:
        assert 0, "Unsupported model: {}".format(args.model)

    checkpoint = torch.load(args.decoder_path)
    net_d.load_state_dict(checkpoint['model_state_dict'])

    if args.cuda:
        content = content.cuda()
        style = style.cuda()
        net_e = net_e.cuda()
        net_d = net_d.cuda()

    # --style transfer
    net_e.eval()
    net_d.eval()
    cF = list(net_e(content))
    sF = list(net_e(style))
    csF = []
    for c, s, t in zip(cF, sF, en_transform):
        if t:
            csF.append(transform(c, s, args.alpha, args.cuda))
        else:
            csF.append(c)

    layers = ['pyramid', 'conv5', 'conv4', 'conv3', 'conv2', 'conv1']
    for l, t in zip(layers, de_transform):
        csF[0] = net_d.forward_multiple(*csF, l)
        sF[0] = net_d.forward_multiple(*sF, l)
        if t:
            csF[0] = transform(csF[0], sF[0], args.alpha, args.cuda)

    # save output
    utils.save_image(csF[0], args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PhotoNet')

    parser.add_argument('--content', type=str, default='content/1.jpg')
    parser.add_argument('--style', type=str, default='style/1.jpg')
    parser.add_argument('--output', type=str, default='results/1.jpg')
    parser.add_argument('--max_size', type=int, default=640)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--model', type=str, default='PhotoNet', choices=['PhotoNet', 'PhotoNAS'])
    parser.add_argument('--encoder_path', type=str, default='ckpoint/vgg_normalised_conv5_1.pth')
    parser.add_argument('--decoder_path', type=str, default='ckpoint/PhotoNet_decoder_ckpoint_epoch_2.pth.tar')

    args = parser.parse_args()
    stylize(args)
