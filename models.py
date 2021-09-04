import torch
import torch.nn as nn


class VGGEncoder(nn.Module):
    def __init__(self):
        super(VGGEncoder, self).__init__()

        # 224 x 224
        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)

        self.pad1_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        # 226 x 226
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
        self.relu1_1 = nn.ReLU(inplace=True)
        # 224 x 224

        self.pad1_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu1_2 = nn.ReLU(inplace=True)
        # 224 x 224

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 112 x 112

        self.pad2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu2_1 = nn.ReLU(inplace=True)
        # 112 x 112

        self.pad2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu2_2 = nn.ReLU(inplace=True)
        # 112 x 112

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 56 x 56

        self.pad3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.relu3_1 = nn.ReLU(inplace=True)
        # 56 x 56

        self.pad3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_2 = nn.ReLU(inplace=True)
        # 56 x 56

        self.pad3_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_3 = nn.ReLU(inplace=True)
        # 56 x 56

        self.pad3_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_4 = nn.ReLU(inplace=True)
        # 56 x 56

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 28 x 28

        self.pad4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 0)
        self.relu4_1 = nn.ReLU(inplace=True)
        # 28 x 28

        self.pad4_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu4_2 = nn.ReLU(inplace=True)
        # 28 x 28

        self.pad4_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu4_3 = nn.ReLU(inplace=True)
        # 28 x 28

        self.pad4_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4_4 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu4_4 = nn.ReLU(inplace=True)
        # 28 x 28

        self.maxPool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 14 x 14

        self.pad5_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu5_1 = nn.ReLU(inplace=True)
        # 14 x 14

    def forward(self, x):
        out = self.conv0(x)

        out = self.pad1_1(out)
        out = self.conv1_1(out)
        out = self.relu1_1(out)

        out1 = out

        out = self.pad1_2(out)
        out = self.conv1_2(out)
        pool1 = self.relu1_2(out)

        out, _ = self.maxpool1(pool1)

        out = self.pad2_1(out)
        out = self.conv2_1(out)
        out = self.relu2_1(out)

        out2 = out

        out = self.pad2_2(out)
        out = self.conv2_2(out)
        pool2 = self.relu2_2(out)

        out, _ = self.maxpool2(pool2)

        out = self.pad3_1(out)
        out = self.conv3_1(out)
        out = self.relu3_1(out)

        out3 = out

        out = self.pad3_2(out)
        out = self.conv3_2(out)
        out = self.relu3_2(out)

        out = self.pad3_3(out)
        out = self.conv3_3(out)
        out = self.relu3_3(out)

        out = self.pad3_4(out)
        out = self.conv3_4(out)
        pool3 = self.relu3_4(out)

        out, _ = self.maxpool3(pool3)

        out = self.pad4_1(out)
        out = self.conv4_1(out)
        out = self.relu4_1(out)

        out4 = out

        out = self.pad4_2(out)
        out = self.conv4_2(out)
        out = self.relu4_2(out)

        out = self.pad4_3(out)
        out = self.conv4_3(out)
        out = self.relu4_3(out)

        out = self.pad4_4(out)
        out = self.conv4_4(out)
        pool4 = self.relu4_4(out)

        out, _ = self.maxPool4(pool4)

        out = self.pad5_1(out)
        out = self.conv5_1(out)
        out = self.relu5_1(out)
        
        return out, out4, out3, out2, out1


class PhotoNetDecoder(nn.Module):
    def __init__(self):
        super(PhotoNetDecoder, self).__init__()

        self.IN4 = nn.InstanceNorm2d(512)
        self.IN3 = nn.InstanceNorm2d(256)
        self.IN2 = nn.InstanceNorm2d(128)
        self.IN1 = nn.InstanceNorm2d(64)
        self.maxPool_mid4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxPool_mid3 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.maxPool_mid2 = nn.MaxPool2d(kernel_size=8, stride=8)
        self.maxPool_mid1 = nn.MaxPool2d(kernel_size=16, stride=16)

        self.conv_pyramid5 = nn.Conv2d(1472, 512, 1, 1, 0)

        self.pad5_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu5_1 = nn.ReLU(inplace=True)
        # 14 x 14

        self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)
        # 28 x 28

        self.pad4_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4_4 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu4_4 = nn.ReLU(inplace=True)
        # 28 x 28

        self.IN4_1 = nn.InstanceNorm2d(512)
        self.pad4_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4_3 = nn.Conv2d(1024, 512, 3, 1, 0)
        self.relu4_3 = nn.ReLU(inplace=True)
        # 28 x 28

        self.pad4_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu4_2 = nn.ReLU(inplace=True)
        # 28 x 28

        self.pad4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)
        self.relu4_1 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56

        self.pad3_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_4 = nn.ReLU(inplace=True)
        # 56 x 56

        self.IN3_1 = nn.InstanceNorm2d(256)
        self.pad3_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_3 = nn.Conv2d(512, 256, 3, 1, 0)
        self.relu3_3 = nn.ReLU(inplace=True)
        # 56 x 56

        self.pad3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_2 = nn.ReLU(inplace=True)
        # 56 x 56

        self.pad3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)
        self.relu3_1 = nn.ReLU(inplace=True)
        # 56 x 56

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 x 112

        self.pad2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu2_2 = nn.ReLU(inplace=True)
        # 112 x 112

        self.IN2_1 = nn.InstanceNorm2d(128)
        self.pad2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2_1 = nn.Conv2d(256, 64, 3, 1, 0)
        self.relu2_1 = nn.ReLU(inplace=True)
        # 112 x 112

        self.unpool1 = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        self.pad1_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu1_2 = nn.ReLU(inplace=True)
        # 224 x 224

        self.IN1_1 = nn.InstanceNorm2d(64)
        self.pad1_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv1_1 = nn.Conv2d(128, 3, 3, 1, 0)

    def forward(self, x, skip4, skip3, skip2, skip1):
        skip4_in = self.IN4(skip4)
        mid4 = self.maxPool_mid4(skip4_in)
        skip3_in = self.IN3(skip3)
        mid3 = self.maxPool_mid3(skip3_in)
        skip2_in = self.IN2(skip2)
        mid2 = self.maxPool_mid2(skip2_in)
        skip1_in = self.IN1(skip1)
        mid1 = self.maxPool_mid1(skip1_in)

        out = x
        out = torch.cat((out, mid4, mid3, mid2, mid1), 1)
        out = self.conv_pyramid5(out)
        # 14 x 14

        out = self.pad5_1(out)
        out = self.conv5_1(out)
        out = self.relu5_1(out)
        out = self.unpool4(out)
        # 28 x 28

        out = self.pad4_4(out)
        out = self.conv4_4(out)
        out = self.relu4_4(out)

        skip4 = self.IN4_1(skip4)
        out = torch.cat((out, skip4), 1)
        out = self.pad4_3(out)
        out = self.conv4_3(out)
        out = self.relu4_3(out)

        out = self.pad4_2(out)
        out = self.conv4_2(out)
        out = self.relu4_2(out)

        out = self.pad4_1(out)
        out = self.conv4_1(out)
        out = self.relu4_1(out)
        out = self.unpool3(out)
        # 56 x 56

        out = self.pad3_4(out)
        out = self.conv3_4(out)
        out = self.relu3_4(out)

        skip3 = self.IN3_1(skip3)
        out = torch.cat((out, skip3), 1)
        out = self.pad3_3(out)
        out = self.conv3_3(out)
        out = self.relu3_3(out)

        out = self.pad3_2(out)
        out = self.conv3_2(out)
        out = self.relu3_2(out)

        out = self.pad3_1(out)
        out = self.conv3_1(out)
        out = self.relu3_1(out)
        out = self.unpool2(out)
        # 112 x 112

        out = self.pad2_2(out)
        out = self.conv2_2(out)
        out = self.relu2_2(out)

        skip2 = self.IN2_1(skip2)
        out = torch.cat((out, skip2), 1)
        out = self.pad2_1(out)
        out = self.conv2_1(out)
        out = self.relu2_1(out)
        out = self.unpool1(out)

        out = self.pad1_2(out)
        out = self.conv1_2(out)
        out = self.relu1_2(out)

        skip1 = self.IN1_1(skip1)
        out = torch.cat((out, skip1), 1)
        out = self.pad1_1(out)
        out = self.conv1_1(out)
        # 224 x 224

        return out

    def forward_multiple(self, x, skip4, skip3, skip2, skip1, layer):
        out = x

        if layer == 'pyramid':
            skip4_in = self.IN4(skip4)
            mid4 = self.maxPool_mid4(skip4_in)
            skip3_in = self.IN3(skip3)
            mid3 = self.maxPool_mid3(skip3_in)
            skip2_in = self.IN2(skip2)
            mid2 = self.maxPool_mid2(skip2_in)
            skip1_in = self.IN1(skip1)
            mid1 = self.maxPool_mid1(skip1_in)

            out = torch.cat((out, mid4, mid3, mid2, mid1), 1)
            out = self.conv_pyramid5(out)
            # 14 x 14

        elif layer == 'conv5':
            out = self.pad5_1(out)
            out = self.conv5_1(out)
            out = self.relu5_1(out)
            out = self.unpool4(out)
            # 28 x 28

            out = self.pad4_4(out)
            out = self.conv4_4(out)
            out = self.relu4_4(out)

        elif layer == 'conv4':
            skip4 = self.IN4_1(skip4)
            out = torch.cat((out, skip4), 1)
            out = self.pad4_3(out)
            out = self.conv4_3(out)
            out = self.relu4_3(out)

            out = self.pad4_2(out)
            out = self.conv4_2(out)
            out = self.relu4_2(out)

            out = self.pad4_1(out)
            out = self.conv4_1(out)
            out = self.relu4_1(out)
            out = self.unpool3(out)
            # 56 x 56

            out = self.pad3_4(out)
            out = self.conv3_4(out)
            out = self.relu3_4(out)

        elif layer == 'conv3':
            skip3 = self.IN3_1(skip3)
            out = torch.cat((out, skip3), 1)
            out = self.pad3_3(out)
            out = self.conv3_3(out)
            out = self.relu3_3(out)

            out = self.pad3_2(out)
            out = self.conv3_2(out)
            out = self.relu3_2(out)

            out = self.pad3_1(out)
            out = self.conv3_1(out)
            out = self.relu3_1(out)
            out = self.unpool2(out)
            # 112 x 112

            out = self.pad2_2(out)
            out = self.conv2_2(out)
            out = self.relu2_2(out)

        elif layer == 'conv2':
            skip2 = self.IN2_1(skip2)
            out = torch.cat((out, skip2), 1)
            out = self.pad2_1(out)
            out = self.conv2_1(out)
            out = self.relu2_1(out)
            out = self.unpool1(out)

            out = self.pad1_2(out)
            out = self.conv1_2(out)
            out = self.relu1_2(out)

        elif layer == 'conv1':
            skip1 = self.IN1_1(skip1)
            out = torch.cat((out, skip1), 1)
            out = self.pad1_1(out)
            out = self.conv1_1(out)
            # 224 x 224

        return out


class NASDecoder(nn.Module):
    def __init__(self):
        super(NASDecoder, self).__init__()

        # d0_control: transfer, skip1, skip2, skip3, skip4
        # d1_control: transfer, conv15, conv16
        # d2_control: transfer, IN, skip+conv17, conv18, conv19, conv20, transfer
        # d3_control: transfer, IN, skip+conv21, conv22, conv23, conv24, transfer
        # d4_control: transfer, IN, skip+conv25, conv26, transfer
        # d5_control: transfer, IN, skip+conv27, transfer
        # decoder

        self.IN4 = nn.InstanceNorm2d(512)
        self.IN3 = nn.InstanceNorm2d(256)
        self.IN2 = nn.InstanceNorm2d(128)
        self.IN1 = nn.InstanceNorm2d(64)
        self.maxPool_mid4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.maxPool_mid3 = nn.MaxPool2d(kernel_size=4, stride=4, return_indices=True)
        self.maxPool_mid2 = nn.MaxPool2d(kernel_size=8, stride=8, return_indices=True)
        self.maxPool_mid1 = nn.MaxPool2d(kernel_size=16, stride=16, return_indices=True)

        self.conv_pyramid5 = nn.Conv2d(1152, 512, 1, 1, 0)

        self.pad5_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu5_1 = nn.ReLU(inplace=True)
        # 14 x 14

        self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)
        # 28 x 28

        self.pad4_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4_4 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu4_4 = nn.ReLU(inplace=True)
        # 28 x 28

        self.IN4_1 = nn.InstanceNorm2d(512)
        self.pad4_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4_3 = nn.Conv2d(1024, 512, 3, 1, 0)
        self.relu4_3 = nn.ReLU(inplace=True)
        # 28 x 28

        self.pad4_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu4_2 = nn.ReLU(inplace=True)
        # 28 x 28

        self.pad4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4_1 = nn.Conv2d(512, 256, 1, 1, 0)
        self.relu4_1 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56

        self.pad3_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_4 = nn.ReLU(inplace=True)
        # 56 x 56

        self.IN3_1 = nn.InstanceNorm2d(256)
        self.pad3_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_3 = nn.ReLU(inplace=True)
        # 56 x 56

        self.pad3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_2 = nn.ReLU(inplace=True)
        # 56 x 56

        self.pad3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_1 = nn.Conv2d(256, 128, 1, 1, 0)
        self.relu3_1 = nn.ReLU(inplace=True)
        # 56 x 56

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 x 112

        self.pad2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu2_2 = nn.ReLU(inplace=True)
        # 112 x 112

        self.IN2_1 = nn.InstanceNorm2d(128)
        self.pad2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)
        self.relu2_1 = nn.ReLU(inplace=True)
        # 112 x 112

        self.unpool1 = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        self.pad1_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu1_2 = nn.ReLU(inplace=True)
        # 224 x 224

        self.IN1_1 = nn.InstanceNorm2d(64)
        self.pad1_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv1_1 = nn.Conv2d(128, 3, 3, 1, 0)

    def forward(self, x, skip4, skip3, skip2, skip1):
        skip4_in = self.IN4(skip4)
        mid4, _ = self.maxPool_mid4(skip4_in)
        skip3_in = self.IN3(skip3)
        mid3, _ = self.maxPool_mid3(skip3_in)
        skip2_in = self.IN2(skip2)
        mid2, _ = self.maxPool_mid2(skip2_in)
        skip1_in = self.IN1(skip1)
        mid1, _ = self.maxPool_mid1(skip1_in)

        out = x
        out = torch.cat((out, mid4, mid2), 1)
        out = self.conv_pyramid5(out)
        # 14 x 14

        # out = self.pad5_1(out)
        # out = self.conv5_1(out)
        # out = self.relu5_1(out)
        out = self.unpool4(out)
        # 28 x 28

        # out = self.pad4_4(out)
        # out = self.conv4_4(out)
        # out = self.relu4_4(out)

        # skip4 = self.IN4_1(skip4)
        out = torch.cat((out, skip4), 1)
        out = self.pad4_3(out)
        out = self.conv4_3(out)
        out = self.relu4_3(out)

        # out = self.pad4_2(out)
        # out = self.conv4_2(out)
        # out = self.relu4_2(out)

        # out = self.pad4_1(out)
        out = self.conv4_1(out)
        out = self.relu4_1(out)
        out = self.unpool3(out)
        # 56 x 56

        # out = self.pad3_4(out)
        # out = self.conv3_4(out)
        # out = self.relu3_4(out)

        # skip3 = self.IN3_1(skip3)
        # out = torch.cat((out, skip3), 1)
        out = self.pad3_3(out)
        out = self.conv3_3(out)
        out = self.relu3_3(out)

        # out = self.pad3_2(out)
        # out = self.conv3_2(out)
        # out = self.relu3_2(out)

        # out = self.pad3_1(out)
        out = self.conv3_1(out)
        out = self.relu3_1(out)
        out = self.unpool2(out)
        # 112 x 112

        # out = self.pad2_2(out)
        # out = self.conv2_2(out)
        # out = self.relu2_2(out)

        # skip2 = self.IN2_1(skip2)
        # out = torch.cat((out, skip2), 1)
        out = self.pad2_1(out)
        out = self.conv2_1(out)
        out = self.relu2_1(out)
        out = self.unpool1(out)

        # out = self.pad1_2(out)
        # out = self.conv1_2(out)
        # out = self.relu1_2(out)

        skip1 = self.IN1_1(skip1)
        out = torch.cat((out, skip1), 1)
        out = self.pad1_1(out)
        out = self.conv1_1(out)
        # 224 x 224

        return out

    def forward_multiple(self, x, skip4, skip3, skip2, skip1, layer):
        out = x

        if layer == 'pyramid':
            skip4_in = self.IN4(skip4)
            mid4, _ = self.maxPool_mid4(skip4_in)
            skip3_in = self.IN3(skip3)
            mid3, _ = self.maxPool_mid3(skip3_in)
            skip2_in = self.IN2(skip2)
            mid2, _ = self.maxPool_mid2(skip2_in)
            skip1_in = self.IN1(skip1)
            mid1, _ = self.maxPool_mid1(skip1_in)

            out = torch.cat((out, mid4, mid2), 1)
            out = self.conv_pyramid5(out)
            # 14 x 14

        elif layer == 'conv5':
            # out = self.pad5_1(out)
            # out = self.conv5_1(out)
            # out = self.relu5_1(out)
            out = self.unpool4(out)
            # 28 x 28

            # out = self.pad4_4(out)
            # out = self.conv4_4(out)
            # out = self.relu4_4(out)

        elif layer == 'conv4':
            # skip4 = self.IN4_1(skip4)
            out = torch.cat((out, skip4), 1)
            out = self.pad4_3(out)
            out = self.conv4_3(out)
            out = self.relu4_3(out)

            # out = self.pad4_2(out)
            # out = self.conv4_2(out)
            # out = self.relu4_2(out)

            # out = self.pad4_1(out)
            out = self.conv4_1(out)
            out = self.relu4_1(out)
            out = self.unpool3(out)
            # 56 x 56

            # out = self.pad3_4(out)
            # out = self.conv3_4(out)
            # out = self.relu3_4(out)

        elif layer == 'conv3':
            # skip3 = self.IN3_1(skip3)
            # out = torch.cat((out, skip3), 1)
            out = self.pad3_3(out)
            out = self.conv3_3(out)
            out = self.relu3_3(out)

            # out = self.pad3_2(out)
            # out = self.conv3_2(out)
            # out = self.relu3_2(out)

            # out = self.pad3_1(out)
            out = self.conv3_1(out)
            out = self.relu3_1(out)
            out = self.unpool2(out)
            # 112 x 112

            # out = self.pad2_2(out)
            # out = self.conv2_2(out)
            # out = self.relu2_2(out)

        elif layer == 'conv2':
            # skip2 = self.IN2_1(skip2)
            # out = torch.cat((out, skip2), 1)
            out = self.pad2_1(out)
            out = self.conv2_1(out)
            out = self.relu2_1(out)
            out = self.unpool1(out)

            # out = self.pad1_2(out)
            # out = self.conv1_2(out)
            # out = self.relu1_2(out)

        elif layer == 'conv1':
            skip1 = self.IN1_1(skip1)
            out = torch.cat((out, skip1), 1)
            out = self.pad1_1(out)
            out = self.conv1_1(out)
            # 224 x 224

        return out
