import torch
import torch.nn as nn
import modules as mo
from timm.models.layers import DropPath

class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)

class StarBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x

class ResidualDenseBlock_out(nn.Module): 
    def __init__(self, input, output, bias=True):
        super(ResidualDenseBlock_out, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input + 4 * 32, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        # initialization
        mo.initialize_weights([self.conv5], 0.)

    def forward(self, x):
        # print("Input shape:", x.shape)
        x1 = self.lrelu(self.conv1(x))
        # print("After conv1 shape:", x1.shape)
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        # print("After conv2 shape:", x2.shape)
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        # print("After conv3 shape:", x3.shape)
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        # print("After conv4 shape:", x4.shape)
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # print("After conv5 (final output) shape:", x5.shape)
        return x5

class StarDenseBlock(nn.Module): 
    def __init__(self, input, output, bias=True):
        super(StarDenseBlock, self).__init__()
        dim2 = 2 * input
        dim3 = 4 * input
        dim4 = 8 * input
        dim5 = 16 * input
        self.block1_1 = StarBlock(dim=input, mlp_ratio=2)
        self.block1_2 = StarBlock(dim=input, mlp_ratio=2)
        self.block2_1 = StarBlock(dim=dim2, mlp_ratio=2)
        self.block2_2 = StarBlock(dim=dim2, mlp_ratio=2)
        self.block3_1 = StarBlock(dim=dim3, mlp_ratio=2)
        self.block3_2 = StarBlock(dim=dim3, mlp_ratio=2)
        self.block3_3 = StarBlock(dim=dim3, mlp_ratio=2)
        self.block3_4 = StarBlock(dim=dim3, mlp_ratio=2)
        self.block3_5 = StarBlock(dim=dim3, mlp_ratio=2)
        self.block3_6 = StarBlock(dim=dim3, mlp_ratio=2)
        self.block4_1 = StarBlock(dim=dim4, mlp_ratio=2)
        self.block4_2 = StarBlock(dim=dim4, mlp_ratio=2)
        self.block4_3 = StarBlock(dim=dim4, mlp_ratio=2)
        self.conv5 = ConvBN(dim5, output)
        self.lrelu = nn.LeakyReLU(inplace=True)
        mo.initialize_weights([self.conv5], 0.)

    def forward(self, x):
        # print("Input shape:", x.shape)
        x1 = self.block1_1(x)
        x1 = self.block1_2(x1)
        # print("After conv1 shape:", x1.shape)
        # 使用 Block 进行特征提取
        x2 = self.block2_1(torch.cat((x, x1), 1))
        x2 = self.block2_2(x2)
        # print("After block2 shape:", x2.shape)
        x3 = self.block3_1(torch.cat((x, x1, x2), 1))
        x3 = self.block3_2(x3)
        x3 = self.block3_3(x3)
        x3 = self.block3_4(x3)
        x3 = self.block3_5(x3)
        x3 = self.block3_6(x3)
        # print("After block3 shape:", x3.shape)
        x4 = self.block4_1(torch.cat((x, x1, x2, x3), 1))
        x4 = self.block4_2(x4)
        x4 = self.block4_3(x4)
        # print("After block4 shape:", x4.shape)
        # 输出层
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # print("After conv5 (final output) shape:", x5.shape)
        return x5

class StarDenseBlock2(nn.Module):  
    def __init__(self, input, output, bias=True):
        super(StarDenseBlock2, self).__init__()  # 修改 super() 调用以匹配新名称
        self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        # 定义每个 Block 的通道数，保持一致，防止通道数增长
        block_channels = 32  # 可以根据需要调整
        self.block2 = StarBlock(dim=input + 32, mlp_ratio=2)
        self.block3 = StarBlock(dim=input + 2 * 32, mlp_ratio=2)
        # 使用 1x1 卷积进行通道数压缩
        self.compress_conv2 = nn.Conv2d(input + 32, block_channels, kernel_size=1, bias=bias)
        self.compress_conv3 = nn.Conv2d(input + 2 * 32, block_channels, kernel_size=1, bias=bias)
        self.conv4 = nn.Conv2d(input + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input + 4 * 32, output, 3, 1, 1, bias=bias)
        mo.initialize_weights([self.compress_conv2, self.compress_conv3, self.conv5], scale=1)
        mo.initialize_weights([self.conv5], 0.)

    def forward(self, x):
        # print("Input shape:", x.shape)
        x1 = self.lrelu(self.conv1(x))
        # print("After conv1 shape:", x1.shape)
        # 使用 Block 进行特征提取
        x2 = self.block2(torch.cat((x, x1), 1))
        x2 = self.lrelu(self.compress_conv2(x2))
        # print("After block2 shape:", x2.shape)
        x3 = self.block3(torch.cat((x, x1, x2), 1))
        x3 = self.lrelu(self.compress_conv3(x3))
        # print("After block3 shape:", x3.shape)
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        # print("After block4 shape:", x4.shape)
        # 输出层
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # print("After conv5 (final output) shape:", x5.shape)
        return x5

class StarDenseBlock_s(nn.Module):
    def __init__(self, input, output, bias=True):
        super(StarDenseBlock_s, self).__init__()
        self.block = StarBlock(dim=input, mlp_ratio=2)  
        self.conv2 = nn.Conv2d(2 * input, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(2 * input + 32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(2 * input + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(2 * input + 3 * 32, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        # 初始化权重
        mo.initialize_weights([self.conv5], 0.)

    def forward(self, x):
        # print("Input shape:", x.shape)
        x1 = self.block(x) 
        # print("After block (replaces conv1) shape:", x1.shape)
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        # print("After conv2 shape:", x2.shape)
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        # print("After conv3 shape:", x3.shape)
        x4 = self.sigmoid(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        # print("After conv4 shape:", x4.shape)
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # print("After conv5 (final output) shape:", x5.shape)
        return x5


if __name__ == '__main__':
    # 假设输入图像有 3 个通道，尺寸为 64x64
    input_tensor = torch.randn(1, 3, 64, 64)  # batch size 为 1
    model_res = ResidualDenseBlock_out(input=3,output=3)
    model_star = StarDenseBlock(input=3,output=3)
    model_star2 = StarDenseBlock2(input=3,output=3)
    model_stars = StarDenseBlock_s(input=3,output=3)
    model_Block = StarBlock(dim=3)
    # 前向传播，打印每层输出的形状
    output = model_star(input_tensor)
