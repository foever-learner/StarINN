import torch
import torch.nn as nn
from invblock import INV_block

class StarINN(nn.Module):
    def __init__(self):
        super(StarINN, self).__init__()
        self.inv1 = INV_block()
        self.inv2 = INV_block()
        self.inv3 = INV_block()
        self.inv4 = INV_block()
        self.inv5 = INV_block()
        self.inv6 = INV_block()
        self.inv7 = INV_block()
        self.inv8 = INV_block()
        self.inv9 = INV_block()
        self.inv10 = INV_block()
        self.inv11 = INV_block()
        self.inv12 = INV_block()


    def forward(self, x, rev=False):
        if not rev:
            # 前向传播：顺序通过12个模块
            out = self.inv1(x)
            out = self.inv2(out)
            out = self.inv3(out)
            out = self.inv4(out)
            out = self.inv5(out)
            out = self.inv6(out)
            out = self.inv7(out)
            out = self.inv8(out)
            out = self.inv9(out)
            out = self.inv10(out)
            out = self.inv11(out)
            out = self.inv12(out)  
        else:
            # 反向传播：逆序通过12个模块
            out = self.inv12(x, rev=True)  
            out = self.inv11(out, rev=True)
            out = self.inv10(out, rev=True)
            out = self.inv9(out, rev=True)
            out = self.inv8(out, rev=True)
            out = self.inv7(out, rev=True)
            out = self.inv6(out, rev=True)
            out = self.inv5(out, rev=True)
            out = self.inv4(out, rev=True)
            out = self.inv3(out, rev=True)
            out = self.inv2(out, rev=True)
            out = self.inv1(out, rev=True)  
        return out


if __name__ == '__main__' :
    net = StarstegoNet()
    # 设置模型为评估模式（避免 Dropout 等不确定性）
    net.eval()
    # 定义输入数据，假设是 64x64 尺寸的彩色图像
    input_data = torch.randn(1, 24, 64, 64).cuda()  # 随机输入，放在GPU上
    # 将模型放到GPU上（如果可用）
    net = net.cuda()

    # 定义一个函数来测试和打印输出范围
    def test_output_range(model, input_data, rev=False):
        with torch.no_grad():  # 禁用梯度计算，加快测试速度
            output = model(input_data, rev=rev)
            output_min = output.min().item()
            output_max = output.max().item()
            print(f"Output range for rev={rev}: [{output_min}, {output_max}]")

    # 测试 StarstegoNet
    print("Testing StarstegoNet:")
    test_output_range(net, input_data, rev=False)
    test_output_range(net, input_data, rev=True)
