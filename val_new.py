import torch 
import os
from model import Model
import config as c
import modules as mo
from tqdm import tqdm
import pyiqa  # 导入pyiqa库

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r''

# 加载模型
def load_model():
    net = Model()
    net = net.to(device)
    net = torch.nn.DataParallel(net, device_ids=c.device_ids)
    checkpoint = torch.load(MODEL_PATH)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    return net

def contains_nan_in_min_max(steg, secret_rev):
    return torch.isnan(torch.min(steg)).any() or torch.isnan(torch.max(steg)).any() or \
           torch.isnan(torch.min(secret_rev)).any() or torch.isnan(torch.max(secret_rev)).any()

# 生成并直接计算指标
def validate_and_evaluate():
    net = load_model()
    dwt = mo.DWT()
    iwt = mo.IWT()
    
    # 创建指标
    iqa_metric_psnr = pyiqa.create_metric('psnr', device=device)
    iqa_metric_psnry = pyiqa.create_metric('psnry', device=device)
    iqa_metric_ssim = pyiqa.create_metric('ssim', device=device)
    iqa_metric_niqe = pyiqa.create_metric('niqe', device=device)

    # 初始化累加器
    total_psnr_cover_steg = 0
    total_psnr_secret_secret_rev = 0
    total_psnry_cover_steg = 0
    total_psnry_secret_secret_rev = 0
    total_ssim_cover_steg = 0
    total_ssim_secret_secret_rev = 0
    total_niqe_cover = 0
    total_niqe_steg = 0
    num_images = 0
    total_mse_cover_steg = 0
    total_mse_secret_secret_rev = 0
    total_rmse_cover_steg = 0
    total_rmse_secret_secret_rev = 0
    total_mae_cover_steg = 0
    total_mae_secret_secret_rev = 0

    with torch.no_grad():  # 禁用梯度计算
        with tqdm(mo.testloader, desc="Validation") as v:
            for data in v:
                data = data.to(device)
                cover = data[data.shape[0] // 2:]
                secret = data[:data.shape[0] // 2]
                
                # 生成stego和secret_rev图像
                cover_input = dwt(cover)
                secret_input = dwt(secret)
                input_img = torch.cat((cover_input, secret_input), 1)
                
                # 隐藏过程
                output = net(input_img)
                output_steg = output.narrow(1, 0, 4 * c.channels_in)
                steg = iwt(output_steg)
                
                # 添加噪声并进行恢复
                output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
                output_z = torch.randn_like(output_z).to(device)
                output_rev = torch.cat((output_steg, output_z), 1)
                output_image = net(output_rev, True)
                secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
                secret_rev = iwt(secret_rev)

                # 将图像裁剪到[0,1]范围，并转换为float类型
                cover = cover.clamp(0,1).float().to(device)
                secret = secret.clamp(0,1).float().to(device)
                steg = steg.clamp(0,1).float().to(device)
                secret_rev = secret_rev.clamp(0,1).float().to(device)

                if contains_nan_in_min_max(steg, secret_rev):
                    print('pass')
                    continue  # 跳过包含NaN的图像对

                # 计算指标
                psnr_cover_steg = iqa_metric_psnr(cover, steg)
                psnr_secret_secret_rev = iqa_metric_psnr(secret, secret_rev)
                psnry_cover_steg = iqa_metric_psnry(cover, steg)
                psnry_secret_secret_rev = iqa_metric_psnry(secret, secret_rev)
                ssim_cover_steg = iqa_metric_ssim(cover, steg)
                ssim_secret_secret_rev = iqa_metric_ssim(secret, secret_rev)
                niqe_cover = iqa_metric_niqe(cover)
                niqe_steg = iqa_metric_niqe(steg)
                mse_cover_steg = torch.mean((cover - steg) ** 2) * (255 ** 2)
                mse_secret_secret_rev = torch.mean((secret - secret_rev) ** 2) * (255 ** 2)
                rmse_cover_steg = torch.sqrt(mse_cover_steg)
                rmse_secret_secret_rev = torch.sqrt(mse_secret_secret_rev)
                mae_cover_steg = torch.mean(torch.abs(cover - steg)) * 255
                mae_secret_secret_rev = torch.mean(torch.abs(secret - secret_rev)) * 255

                # 累加指标
                total_psnr_cover_steg += psnr_cover_steg.sum().item()
                total_psnr_secret_secret_rev += psnr_secret_secret_rev.sum().item()
                total_psnry_cover_steg += psnry_cover_steg.sum().item()
                total_psnry_secret_secret_rev += psnry_secret_secret_rev.sum().item()
                total_ssim_cover_steg += ssim_cover_steg.sum().item()
                total_ssim_secret_secret_rev += ssim_secret_secret_rev.sum().item()
                total_niqe_cover += niqe_cover.sum().item()
                total_niqe_steg += niqe_steg.sum().item()
                num_images += cover.shape[0]
                total_mse_cover_steg += mse_cover_steg.item()
                total_mse_secret_secret_rev += mse_secret_secret_rev.item()
                total_rmse_cover_steg += rmse_cover_steg.item()
                total_rmse_secret_secret_rev += rmse_secret_secret_rev.item()
                total_mae_cover_steg += mae_cover_steg.item()
                total_mae_secret_secret_rev += mae_secret_secret_rev.item()

    # 计算平均指标
    avg_psnr_cover_steg = total_psnr_cover_steg / num_images
    avg_psnr_secret_secret_rev = total_psnr_secret_secret_rev / num_images
    avg_psnry_cover_steg = total_psnry_cover_steg / num_images
    avg_psnry_secret_secret_rev = total_psnry_secret_secret_rev / num_images
    avg_ssim_cover_steg = total_ssim_cover_steg / num_images
    avg_ssim_secret_secret_rev = total_ssim_secret_secret_rev / num_images
    avg_niqe_cover = total_niqe_cover / num_images
    avg_niqe_steg = total_niqe_steg / num_images
    avg_mse_cover_steg = total_mse_cover_steg / num_images
    avg_mse_secret_secret_rev = total_mse_secret_secret_rev / num_images
    avg_rmse_cover_steg = total_rmse_cover_steg / num_images
    avg_rmse_secret_secret_rev = total_rmse_secret_secret_rev / num_images
    avg_mae_cover_steg = total_mae_cover_steg / num_images
    avg_mae_secret_secret_rev = total_mae_secret_secret_rev / num_images

    # 输出结果
    print("秘密图像恢复PSNR:", avg_psnr_secret_secret_rev)
    print("秘密图像恢复Y-PSNR:", avg_psnry_secret_secret_rev)
    print("秘密图像恢复SSIM:", avg_ssim_secret_secret_rev)
    print("秘密图像恢复MSE (0-255):", avg_mse_secret_secret_rev)
    print("秘密图像恢复RMSE (0-255):", avg_rmse_secret_secret_rev)
    print("秘密图像恢复MAE (0-255):", avg_mae_secret_secret_rev)

    print("含密图像PSNR:", avg_psnr_cover_steg)
    print("含密图像Y-PSNR:", avg_psnry_cover_steg)
    print("含密图像SSIM:", avg_ssim_cover_steg)
    print("含密图像MSE (0-255):", avg_mse_cover_steg)
    print("含密图像RMSE (0-255):", avg_rmse_cover_steg)
    print("含密图像MAE (0-255):", avg_mae_cover_steg)
    print("原始图像NIQE:", avg_niqe_cover)
    print("含密图像NIQE:", avg_niqe_steg)
if __name__ == "__main__":
    validate_and_evaluate()
