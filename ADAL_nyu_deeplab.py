from __future__ import print_function
import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
import torchvision
import network
from utils import soft_cross_entropy, kldiv
# from utils.visualizer import VisdomPlotter
from utils.misc import pack_images, denormalize
from dataloader import get_dataloader
from utils.stream_metrics import StreamSegMetrics
import random, os
import numpy as np
from PIL import Image
from semantic_dist_estimator import semantic_dist_estimator

# vp = VisdomPlotter('15550', env='DFAD-nyuv2')

def generate_random_numbers(batch_size):
    # 生成batch_size个0到999之间（包含0和999）的随机整数
    random_tensor = torch.randint(0, 1000, (batch_size,))
    return random_tensor

# 示例：设定batch_size值
# batch_size = 10
# random_numbers = generate_random_numbers(batch_size)
def generate_cyclic_tensor(batch_size):
    # 使用torch.arange生成一个从0到12的序列
    base_sequence = torch.arange(13)
    # 重复这个序列，直到至少有batch_size个数
    repeated_sequence = base_sequence.repeat((batch_size // 13) + 1)
    # 截取前batch_size个数
    cyclic_tensor = repeated_sequence[:batch_size]
    return cyclic_tensor


def process_features_and_update_estimator(features, logits, estimator):
    """
    处理特征和logits，并更新估计器。

    参数:
    features (torch.Tensor): 特征tensor，形状为 (B, N, Hs, Ws)
    logits (torch.Tensor): logits tensor，形状为 (B, C, H, W)，其中C是类别数
    estimator: 用于更新的估计器对象

    返回:
    None
    """
    B, N, Hs, Ws = features.size()

    # 计算预测
    pred = torch.argmax(logits, dim=1).long()

    # 调整预测的大小以匹配特征大小
    src_mask = F.interpolate(pred.unsqueeze(0).float(), size=(Hs, Ws), mode='nearest').squeeze(0).long()
    src_mask = src_mask.contiguous().view(B * Hs * Ws, )

    # 调整特征的形状
    features = features.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, N)

    # 更新估计器
    estimator.update(features=features.detach().clone(), labels=src_mask)

def sample_difference(teacher_means, teacher_covs, student_means, student_covs):
    # 确保所有输入都是 torch.Tensor
    teacher_means = torch.as_tensor(teacher_means)
    teacher_covs = torch.as_tensor(teacher_covs)
    student_means = torch.as_tensor(student_means)
    student_covs = torch.as_tensor(student_covs)

    # 计算差异分布的均值和方差
    diff_means = teacher_means - student_means
    diff_vars = teacher_covs + student_covs

    # 使用 torch.randn 生成标准正态分布样本，然后调整均值和方差
    epsilon = torch.randn_like(diff_means)
    samples = diff_means + torch.sqrt(diff_vars) * epsilon

    return samples


def save_image(fake, filename='output.png'):
    # 假设图像的 shape 是 (64, 3, 128, 128)
    num_images, channels, height, width = fake.shape
    fake_ = (fake - fake.min()) / (fake.max() - fake.min())
    # 将生成器输出的图像从 Tensor 转换为 NumPy 数组
    images = fake_.detach().cpu().numpy()

    # 归一化张量到 [0, 255] 并转换数据类型为 uint8
    # images = (images * 0.5 + 0.5) * 255
    images = (images) * 255
    images = images.astype(np.uint8)

    # 将 (num_images, channels, height, width) 转换为 (num_images, height, width, channels)
    images = np.transpose(images, (0, 2, 3, 1))

    # 确定拼贴的网格尺寸，例如 8x8
    grid_size = int(np.ceil(np.sqrt(num_images)))

    # 创建一个空白的大图像以保存拼贴后的图像
    grid_height = grid_size * height
    grid_width = grid_size * width
    grid_image = Image.new('RGB', (grid_width, grid_height))

    # 将每一个小图像放置在大图像的相应位置
    for idx, image in enumerate(images):
        row = idx // grid_size
        col = idx % grid_size
        image_pil = Image.fromarray(image)
        grid_image.paste(image_pil, (col * width, row * height))

    # 最后保存这张拼贴后的大图像
    grid_image.save(filename)

def train(args, teacher, student, generator, device, optimizer, epoch,my_hooks,tea_est,stu_est):
    teacher.eval()
    student.train()
    generator.train()
    optimizer_S, optimizer_G = optimizer
    for i in range( args.epoch_itrs ):

        for k in range(5):
            # z = torch.randn( (args.batch_size, args.nz, 1, 1) ).to(device)
            z = generate_cyclic_tensor(args.batch_size).to(device)
            optimizer_S.zero_grad()
            with torch.no_grad():
                fake_fuse = generator(z, stu_est.get_dist_vec(), fuse = True).detach()
            t_logit = teacher(fake_fuse)
            s_logit = student(fake_fuse)
            loss_S = F.l1_loss(s_logit, t_logit.detach()) #(s_logit - t_logit.detach()).abs().mean() #+ kldiv(s_logit, t_logit.detach()) #kldiv(s_logit, t_logit.detach()) 
            loss_S.backward()
            optimizer_S.step()

        z = generate_cyclic_tensor(args.batch_size).to(device)

        optimizer_G.zero_grad()
        generator.train()
        fake = generator(z, None)

        t_logit,tea_fea = teacher(fake,True)
        loss_bn = sum([h.r_feature for h in my_hooks])



        s_logit, stu_fea = student(fake, True)

        # loss_G = -torch.log( F.l1_loss( s_logit, t_logit )+1 ) + 10 * loss_bn
        loss_G = -torch.log( F.l1_loss( s_logit, t_logit )+1 ) + 0.8  * loss_bn

        #loss_G = -F.l1_loss( s_logit, t_logit )

        loss_G.backward()
        optimizer_G.step()

        if i % 10 == 0:
            # 处理教师网络的特征和更新估计器
            process_features_and_update_estimator(tea_fea, t_logit, tea_est)

            # 处理学生网络的特征和更新估计器
            process_features_and_update_estimator(stu_fea, s_logit, stu_est)


        # if is_bias:
        #     dist_vec = torch.zeros(13, 512).cuda()
        # else:
        #     dist_vec = None
        #
        # if is_bias and epoch % 20 == 0:

        if epoch % 20 == 0:
            # dist_vec = torch.zeros(13, 512).cuda()
            teacher_means,teacher_covs = tea_est.get(512)
            student_means,student_covs = stu_est.get(512)

            dist_vec = sample_difference(teacher_means, teacher_covs, student_means, student_covs)
            stu_est.update_dist_vec(dist_vec)

            # # 计算差异
            # mean_diffs = teacher_means - student_means  # 13 * 2048
            # cov_diffs = teacher_covs - student_covs  # 13 * 2048 * 2048
            #
            # # 初始化一个空的向量
            #
            # # 对每个类别
            # for idx in range(13):
            #     # 从均值差异分布采样
            #     mean_sample = torch.randn(512).cuda() * mean_diffs[idx]
            #
            #     # 从协方差差异分布采样
            #     # cov_sample = torch.distributions.MultivariateNormal(
            #     #     loc=torch.zeros(1000),
            #     #     covariance_matrix=cov_diffs[i].unsqueeze(0)).sample()
            #     # cov_sample = cov_sample.cuda()
            #     # 将均值差异和协方差差异合并
            #     dist_vec[idx] =  mean_sample + torch.randn(512).cuda() * cov_diffs[idx]




        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tG_Loss: {:.6f} S_loss: {:.6f}'.format(
                epoch, i, args.epoch_itrs, 100*float(i)/float(args.epoch_itrs), loss_G.item(), loss_S.item()))
            # vp.add_scalar('Loss_S', (epoch-1)*args.epoch_itrs+i, loss_S.item())
            # vp.add_scalar('Loss_G', (epoch-1)*args.epoch_itrs+i, loss_G.item())
    # save_image(fake.detach(), filename=os.path.join("./save_image", "{}_image.png".format(epoch)))
    # save_image(fake_fuse.detach(), filename=os.path.join("./save_image_fuse", "{}_image.png".format(epoch)))

    for h in my_hooks:
        h.update_mmt()

def test(args, student, teacher, generator, device, test_loader,epoch):
    save_img = False
    student.eval()
    generator.eval()
    teacher.eval()

    seg_metrics = StreamSegMetrics(13)
    img_idx = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            # z = torch.randn( (data.shape[0], args.nz, 1, 1), device=data.device, dtype=data.dtype )
            # fake = generator(z)
            output = student(data)

            if save_img and  epoch % 30 == 0:
                if save_img:
                    os.makedirs('results/nyu-DFAD/{}'.format(epoch), exist_ok=True)
                t_out = teacher(data)

                input_imgs = (data*255).clamp(0,255).detach().cpu().numpy().transpose(0,2,3,1).astype('uint8')
                colored_preds = test_loader.dataset.decode_target( output.max(1)[1].detach().cpu().numpy() ).astype('uint8')
                colored_teacher_preds = test_loader.dataset.decode_target( t_out.max(1)[1].detach().cpu().numpy() ).astype('uint8')
                colored_targets = test_loader.dataset.decode_target( target.detach().cpu().numpy() ).astype('uint8')
                for _pred, _img, _target, _tpred in zip( colored_preds, input_imgs, colored_targets, colored_teacher_preds  ):
                    Image.fromarray( _pred ).save('results/nyu-DFAD/{}/{}_pred.png'.format(epoch, img_idx))
                    Image.fromarray( _img ).save('results/nyu-DFAD/{}/{}_img.png'.format(epoch, img_idx))
                    Image.fromarray( _target ).save('results/nyu-DFAD/{}/{}_target.png'.format(epoch, img_idx))
                    Image.fromarray( _tpred ).save('results/nyu-DFAD/{}/{}_teacher.png'.format(epoch, img_idx))
                    img_idx+=1

            # if i==0:
            #     t_out = teacher(data)
            #     t_out_onfake = teacher(fake)
            #     s_out_onfake = student(fake)
                # vp.add_image( 'input', pack_images( ((data+1)/2).clamp(0,1).detach().cpu().numpy() ) )
                # vp.add_image( 'generated', pack_images( ((fake+1)/2).clamp(0,1).detach().cpu().numpy() ) )
                # vp.add_image( 'target', pack_images( test_loader.dataset.decode_target(target.cpu().numpy()), channel_last=True ).astype('uint8') )
                # vp.add_image( 'pred',   pack_images( test_loader.dataset.decode_target(output.max(1)[1].detach().cpu().numpy().astype('uint8')), channel_last=True ).astype('uint8') )
                # vp.add_image( 'teacher',   pack_images( test_loader.dataset.decode_target(t_out.max(1)[1].detach().cpu().numpy().astype('uint8')), channel_last=True ).astype('uint8') )
                # vp.add_image( 'teacher-onfake',   pack_images( test_loader.dataset.decode_target(t_out_onfake.max(1)[1].detach().cpu().numpy().astype('uint8')), channel_last=True ).astype('uint8') )
                # vp.add_image( 'student-onfake',   pack_images( test_loader.dataset.decode_target(s_out_onfake.max(1)[1].detach().cpu().numpy().astype('uint8')), channel_last=True ).astype('uint8') )
            seg_metrics.update(output.max(1)[1].detach().cpu().numpy().astype('uint8'), target.detach().cpu().numpy().astype('uint8'))

    results = seg_metrics.get_results()
    print('\nTest set: Acc= %.6f, mIoU: %.6f\n'%(results['Overall Acc'],results['Mean IoU']))
    return results



def test_teacher(args, student, teacher, generator, device, test_loader):
    student.eval()
    generator.eval()
    teacher.eval()
    if args.save_img:
        os.makedirs('results/nyu-DFAD', exist_ok=True)
    seg_metrics = StreamSegMetrics(13)
    img_idx = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            # z = torch.randn( (data.shape[0], args.nz, 1, 1), device=data.device, dtype=data.dtype )
            # fake = generator(z)
            output = teacher(data)

            if args.save_img:
                t_out = teacher(data)

                input_imgs = (((data+1)/2)*255).clamp(0,255).detach().cpu().numpy().transpose(0,2,3,1).astype('uint8')
                colored_preds = test_loader.dataset.decode_target( output.max(1)[1].detach().cpu().numpy() ).astype('uint8')
                colored_teacher_preds = test_loader.dataset.decode_target( t_out.max(1)[1].detach().cpu().numpy() ).astype('uint8')
                colored_targets = test_loader.dataset.decode_target( target.detach().cpu().numpy() ).astype('uint8')
                for _pred, _img, _target, _tpred in zip( colored_preds, input_imgs, colored_targets, colored_teacher_preds  ):
                    Image.fromarray( _pred ).save('results/nyu-DFAD/%d_pred.png'%img_idx)
                    Image.fromarray( _img ).save('results/nyu-DFAD/%d_img.png'%img_idx)
                    Image.fromarray( _target ).save('results/nyu-DFAD/%d_target.png'%img_idx)
                    Image.fromarray( _tpred ).save('results/nyu-DFAD/%d_teacher.png'%img_idx)
                    img_idx+=1

            # if i==0:
            #     t_out = teacher(data)
            #     t_out_onfake = teacher(fake)
            #     s_out_onfake = student(fake)
                # vp.add_image( 'input', pack_images( ((data+1)/2).clamp(0,1).detach().cpu().numpy() ) )
                # vp.add_image( 'generated', pack_images( ((fake+1)/2).clamp(0,1).detach().cpu().numpy() ) )
                # vp.add_image( 'target', pack_images( test_loader.dataset.decode_target(target.cpu().numpy()), channel_last=True ).astype('uint8') )
                # vp.add_image( 'pred',   pack_images( test_loader.dataset.decode_target(output.max(1)[1].detach().cpu().numpy().astype('uint8')), channel_last=True ).astype('uint8') )
                # vp.add_image( 'teacher',   pack_images( test_loader.dataset.decode_target(t_out.max(1)[1].detach().cpu().numpy().astype('uint8')), channel_last=True ).astype('uint8') )
                # vp.add_image( 'teacher-onfake',   pack_images( test_loader.dataset.decode_target(t_out_onfake.max(1)[1].detach().cpu().numpy().astype('uint8')), channel_last=True ).astype('uint8') )
                # vp.add_image( 'student-onfake',   pack_images( test_loader.dataset.decode_target(s_out_onfake.max(1)[1].detach().cpu().numpy().astype('uint8')), channel_last=True ).astype('uint8') )
            seg_metrics.update(output.max(1)[1].detach().cpu().numpy().astype('uint8'), target.detach().cpu().numpy().astype('uint8'))

    results = seg_metrics.get_results()
    print('\nTest set: Acc= %.6f, mIoU: %.6f\n'%(results['Overall Acc'],results['Mean IoU']))
    return results

class DeepInversionHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module, mmt_rate):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.mmt_rate = mmt_rate
        self.mmt = None
        self.tmp_val = None

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        if self.mmt is None:
            r_feature = torch.norm(module.running_var.data - var, 2) + \
                        torch.norm(module.running_mean.data - mean, 2)
        else:
            mean_mmt, var_mmt = self.mmt
            r_feature = torch.norm(module.running_var.data - (1 - self.mmt_rate) * var - self.mmt_rate * var_mmt, 2) + \
                        torch.norm(module.running_mean.data - (1 - self.mmt_rate) * mean - self.mmt_rate * mean_mmt, 2)

        self.r_feature = r_feature
        self.tmp_val = (mean, var)

    def update_mmt(self):
        mean, var = self.tmp_val
        if self.mmt is None:
            self.mmt = (mean.data, var.data)
        else:
            mean_mmt, var_mmt = self.mmt
            self.mmt = ( self.mmt_rate*mean_mmt+(1-self.mmt_rate)*mean.data,
                         self.mmt_rate*var_mmt+(1-self.mmt_rate)*var.data )

    def remove(self):
        self.hook.remove()

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='DFAD NYUv2')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=9, metavar='N',
                        help='input batch size for testing (default: 9)')
    
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--epoch_itrs', type=int, default=50)
    parser.add_argument('--num_classes', type=int, default=13)
    parser.add_argument('--lr_S', type=float, default=0.1, metavar='LR',
    # parser.add_argument('--lr_S', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--lr_G', type=float, default=1e-3,
    # parser.add_argument('--lr_G', type=float, default=1e-4,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--data_root', type=str, default='data')

    parser.add_argument('--dataset', type=str, default='nyuv2', choices=['nyuv2'],
                        help='dataset name (default: nyuv2)')
    parser.add_argument('--model', type=str, default='deeplabv3_resnet50', choices=['deeplabv3_resnet50'],
                        help='model name (default: deeplabv3_resnet50)')
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--ckpt', type=str, default='checkpoint/teacher/nyuv2-deeplabv3_resnet50-256.pt')
    parser.add_argument('--stu_ckpt', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--nz', type=int, default=256)
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--download', action='store_true', default=False)
    parser.add_argument('--step_size', type=int, default=100)
    parser.add_argument('--scheduler', action='store_true', default=False)
    parser.add_argument('--save_img', action='store_true', default=False)

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    print(args)
    args.scheduler = True
    _, test_loader = get_dataloader(args)
    teacher = network.segmentation.deeplabv3.deeplabv3_resnet50(num_classes=13)
    student = network.segmentation.deeplabv3.deeplabv3_mobilenet(num_classes=13, dropout_p=0.5, pretrained_backbone=False)
    generator = network.gan.GeneratorB(nz=args.nz, nc=3, img_size=128, args=args)
    
    teacher.load_state_dict( torch.load( args.ckpt ) )
    print("Teacher restored from %s"%(args.ckpt))

    if args.stu_ckpt is not None:
        student.load_state_dict( torch.load( args.stu_ckpt ) )
        generator.load_state_dict( torch.load( args.stu_ckpt[:-3]+'-generator.pt' ) )
        print('student loaded from %s'%args.stu_ckpt)
    
    teacher = teacher.to(device)
    student = student.to(device)
    generator = generator.to(device)    

    teacher.eval()

    my_hooks = []

    for m in teacher.modules():
        if isinstance(m, nn.BatchNorm2d):
            my_hooks.append(DeepInversionHook(m, 0.9))

    tea_est = semantic_dist_estimator(feature_num=2048)
    stu_est = semantic_dist_estimator(feature_num=320)
    optimizer_S = optim.SGD( student.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9)
    optimizer_G = optim.Adam( generator.parameters(), lr=args.lr_G)

    if args.scheduler:
        scheduler_S =  optim.lr_scheduler.StepLR(optimizer_S, args.step_size, gamma=0.3)
        scheduler_G =  optim.lr_scheduler.StepLR(optimizer_G, args.step_size, gamma=0.3)
    best_result = 0
    if args.test_only:
        # results = test(args, student, teacher, generator, device, test_loader)
        results = test_teacher(args, student, teacher, generator, device, test_loader)
        return

    for epoch in range(1, args.epochs + 1):
        # Train
        train(args, teacher=teacher, student=student, generator=generator, device=device,
              optimizer=[optimizer_S, optimizer_G], epoch=epoch,my_hooks=my_hooks,
              tea_est=tea_est,stu_est=stu_est)
        # Test
        results = test(args, student, teacher, generator, device, test_loader,epoch)

        if results['Mean IoU']>best_result:
            best_result = results['Mean IoU']
            torch.save(student.state_dict(),"checkpoint/student/%s-%s.pt"%('nyuv2', 'deeplabv3_mobilenet'))
            torch.save(generator.state_dict(),"checkpoint/student/%s-%s-generator.pt"%('nyuv2', 'deeplabv3_mobilenet'))
        # vp.add_scalar('mIoU', epoch, results['Mean IoU'])

        if args.scheduler:
            scheduler_S.step()
            scheduler_G.step()
    print("Best mIoU=%.6f"%best_result)


if __name__ == '__main__':
    main()
