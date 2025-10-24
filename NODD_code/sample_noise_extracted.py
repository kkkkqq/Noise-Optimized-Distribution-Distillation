"""
Sample new images from a pre-trained DiT.
"""
import os
import torch
import torch.backends
import torch.jit
from tqdm import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark=True
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import numpy as np
from sliced_models import SlicedDiT

#################################################################################
#                               Sampling Helpers                                #
#################################################################################

from typing import Dict, Union
from torch import Tensor
from torchvision.transforms.functional import hflip

def compute_new_noises(extractor, noises, cur_means, cur_log_vars, cur_stats_lst, num_steps, reg_strength, cfg=True, opt_type='sgd', opt_lr=0.1, cos_strength=10.0, init_ratio=1.0, channel=False, extracted_weight=1.0, x_weight=0.0, no_extract_repel=False, norm_weight=1.0):
    device = cur_means[0].device
    if cfg:
        batch_size = noises[0].size(0)//2
    else:
        batch_size = noises[0].size(0)

    with torch.no_grad():
        if cfg:
            noise = torch.cat([ns.chunk(2)[0] for ns in noises], 0)
            cur_mean = torch.cat([m.chunk(2)[0] for m in cur_means],0)
            cur_log_var = torch.cat([v.chunk(2)[0] for v in cur_log_vars],0)
        else:
            noise = torch.cat(noises, dim=0)
            cur_mean = torch.cat(cur_means, dim=0)
            cur_log_var = torch.cat(cur_log_vars, dim=0)

        shape = noise.shape
        target_norm = np.sqrt(np.prod(shape[1:]))
        cur_stats = cur_stats_lst[0]
        cur_x_stats = cur_stats_lst[1]
        if extracted_weight!=0:
            if channel:
                target_mean = cur_stats['mean'][None,:,None,None].to(device)
                target_std = cur_stats['std'][None,:,None,None].to(device)
            else:
                target_mean = cur_stats['mean'].view(-1).unsqueeze(0).to(device)
                target_std = cur_stats['std'].view(-1).unsqueeze(0).to(device)
        else:
            target_mean = None
            target_std = None
        if x_weight!=0:
            x_mean = cur_x_stats['mean']
            x_mean = x_mean.view(-1).contiguous().unsqueeze(0).to(device)
            x_std = cur_x_stats['std']
            x_std = x_std.view(-1).contiguous().unsqueeze(0).to(device)
        else:
            x_mean = None
            x_std = None

        # print('target mean', target_mean)
        # print('target std', target_std)
        
        var = torch.exp(0.5*cur_log_var)

    noise_tsr = noise.mul(init_ratio).detach().clone().requires_grad_()
    if opt_type=='sgd':
        opt = torch.optim.SGD([noise_tsr], opt_lr, 0.9)
    elif opt_type=='adam':
        opt = torch.optim.Adam([noise_tsr], opt_lr)
    elif opt_type=='adamw':
        opt = torch.optim.AdamW([noise_tsr], opt_lr)
    mask = torch.eye(noise_tsr.size(0)).to(noise_tsr.device).sub(1).mul(-1)
    eye = torch.eye(noise_tsr.size(0)).to(noise_tsr.device)*(-100)
    
    for stp in range(num_steps):
        opt.zero_grad()
        new_tsr_ = cur_mean + var * noise_tsr
        x = new_tsr_.view(new_tsr_.size(0),-1)
        norms = noise_tsr.view(noise_tsr.size(0),-1).norm(dim=1)
        norm_loss = ((norms-target_norm)**2).sum() + torch.abs(norms-target_norm).sum()
        if extracted_weight != 0:
            #这一步是用extract过的样本所计算的损失
            if channel:
                new_tsr_ = extractor(new_tsr_).sub(target_mean).div(target_std)
                new_tsr = new_tsr_.view(new_tsr_.size(0),-1)
            else:
                new_tsr_ = extractor(new_tsr_).view(new_tsr_.size(0),-1)
                new_tsr = new_tsr_.sub(target_mean).div(target_std)
            # print('new tsr shape', new_tsr.shape)
            
            if no_extract_repel:
                cos_loss = 0.0
            else:
                ntsr = new_tsr.div(new_tsr.norm(dim=1,keepdim=True))
                cos_mtx = ntsr.mm(ntsr.T)
                cos_mtx = cos_mtx*mask + eye
                cos_loss = cos_mtx.max(dim=1)[0].sum()
            
            
            if not channel:
                mean_loss = new_tsr.mean(0)
                # print('mean shape: ', mean_loss.shape)
                mean_loss = mean_loss.abs().sum() + mean_loss.square().sum()
                std_ = new_tsr.std(0, unbiased=False)
                std_loss = std_.sub(1)
                # print('std shape', std_loss.shape)
                std_loss = std_loss.abs().sum() + std_loss.square().sum()
            else:
                nch = new_tsr_.shape[1]
                scale_factor = np.prod(new_tsr_.shape[2:])
                mean_loss = new_tsr_.mean([0,2,3])
                mean_loss = (mean_loss.abs().sum() + mean_loss.square().sum())*scale_factor
                std_loss = new_tsr_.permute(1,0,2,3).contiguous().reshape([nch,-1]).std(1, unbiased=False).sub(1)
                std_loss = (std_loss.abs().sum() + std_loss.square().sum())*scale_factor
            loss = norm_weight*norm_loss + cos_strength*cos_loss + reg_strength*(mean_loss + std_loss)
        else:
            loss = 0.0

        #这一步是用没有extract过的样本所计算的损失
        if x_weight != 0:
            new_tsr = x.sub(x_mean).div(x_std)
            ntsr = new_tsr.div(new_tsr.norm(dim=1,keepdim=True))
            cos_mtx = ntsr.mm(ntsr.T)
            cos_mtx = cos_mtx*mask + eye
            cos_loss = cos_mtx.max(dim=1)[0].sum()
            mean_loss = new_tsr.mean(0)
            if no_extract_repel:
                cos_loss = cos_loss/x_weight
            # print('mean shape: ', mean_loss.shape)
            mean_loss = mean_loss.abs().sum() + mean_loss.square().sum()
            std_ = new_tsr.std(0, unbiased=False)
            std_loss = std_.sub(1)
            # print('std shape', std_loss.shape)
            std_loss = std_loss.abs().sum() + std_loss.square().sum()
            x_loss = norm_weight*norm_loss + cos_strength*cos_loss + reg_strength*(mean_loss + std_loss)

        else:
            x_loss = 0.0
        if x_weight==0 and extracted_weight==0:
            loss = norm_loss*norm_weight
        else:
            loss = loss*extracted_weight + x_loss*x_weight
        loss.backward()
        opt.step()
    
    with torch.no_grad():
        noise_tsr = noise_tsr.detach()
        new_noise = noise_tsr.view(*shape)
        new_noises = new_noise.split(batch_size)
        # print('new noises: ', new_noises[0][0,0,:,:])
        # print('max: ', torch.max(new_noises[0]), 'min: ', torch.min(new_noises[0]), 'norm: ', torch.norm(new_noises[0].view(new_noises[0].size(0),-1), dim=1))
        # print('mean: ', torch.mean(new_noises[0][0]), 'std: ', torch.std(new_noises[0][0]))
        # print('new noise shape', new_noises[0].shape)
        if cfg:
            null_noises = [ns.chunk(2)[1] for ns in noises]
            new_noises = [torch.cat([nns,ns],0) for nns, ns in zip(new_noises, null_noises)]
    return new_noises

    


        


    


def main(args):
    # Setup PyTorch:
    dpm = args.devices_per_model #每个模型被切分到了多少个device里
    assert torch.cuda.device_count()>=dpm
    devices = list(range(dpm))
    devices.reverse()
    device = devices[0]
    from group_extractor import GroupExtractor
    if not args.no_opt:
        from group_extractor import GroupExtractor
        tpe = args.extractor_type
        extractor = GroupExtractor(4, tpe)
        ckpt_dir = os.path.join(*os.path.split(args.stats_dir)[:-1])
        extractor.load_state_dict(torch.load(os.path.join(ckpt_dir, f'extractor_{args.extractor_type}.pt')))
        extractor.to(device)
        extractor.eval()
        pseuin = torch.randn((2,4,32,32), device=device)
        extractor = torch.jit.trace_module(extractor, {'forward':pseuin})
        for pa in extractor.parameters():
            pa.requires_grad_(False)
    else:
        extractor = GroupExtractor(4, args.extractor_type)
        extractor = None
    seed = args.seed
    torch.manual_seed(args.seed)
    print(f"Starting seed={seed}.")
    

    # Labels to condition the model
    with open('./misc/class_indices.txt', 'r') as fp:
        all_classes = fp.readlines()
    all_classes = [class_index.strip() for class_index in all_classes]
    if args.spec == 'woof':
        file_list = './misc/class_woof.txt'
    elif args.spec == 'nette':
        file_list = './misc/class_nette.txt'
    elif args.spec == 'imagenet1k':
        file_list = './misc/class_indices.txt'
    else:
        file_list = './misc/class100.txt'
    with open(file_list, 'r') as fp:
        sel_classes = fp.readlines()

    phase = max(0, args.phase)
    cls_from = args.nclass * phase
    cls_to = args.nclass * (phase + 1)
    cls_to = min(cls_to, len(sel_classes))
    sel_classes = sel_classes[cls_from:cls_to]
    sel_classes = [sel_class.strip() for sel_class in sel_classes]
    class_labels = []

        
    print('selected classes for sampling: ')
    print(sel_classes)
    
    for sel_class in sel_classes:
        class_labels.append(all_classes.index(sel_class))

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8

    # model = DiT_models[args.model](
    #     input_size=latent_size,
    #     num_classes=args.num_classes
    # ).to(device)
    model = SlicedDiT(input_size=latent_size, num_classes=args.num_classes)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # important!
    model.slice(args.slice_before, devices, True)
    # model = DDP(model)
    diffusion = create_diffusion(str(args.num_sampling_steps))
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path=f'sd-vae-{args.vae}').to(device)

    batch_size = args.batch_size

    for class_label, sel_class in zip(class_labels, sel_classes):
        print('class label: ', class_label, ', class name: ', sel_class)
        os.makedirs(os.path.join(args.save_dir, sel_class), exist_ok=True)
        #load stats
        if not args.no_opt:
            if args.extracted_weight != 0:
                if args.channel:
                    stats_dct = torch.load(os.path.join(args.stats_dir, sel_class, f'extracted_channels_{args.extractor_type}.pt'))#TODO: 这里把它解comment！
                else:
                    stats_dct = torch.load(os.path.join(args.stats_dir, sel_class, f'extracted_stats_{args.extractor_type}.pt'))#TODO: 这里把它解comment！
            else:
                stats_dct = [None for _ in range(50)]
            if args.x_weight !=0:
                x_stats_dct = torch.load(os.path.join(args.stats_dir, sel_class, 'x_stats.pt'))
            else:
                x_stats_dct = [None for _ in range(50)]
        else:
            stats_dct = None
        zs = torch.randn((args.num_samples, 4, latent_size, latent_size), device=device)

        # partition the noise vectors into batches
        batched_zs = zs.split(batch_size)
        batched_ys = [torch.tensor([class_label]*z.size(0), device=device) for z in batched_zs]
        batched_y_nulls = [torch.tensor([1000]*z.size(0), device=device) for z in batched_zs]
        batched_zs = [torch.cat([z,z],0) for z in batched_zs]

        indices = list(range(diffusion.num_timesteps))[::-1]

        for i in tqdm(indices):

            # psample
            with torch.no_grad():
                new_zs = []
                noises = []
                cur_means = []
                cur_log_vars = []
                batched_ts = [torch.tensor([i] * z.shape[0], device=device) for z in batched_zs]
                for z, y_, y_null, t in zip(batched_zs, batched_ys, batched_y_nulls, batched_ts):
                    y = torch.cat([y_, y_null], 0)
                    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                    out = diffusion.p_mean_variance(model.forward_with_cfg, z, t, 
                                                    clip_denoised=False, model_kwargs=model_kwargs)
                    noises.append(torch.randn_like(z))
                    cur_means.append(out['mean'])
                    cur_log_vars.append(out['log_variance'])
                
            if i!=0 and not args.no_opt:
                new_noises = compute_new_noises(extractor, noises, cur_means, cur_log_vars, [stats_dct[i], x_stats_dct[i]], args.repulsion_steps, args.reg_strength, True, args.opt_type, args.opt_lr, args.cos_strength, args.init_ratio, args.channel, args.extracted_weight, args.x_weight, args.no_extract_repel, args.norm_weight)
            else:
                new_noises = noises
            
            with torch.no_grad():
                for z, t, mean, logvar, nns in zip(batched_zs, batched_ts, cur_means, cur_log_vars, new_noises):
                    nonzero_mask = (
                        (t != 0).float().view(-1, *([1] * (len(z.shape) - 1)))
                    )  # no noise when t == 0
                    # print('mean: ', mean.shape)
                    # print('logvar: ', logvar.shape)
                    # print('nns:', nns.shape)
                    sample = mean + nonzero_mask * torch.exp(0.5 * logvar) * nns
                    new_zs.append(sample)
                batched_zs = new_zs

        with torch.no_grad():
            full_zs = torch.cat([z.chunk(2)[0] for z in batched_zs],0)
            
            for zidx, z in enumerate(full_zs):
                sample = vae.decode(z.unsqueeze(0).chunk(2)[0] / 0.18215).sample.cpu()
                save_image(sample, os.path.join(args.save_dir, sel_class,
                                               f"{zidx + args.total_shift}.png"), normalize=True, value_range=(-1, 1))
            print('saved to ', os.path.join(args.save_dir, sel_class))
        del batched_zs
        
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--spec", type=str, default='none', help='specific subset for generation')
    parser.add_argument("--save-dir", type=str, default='../logs/test', help='the directory to put the generated images')
    parser.add_argument("--num-samples", type=int, default=100, help='the desired IPC for generation')
    parser.add_argument("--total-shift", type=int, default=0, help='index offset for the file name')
    parser.add_argument("--nclass", type=int, default=10, help='the class number for generation')
    parser.add_argument("--phase", type=int, default=0, help='the phase number for generating large datasets')
    
    parser.add_argument('--devices-per-model', type=int, default=2, help='how many devices to split the model on')
    parser.add_argument('--slice-before', nargs='+', type=int, default=[], help='blocks before which a slicing occurs')
    
    parser.add_argument('--batch-size', type=int, default=2, help='batch size for generation')
    parser.add_argument("--repulsion-steps", type=int, default=200, help='number of steps for repulsion')
    parser.add_argument("--reg-strength", type=float, default=0.01, help='strength of regularization')
    parser.add_argument("--cos-strength", type=float, default=10.0, help='strength of regularization')
    
    parser.add_argument('--no-opt', action='store_true', help='turn off optimization')
    parser.add_argument('--stats-dir', type=str, default='../stats/woof/000')
    parser.add_argument('--extractor-type', type=str, default='S_16')
    parser.add_argument('--opt-type', type=str, default='sgd')
    parser.add_argument('--opt-lr', type=float, default=0.1)
    parser.add_argument('--init-ratio', type=float, default=1.0)
    parser.add_argument('--stats-type', type=str, choices=['channel', 'full'], default='channel')
    parser.add_argument('--extracted-weight', type=float, default=1.0)
    parser.add_argument('--x-weight', type=float, default=0.0)
    parser.add_argument('--norm-weight', type=float, default=1.0)
    parser.add_argument('--no-extract-repel', action='store_true')
    

    args = parser.parse_args()

    if isinstance(args.slice_before, int):
        args.slice_before = [args.slice_before]
    if args.stats_type == 'full':
        args.channel = False
    else:
        args.channel = True

    main(args)
