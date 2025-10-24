"""
Sample new images from a pre-trained DiT.
"""
import os
import torch
from tqdm import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
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
from group_extractor import GroupExtractor



def main(args):
    # Setup PyTorch:
    dpm = args.devices_per_model #每个模型被切分到了多少个device里
    assert torch.cuda.device_count()>=dpm
    devices = list(range(dpm))
    seed = args.seed
    torch.manual_seed(args.seed)
    print(f"Starting seed={seed}.")
    device = devices[0]

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
    print('selected classes for stats: ')
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
    # vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path=f'sd-vae-{args.vae}').to(device)

    batch_size = 20

    #构建grouped extractor
    extractors = []
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_dir = os.path.join(*os.path.split(args.save_dir)[:-1])
    for tpe in args.extractor_types:
        extractor = GroupExtractor(4, tpe)
        extractor.to(device)
        extractor.eval()
        for pa in extractor.parameters():
            pa.requires_grad_(False)
        
        if os.path.exists(os.path.join(ckpt_dir, f'extractor_{tpe}.pt')):
            print('loading from ',os.path.join(ckpt_dir, f'extractor_{tpe}.pt'))
            extractor.load_state_dict(torch.load(os.path.join(ckpt_dir, f'extractor_{tpe}.pt')))
        else:
            print('saving to ',os.path.join(ckpt_dir, f'extractor_{tpe}.pt'))
            torch.save(extractor.state_dict(), os.path.join(ckpt_dir, f'extractor_{tpe}.pt'))
        extractors.append(extractor)

    for class_label, sel_class in zip(class_labels, sel_classes):
        print('class label: ', class_label, ', class name: ', sel_class)
        os.makedirs(os.path.join(args.save_dir, sel_class), exist_ok=True)
        svpths = []
        for tpe, uch in zip(args.extractor_types, args.stats_types):
            if uch == 'channel':
                svpths.append(os.path.join(args.save_dir, sel_class, f'extracted_channels_{tpe}.pt'))
            elif uch == 'full':
                svpths.append(os.path.join(args.save_dir, sel_class, f'extracted_stats_{tpe}.pt'))
            else:
                raise NotImplementedError('only support channel or full!')
        if not args.override:
            num_exist = 0
            for svpth in svpths:
                if os.path.exists(svpth):
                    print(f'stats file for {sel_class} exists at {svpth}')
                    num_exist+=1
            if num_exist==len(svpths):
                if os.path.exists(os.path.join(args.save_dir, sel_class, 'x_stats.pt')):
                    print('All required stats file exists, skip this class.')
                    continue

        zs = torch.randn((args.num_samples, 4, latent_size, latent_size), device=device)

        # partition the noise vectors into batches
        batched_zs = zs.split(batch_size)
        batched_ys = [torch.tensor([class_label]*z.size(0), device=device) for z in batched_zs]
        batched_y_nulls = [torch.tensor([1000]*z.size(0), device=device) for z in batched_zs]
        batched_zs = [torch.cat([z,z],0) for z in batched_zs]
        stats_dcts = [dict() for _ in extractors]
        x_stats_dct = dict()

        indices = list(range(diffusion.num_timesteps))[::-1]

        for i in tqdm(indices):
            # psample
            with torch.no_grad():
                new_zs = []
                new_feats = [[] for _ in extractors]
                x_tsrs = []
                batched_ts = [torch.tensor([i] * z.shape[0], device=device) for z in batched_zs]
                for z, y_, y_null, t in zip(batched_zs, batched_ys, batched_y_nulls, batched_ts):
                    y = torch.cat([y_, y_null], 0)
                    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                    out = diffusion.p_mean_variance(model.forward_with_cfg, z, t, 
                                                    clip_denoised=False, model_kwargs=model_kwargs)
                    noise = torch.randn_like(z)
                    nonzero_mask = (
                        (t != 0).float().view(-1, *([1] * (len(z.shape) - 1)))
                    )  # no noise when t == 0
                    sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
                    new_zs.append(sample)
                    fsmp = sample.chunk(2)[0]
                    x_tsrs.append(fsmp)
                    for extractor, new_feat in zip(extractors, new_feats):
                        feature = extractor(fsmp)
                        new_feat.append(feature)
                batched_zs = new_zs

                sdcts = [dict() for _ in extractors]
                for sdct, new_feat, stats_dct, stats_type in zip(sdcts, new_feats, stats_dcts, args.stats_types):
                    feats_tsr = torch.cat(new_feat, dim=0)
                    if stats_type == 'full':
                        sdct['mean'] = torch.mean(feats_tsr, dim=0).cpu()
                        sdct['std'] = torch.std(feats_tsr, dim=0, unbiased=False).cpu()
                        sdct['max'] = torch.max(feats_tsr, dim=0)[0].cpu()
                        sdct['min'] = torch.min(feats_tsr, dim=0)[0].cpu()
                    elif stats_type == 'channel':
                        nch = feats_tsr.shape[1]
                        sdct['mean'] = feats_tsr.mean([0,2,3]).cpu()
                        sdct['std'] = feats_tsr.permute(1,0,2,3).contiguous().reshape([nch,-1]).std(1, unbiased=False).cpu()
                    else:
                        raise NotImplementedError('only support full or channel!')
                    stats_dct[i] = sdct
                    feats_tsr = None
                x_tsr = torch.cat(x_tsrs, dim=0)
                xdct = {'mean': x_tsr.mean(dim=0).cpu(), 'std': x_tsr.std(dim=0, unbiased=False).cpu()}
                x_stats_dct[i] = xdct
        for stats_dct, svpth in zip(stats_dcts, svpths):
            torch.save(stats_dct, svpth)
            print(f'saved stats file to {svpth}')
        torch.save(x_stats_dct, os.path.join(args.save_dir, sel_class, f'x_stats.pt'))

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
    parser.add_argument('--override', action='store_true', help='whether override the stats.pt file if it exists')
    parser.add_argument('--extractor-types', type=str, nargs='+', default=['L'])
    parser.add_argument('--stats-types', choices=['channel', 'full'], type=str, nargs='+', default=['full'])
    
    args = parser.parse_args()

    if isinstance(args.slice_before, int):
        args.slice_before = [args.slice_before]
    if isinstance(args.extractor_types, str):
        args.extractor_types = [args.extractor_types]
    if isinstance(args.stats_types, str):
        args.stats_types = [args.stats_types]
    assert len(args.extractor_types)==len(args.stats_types)
    main(args)
