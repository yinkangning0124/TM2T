import os

from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.train_options import TrainVQTokenizerOptions
from utils.plot_script import *

from networks.modules import *
from networks.quantizer import *
from data.dataset import MotionTokenizeDataset
from scripts.motion_process import *
from torch.utils.data import DataLoader
import codecs as cs
import torch

def loadVQModel(opt):
    vq_encoder = VQEncoderV3(dim_pose - 4, enc_channels, opt.n_down)
    # vq_decoder = VQDecoderV3(opt.dim_vq_latent, dec_channels, opt.n_resblk, opt.n_down)
    quantizer = None
    if opt.q_mode == 'ema':
        quantizer = EMAVectorQuantizer(opt.codebook_size, opt.dim_vq_latent, opt.lambda_beta)
    elif opt.q_mode == 'cmt':
        quantizer = Quantizer(opt.codebook_size, opt.dim_vq_latent, opt.lambda_beta)
    checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model', 'finest.tar'),
                            map_location=opt.device)
    vq_encoder.load_state_dict(checkpoint['vq_encoder'])
    quantizer.load_state_dict(checkpoint['quantizer'])
    return vq_encoder, quantizer


if __name__ == '__main__':
    parser = TrainVQTokenizerOptions()
    opt = parser.parse()

    opt.is_train = False
    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    if opt.gpu_id != -1:
        torch.cuda.set_device(opt.gpu_id)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.data_root = './dataset/Mixamo/'

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)

    opt.joints_num = 22
    opt.max_motion_length = 196
    dim_pose = 263

    enc_channels = [1024, opt.dim_vq_latent]
    dec_channels = [opt.dim_vq_latent, 1024, dim_pose]

    vq_encoder, quantizer = loadVQModel(opt)

    all_params = 0
    pc_vq_enc = sum(param.numel() for param in vq_encoder.parameters())
    print(vq_encoder)
    print("Total parameters of encoder net: {}".format(pc_vq_enc))
    all_params += pc_vq_enc

    pc_quan = sum(param.numel() for param in quantizer.parameters())
    print(quantizer)
    print("Total parameters of codebook: {}".format(pc_quan))
    all_params += pc_quan

    print('Total parameters of all models: {}'.format(all_params))

    token_data_dir = pjoin(opt.data_root, opt.name)
    os.makedirs(token_data_dir, exist_ok=True)

    start_token = opt.codebook_size
    end_token = opt.codebook_size + 1
    pad_token = opt.codebook_size + 2

    max_length = 55
    num_replics = 5
    opt.unit_length = 4

    file_root = 'new_joint_vecs'
    files = os.listdir(pjoin(opt.data_root, file_root))
    vq_encoder.to(opt.device)
    quantizer.to(opt.device)
    vq_encoder.eval()
    quantizer.eval()
    with torch.no_grad():
        for file in files:
            for e in range(num_replics):
                motion = np.load(file=pjoin(opt.data_root, file_root, file), allow_pickle=True)
                motion = torch.from_numpy(motion).to(opt.device).float()
                motion = motion[:196, :]
                motion = motion.unsqueeze(0)
                #motion = motion.to(opt.device).float()
                pre_latents = vq_encoder(motion[..., :-4])
                indices = quantizer.map2index(pre_latents)
                indices = list(indices.cpu().numpy())
                indices = [str(token) for token in indices]
                with cs.open(pjoin(token_data_dir, '%s.txt'%file[:-4]), 'a+') as f:
                    if e!= 0:
                            f.write('\n')
                    f.write(' '.join(indices))
