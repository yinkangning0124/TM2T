import os
from os.path import join as pjoin
import codecs as cs

import utils.paramUtil as paramUtil
from options.evaluate_options import TestT2MOptions
from utils.plot_script import *

from networks.transformer import TransformerV3, TransformerV2
from networks.quantizer import *
from networks.modules import *
from networks.translator import Translator
from data.dataset import Motion2TextEvalDataset
from scripts.motion_process import *
from torch.utils.data import DataLoader
from utils.word_vectorizer import WordVectorizerV2


def build_models(opt):
    if opt.m2t_v3:
        m2t_transformer = TransformerV3(n_mot_vocab, opt.mot_pad_idx, n_txt_vocab, opt.txt_pad_idx,
                                        d_src_word_vec=512, d_trg_word_vec=300,
                                        d_model=opt.d_model, d_inner=opt.d_inner_hid, n_enc_layers=opt.n_enc_layers,
                                        n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v,
                                        dropout=0.1,
                                        n_src_position=100, n_trg_position=50)
    else:
        m2t_transformer = TransformerV2(n_mot_vocab, opt.mot_pad_idx, n_txt_vocab, opt.txt_pad_idx, d_src_word_vec=512,
                                        d_trg_word_vec=512,
                                        d_model=opt.d_model, d_inner=opt.d_inner_hid, n_enc_layers=opt.n_enc_layers,
                                        n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v,
                                        dropout=0.1,
                                        n_src_position=100, n_trg_position=50,
                                        trg_emb_prj_weight_sharing=opt.proj_share_weight
                                        )
    checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model', '%s.tar'%(opt.which_epoch)),
                            map_location=opt.device)
    m2t_transformer.load_state_dict(checkpoint['m2t_transformer'])
    print('Loading m2t_transformer model: Epoch %03d Total_Iter %03d' % (checkpoint['ep'], checkpoint['total_it']))

    return m2t_transformer



if __name__ == '__main__':
    parser = TestT2MOptions()
    opt = parser.parse()

    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)
    if opt.gpu_id != -1:
        torch.cuda.set_device(opt.gpu_id)

    opt.data_root = './mixamo_datasets/humanml3d_form/'
    opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
    opt.m_token_dir = pjoin(opt.data_root, 'VQVAEV3_CB1024_CMT_H1024_NRES3')
    opt.joints_num = 22
    opt.max_motion_token = 55
    opt.max_motion_frame = 196
    dim_pose = 263

    n_mot_vocab = opt.codebook_size + 3
    opt.mot_start_idx = opt.codebook_size
    opt.mot_end_idx = opt.codebook_size + 1
    opt.mot_pad_idx = opt.codebook_size + 2

    enc_channels = [1024, opt.dim_vq_latent]
    dec_channels = [opt.dim_vq_latent, 1024, dim_pose]

    w_vectorizer = WordVectorizerV2('./glove', 'our_vab')
    n_txt_vocab = len(w_vectorizer) + 1
    _, _, opt.txt_start_idx = w_vectorizer['sos/OTHER']
    _, _, opt.txt_end_idx = w_vectorizer['eos/OTHER']
    opt.txt_pad_idx = len(w_vectorizer)

    m2t_transformer = build_models(opt)

    opt.repeat_times = opt.repeat_times if opt.sample else 1

    if opt.sample:
        m2t_transformer.to(opt.device)
    else:
        translator = Translator(m2t_transformer, beam_size=opt.beam_size, max_seq_len=30,
                                src_pad_idx=opt.mot_pad_idx, trg_pad_idx=opt.txt_pad_idx,
                                trg_sos_idx=opt.txt_start_idx, trg_eos_idx=opt.txt_end_idx)
        translator.to(opt.device)

    '''
    update dataloader
    '''
    motion = np.load(pjoin(opt.motion_dir, 'markermanhiphopdance_my1.npy'), allow_pickle=True)
    motion = motion[:196, :]
    #motion = motion.unsqueeze(0)
    m_tokens = []
    with cs.open(pjoin(opt.m_token_dir, '00004.txt'), 'r') as f:
        for line in f.readlines():
            m_tokens.append(line.strip().split(' '))

    m_tokens = m_tokens[0]
    m_tokens = [int(token) for token in m_tokens]
    m_tokens = [opt.mot_start_idx] + \
                   m_tokens + \
                   [opt.mot_end_idx] + \
                   [opt.mot_pad_idx] * (opt.max_motion_token - len(m_tokens) - 2)    
    m_tokens = np.array(m_tokens, dtype=int)


    print('Generating Results')
    result_dict = {}
    with torch.no_grad():
        m_tokens = torch.from_numpy(m_tokens).to(opt.device).long()
        m_tokens = m_tokens.unsqueeze(0)
        # m_tokens = m_tokens.detach().to(opt.device).long()
        pred_tokens = translator.translate_sentence(m_tokens)
        pred_tokens = pred_tokens[1:-1]
        pred_caption = ' '.join(w_vectorizer.itos(i) for i in pred_tokens)
        print(pred_caption)