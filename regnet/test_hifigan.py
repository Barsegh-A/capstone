import argparse
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from data_utils import RegnetLoader
from model import Regnet
from config import _C as config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
from pathlib import Path

sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))
print(sys.path)

import hifigan

def test_model():
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    model = Regnet()
    valset = RegnetLoader(config.test_files)
    test_loader = DataLoader(valset, num_workers=4, shuffle=False,
                             batch_size=config.batch_size, pin_memory=False)
    if config.checkpoint_path != '':
        model.load_checkpoint(config.checkpoint_path)
    model.setup()
    model.eval()


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            model.parse_batch(batch)
            model.forward()            
            for j in range(len(model.fake_B)):
                plt.figure(figsize=(8, 9))
                plt.subplot(311)
                plt.imshow(model.real_B[j].data.cpu().numpy(), 
                                aspect='auto', origin='lower')
                plt.title(model.video_name[j]+"_ground_truth")
                plt.subplot(312)
                plt.imshow(model.fake_B[j].data.cpu().numpy(), 
                                aspect='auto', origin='lower')
                plt.title(model.video_name[j]+"_predict")
                plt.subplot(313)
                plt.imshow(model.fake_B_postnet[j].data.cpu().numpy(), 
                                aspect='auto', origin='lower')
                plt.title(model.video_name[j]+"_postnet")
                plt.tight_layout()
                os.makedirs(config.save_dir, exist_ok=True)
                plt.savefig(os.path.join(config.save_dir, model.video_name[j]+".jpg"))
                plt.close()
                np.save(os.path.join(config.save_dir, model.video_name[j]+".npy"), 
                          model.fake_B[j].data.cpu().numpy())
                mel_spec = model.fake_B[j].data.cpu().numpy()
                save_path = os.path.join(config.save_dir, model.video_name[j]+".wav")

                hifigan.inference_regnet(mel_spec, save_path, config.hifigan_path, device)

    model.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, default='',
                        help='file for configuration')
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        config.merge_from_file(args.config_file)
 
    config.merge_from_list(args.opts)
    # config.freeze()


    torch.backends.cudnn.enabled = config.cudnn_enabled
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    print("Dynamic Loss Scaling:", config.dynamic_loss_scaling)
    print("cuDNN Enabled:", config.cudnn_enabled)
    print("cuDNN Benchmark:", config.cudnn_benchmark)

    test_model()