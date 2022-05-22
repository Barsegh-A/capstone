import argparse
import os
import numpy as np
import torch.utils.data
import librosa
from scipy.io.wavfile import read
import matplotlib.pyplot as plt

MAX_WAV_VALUE = 32768.0
mel_basis_regnet = torch.tensor(librosa.filters.mel(22050, n_fft=1024, fmin=125, fmax=7600, n_mels=80))

def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate

def get_spectrogram_regnet_np(y):
    spectrogram = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    mel_spec = np.dot(mel_basis_regnet, spectrogram)
    mel_spec = 20 * np.log10(np.maximum(1e-5, mel_spec)) - 20
    mel_spec = np.clip((mel_spec + 100) / 100, 0, 1.0)

    return mel_spec

def extract_from_wav(joined_path):
    wav, sr = load_wav(joined_path)
    wav = wav / MAX_WAV_VALUE
    wav = wav[:220500]
    mel_spec = get_spectrogram_regnet_np(wav)
    mel_spec = mel_spec[:, :860]

    return mel_spec

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_wavs_dir', default="original_wavs")
    parser.add_argument('--hifigan_wavs_dir', default="hifigan_wavs")
    parser.add_argument('--wavenet_wavs_dir', default="wavenet_wavs")
    parser.add_argument('--regnet_mels_dir', default="regnet_mels")
    parser.add_argument('--output_dir', default="visualizations")
    a = parser.parse_args()

    errors_wavenet = []
    errors_hifigan = []
    filelist = os.listdir(a.input_wavs_dir)
    for i, filname in enumerate(filelist):

        if filname.startswith('.'):
            continue

        mel_spec_original = extract_from_wav(os.path.join(a.input_wavs_dir, filname))

        plt.figure(figsize=(8, 9))
        plt.subplot(411)
        plt.imshow(mel_spec_original,
                   aspect='auto', origin='lower')
        plt.title(filname.split('.')[0] + "_ground_truth")

        mel_regnet = np.load(os.path.join(a.regnet_mels_dir, filname.split('.')[0]) + '.npy')
        plt.subplot(412)
        plt.imshow(mel_regnet,
                   aspect='auto', origin='lower')
        plt.title(filname.split('.')[0] + "_regnet_prevocoder")

        mel_spec_wavenet = extract_from_wav(os.path.join(a.wavenet_wavs_dir, filname))
        plt.subplot(413)
        plt.imshow(mel_spec_wavenet,
                   aspect='auto', origin='lower')
        plt.title(filname.split('.')[0] + "_wavenet")

        mel_spec_hifigan = extract_from_wav(os.path.join(a.hifigan_wavs_dir, filname))
        plt.subplot(414)
        plt.imshow(mel_spec_hifigan,
                   aspect='auto', origin='lower')
        plt.title(filname.split('.')[0] + "_hifigan")
        plt.tight_layout()

        errors_hifigan.append(np.sqrt(np.sum((mel_spec_hifigan - mel_regnet)**2)))
        errors_wavenet.append(np.sqrt(np.sum((mel_spec_wavenet - mel_regnet) ** 2)))


        os.makedirs(a.output_dir, exist_ok=True)
        plt.savefig(os.path.join(a.output_dir, filname.split('.')[0] + '.jpg'))

    print("WaveNet MSE: ", np.mean(errors_wavenet))
    print("HifiGan MSE: ", np.mean(errors_hifigan))


if __name__ == '__main__':
    main()
