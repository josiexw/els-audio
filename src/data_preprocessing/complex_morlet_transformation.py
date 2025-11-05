import argparse
import h5py
import numpy as np 
import os

from scipy.io import wavfile
from ssqueezepy import Wavelet, cwt, icwt

INT16_TO_FLOAT32 = 32768.0
FLOAT32_TO_INT16 = 32767

def compute_cwt(wav_file, wavelet_type=('morlet', {'mu': 5}), nv=24):
    sample_rate, data = wavfile.read(wav_file)
    data = data.astype(np.float32) / INT16_TO_FLOAT32

    wavelet = Wavelet(wavelet_type)
    Wx, scales = cwt(data, wavelet=wavelet, fs=sample_rate, nv=nv)
    Wx = Wx.astype(np.complex64)
    return Wx, scales, sample_rate

def invert_cwt(Wx, scales, sample_rate, reconstruct_path, wavelet_type=('morlet', {'mu': 5})):
    wavelet = Wavelet(wavelet_type)
    audio_recreation = icwt(Wx, wavelet, scales=scales)
    audio_recreation /= np.max(np.abs(audio_recreation))
    audio_recreation_int16 = np.int16(audio_recreation * FLOAT32_TO_INT16)
    wavfile.write(reconstruct_path, sample_rate, audio_recreation_int16)

def preprocess_db(db_path, target_db_path):
    for root, _, files in os.walk(db_path):
        for file in files:
            audio_id = file.split('.')[0]
            Wx, scales, sample_rate  = compute_cwt(str(os.path.join(root, file)))

            with h5py.File(f'{target_db_path}/{audio_id}.h5', 'w') as f:
                f.create_dataset('Wx', data=Wx, compression='gzip')
                f.create_dataset('scales', data=scales)
                f.create_dataset('sample_rate', data=sample_rate)

def main():
    parser = argparse.ArgumentParser(description='Compute CWT for all files in a specified directory.')
    parser.add_argument('--db_path', type=str, required=True, help='Path to audio database directory.')
    parser.add_argument('--target_db_path', type=str, required=True, help='Path to target directory to store preprocessed audio files.')
    args = parser.parse_args()

    preprocess_db(args.db_path, args.target_db_path)

if __name__ == '__main__':
    main()