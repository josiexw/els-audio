import argparse
import numpy as np 
import os

from scipy.io import wavfile
from ssqueezepy import Wavelet, cwt, icwt

INT16_TO_FLOAT32 = 32768.0
FLOAT32_TO_INT16 = 32767

def compute_cwt(wav_file, mu=5, nv=24):
    sample_rate, data = wavfile.read(wav_file)
    data = data.astype(np.float32) / INT16_TO_FLOAT32

    wavelet = Wavelet(('morlet', {'mu': mu}))
    Wx, scales = cwt(data, wavelet=wavelet, fs=sample_rate, nv=nv)
    return Wx, scales, wavelet, sample_rate

def invert_cwt(Wx, scales, wavelet, sample_rate, reconstruct_path):
    audio_recreation = icwt(Wx, wavelet, scales=scales)
    audio_recreation /= np.max(np.abs(audio_recreation))
    audio_recreation_int16 = np.int16(audio_recreation * FLOAT32_TO_INT16)
    wavfile.write(reconstruct_path, sample_rate, audio_recreation_int16)

def preprocess_db(db_path):
    for root, _, files in os.walk(db_path):
        for file in files:
            compute_cwt(str(os.path.join(root, file)))

def main():
    parser = argparse.ArgumentParser(description='Compute CWT for all files in a specified directory.')
    parser.add_argument('--db_path', type=str, required=True, help='Path to database directory')
    args = parser.parse_args()

    preprocess_db(args.db_path)

if __name__ == '__main__':
    main()