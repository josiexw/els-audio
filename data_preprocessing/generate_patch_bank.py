import argparse 
import h5py
import numpy as np 
import os 

from tqdm import tqdm
from typing import Iterator, Tuple, List 

def generate_1d_hann_window(I: int) -> np.ndarray:
    i = np.arange(I, dtype=np.float32)
    return 0.5 * (1.0 - np.cos((2.0 * np.pi * i) / (I - 1.0)))

def get_hann_patch_generator(
        Wx: np.ndarray,
        T_len: int = 1024,
        S_len: int = 32,
        T_stride: int = None,
        S_stride: int = 8,
) -> Iterator[np.ndarray]:
    _, n_time = Wx.shape 
    if T_stride is None:
        T_stride = max(1, T_len // 2)
    
    pad_top, pad_bottom = S_len - 1, S_len - 1
    total_hops = int(np.ceil((n_time - T_len) / T_stride)) + 1 if n_time > T_len else 1
    expected_len = (total_hops - 1) * T_stride + T_len
    pad_right = max(0, expected_len - n_time)
    Wx_pad = np.pad(Wx, ((pad_top, pad_bottom), (0, pad_right)), mode='reflect')

    hann_time = generate_1d_hann_window(T_len)
    hann_scale = generate_1d_hann_window(S_len)
    hann_2d = np.outer(hann_scale, hann_time).astype(np.float32)

    s_starts = list(range(0, Wx_pad.shape[0] - S_len + 1, S_stride)) 
    t_starts = list(range(0, Wx_pad.shape[1] - T_len + 1, T_stride))

    for s in s_starts:
        for t in t_starts:
            patch = Wx_pad[s:s+S_len, t:t+T_len]
            hann_patch = patch * hann_2d
            hann_patch_concat = np.stack([hann_patch.real, hann_patch.imag], axis=0).astype(np.float32)
            yield hann_patch_concat

def compute_patch_bank(
        preprocess_dir: str,
        save_dir: str,
        T_len: int,
        S_len: int, 
        T_stride: int = None,
        S_stride: int = 8,
        batch_size: int = 128
):
    for root, _, files in os.walk(preprocess_dir):
        for file in files:
            audio_id = file.split('.')[0]
            save_path = os.path.join(save_dir, f'{audio_id}_patches.h5')

            with h5py.File(os.path.join(root, file), 'r') as f:
                Wx = f['Wx'][:]
            
            patch_gen = get_hann_patch_generator(Wx, T_len, S_len, T_stride, S_stride)
            with h5py.File(save_path, 'w') as g:
                dataset = g.create_dataset(
                    'patches',
                    shape=(0, 2, S_len, T_len),
                    maxshape=(None, 2, S_len, T_len),
                    dtype=np.float32,
                    chunks=(min(batch_size, 32), 2, S_len, T_len)
                )

                buffer, total = [], 0
                for patch in tqdm(patch_gen):
                    buffer.append(patch)
                    if len(buffer) >= batch_size:
                        to_write = np.stack(buffer, axis=0)
                        new_size = total + to_write.shape[0]
                        dataset.resize(new_size, axis=0)
                        dataset[total:new_size] = to_write
                        total = new_size
                        buffer.clear()
                    
                if buffer:
                    to_write = np.stack(buffer, axis=0)
                    new_size = total + to_write.shape[0]
                    dataset.resize(new_size, axis=0)
                    dataset[total:new_size] = to_write
                    total = new_size
                    buffer.clear()

def main():
    parser = argparse.ArgumentParser(description='Compute patches of complex audio transforms.')
    parser.add_argument('--preprocess_dir', type=str, required=True, help='Path to preprocessed complex audio transforms.')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to target directory to store computed patches.')
    parser.add_argument('--T_len', type=int, required=False, help='Length of patch along time dimension.')
    parser.add_argument('--S_len', type=int, required=False, help='Length of patch along scale dimension.')
    args = parser.parse_args()

    compute_patch_bank(args.preprocess_dir, args.save_dir, args.T_len, args.S_len)

if __name__ == '__main__':
    main()