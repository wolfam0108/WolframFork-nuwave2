"""
Chunked inference for NU-Wave 2
Processes long audio files in overlapping chunks to fit in VRAM
"""

from lightning_model import NuWave2
from omegaconf import OmegaConf as OC
import os
import argparse
import torch
import librosa as rosa
from scipy.io.wavfile import write as swrite
import numpy as np
from scipy.signal import sosfiltfilt
from scipy.signal import cheby1
from scipy.signal import resample_poly
from tqdm import tqdm


def process_chunked(model, wav_l, band, steps, noise_schedule, 
                    chunk_size=48000*10, overlap=48000*2, device='cuda'):
    """
    Process audio in overlapping chunks
    
    Args:
        model: NuWave2 model
        wav_l: Low-res audio tensor [1, T]
        band: Band tensor [1, F]
        steps: Diffusion steps
        noise_schedule: Noise schedule
        chunk_size: Chunk size in samples (default: 10 seconds at 48kHz)
        overlap: Overlap size in samples (default: 2 seconds)
        device: Device to use
    
    Returns:
        Reconstructed audio tensor
    """
    total_length = wav_l.shape[1]
    hop = chunk_size - overlap
    
    # Output buffer with weights for overlap-add
    output = torch.zeros(1, total_length, device=device)
    weights = torch.zeros(1, total_length, device=device)
    
    # Create crossfade window
    fade_len = overlap // 2
    fade_in = torch.linspace(0, 1, fade_len, device=device)
    fade_out = torch.linspace(1, 0, fade_len, device=device)
    
    # Calculate number of chunks
    num_chunks = (total_length - overlap) // hop + 1
    
    print(f"Processing {num_chunks} chunks (chunk_size={chunk_size/48000:.1f}s, overlap={overlap/48000:.1f}s)")
    
    for i in tqdm(range(num_chunks), desc="Processing chunks"):
        start = i * hop
        end = min(start + chunk_size, total_length)
        
        # Adjust start if this is the last chunk and it's too short
        if end - start < chunk_size and i > 0:
            start = max(0, end - chunk_size)
        
        chunk = wav_l[:, start:end]
        band_chunk = band.clone()
        
        # Process chunk
        with torch.no_grad():
            chunk_recon, _ = model.inference(chunk, band_chunk, steps, noise_schedule)
        
        # Build weight window for this chunk
        chunk_len = end - start
        window = torch.ones(chunk_len, device=device)
        
        # Apply fade-in at start (except for first chunk)
        if i > 0 and fade_len <= chunk_len:
            window[:fade_len] = fade_in
        
        # Apply fade-out at end (except for last chunk)
        if end < total_length and fade_len <= chunk_len:
            window[-fade_len:] = fade_out
        
        # Accumulate with weights
        output[:, start:end] += chunk_recon[:, :chunk_len] * window
        weights[:, start:end] += window
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
    
    # Normalize by weights
    output = output / (weights + 1e-8)
    
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help="Checkpoint path")
    parser.add_argument('-i', '--wav', type=str, required=True, help="Input audio file")
    parser.add_argument('--sr', type=int, required=True, help="Sampling rate of input audio (or target SR for --gt)")
    parser.add_argument('--steps', type=int, default=8, help="Steps for sampling")
    parser.add_argument('--gt', action="store_true", help="Input is 48kHz ground truth audio")
    parser.add_argument('--device', type=str, default='cuda', help="Device: 'cuda' or 'cpu'")
    parser.add_argument('--chunk_sec', type=float, default=10.0, help="Chunk size in seconds")
    parser.add_argument('--overlap_sec', type=float, default=2.0, help="Overlap size in seconds")
    parser.add_argument('-o', '--output', type=str, default=None, help="Output file path")
    parser.add_argument('--mono', action="store_true", help="Force mono processing")

    args = parser.parse_args()
    
    # Load config and model
    hparams = OC.load('hparameter.yaml')
    os.makedirs(hparams.log.test_result_dir, exist_ok=True)
    
    if args.steps == 8:
        noise_schedule = eval(hparams.dpm.infer_schedule)
    else:
        noise_schedule = None
    
    print(f"Loading model from {args.checkpoint}...")
    model = NuWave2(hparams).to(args.device)
    model.eval()
    
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
    
    # Calculate filter parameters
    highcut = args.sr // 2
    nyq = 0.5 * hparams.audio.sampling_rate
    hi = highcut / nyq
    
    print(f"Loading audio from {args.wav}...")
    
    # Load audio (preserve stereo if available)
    if args.gt:
        wav_raw, _ = rosa.load(args.wav, sr=hparams.audio.sampling_rate, mono=False)
    else:
        wav_raw, _ = rosa.load(args.wav, sr=args.sr, mono=False)
    
    # Handle mono vs stereo
    if wav_raw.ndim == 1:
        wav_raw = wav_raw[np.newaxis, :]  # [1, T]
        is_stereo = False
    else:
        is_stereo = wav_raw.shape[0] == 2
    
    if args.mono and is_stereo:
        wav_raw = np.mean(wav_raw, axis=0, keepdims=True)
        is_stereo = False
    
    num_channels = wav_raw.shape[0]
    print(f"Audio: {num_channels} channel(s), {'stereo' if is_stereo else 'mono'}")
    
    # Prepare band tensor
    fft_size = hparams.audio.filter_length // 2 + 1
    band = torch.zeros(fft_size, dtype=torch.int64)
    band[:int(hi * fft_size)] = 1
    band = band.unsqueeze(0).to(args.device)
    
    chunk_size = int(args.chunk_sec * hparams.audio.sampling_rate)
    overlap = int(args.overlap_sec * hparams.audio.sampling_rate)
    
    # Process each channel
    recon_channels = []
    
    for ch_idx in range(num_channels):
        ch_name = f"Channel {ch_idx + 1}" if num_channels > 1 else "Audio"
        print(f"\n=== Processing {ch_name} ===")
        
        wav = wav_raw[ch_idx].copy()
        wav /= np.max(np.abs(wav)) + 1e-8
        
        if args.gt:
            wav = wav[:len(wav) - len(wav) % hparams.audio.hop_length]
            
            # Apply lowpass filter
            order = 8
            sos = cheby1(order, 0.05, hi, btype='lowpass', output='sos')
            wav_l = sosfiltfilt(sos, wav)
            
            # Downsample then upsample to simulate low-res
            wav_l = resample_poly(wav_l, highcut * 2, hparams.audio.sampling_rate)
            wav_l = resample_poly(wav_l, hparams.audio.sampling_rate, highcut * 2)
            
            if len(wav_l) < len(wav):
                wav_l = np.pad(wav_l, (0, len(wav) - len(wav_l)), 'constant', constant_values=0)
            elif len(wav_l) > len(wav):
                wav_l = wav_l[:len(wav)]
        else:
            # Upsample to 48kHz
            wav_l = resample_poly(wav, hparams.audio.sampling_rate, args.sr)
            wav_l = wav_l[:len(wav_l) - len(wav_l) % hparams.audio.hop_length]
        
        print(f"Duration: {len(wav_l)/hparams.audio.sampling_rate:.2f} seconds")
        
        wav_l_tensor = torch.from_numpy(wav_l.copy()).float().unsqueeze(0).to(args.device)
        
        # Process with chunking
        wav_recon = process_chunked(
            model, wav_l_tensor, band, args.steps, noise_schedule,
            chunk_size=chunk_size, overlap=overlap, device=args.device
        )
        
        # Clamp and convert to numpy
        wav_recon = torch.clamp(wav_recon, min=-1, max=1 - torch.finfo(torch.float16).eps)
        recon_channels.append(wav_recon[0].detach().cpu().numpy())
    
    # Combine channels
    if is_stereo:
        # Stack as [T, 2] for scipy.io.wavfile
        min_len = min(len(recon_channels[0]), len(recon_channels[1]))
        wav_output = np.stack([recon_channels[0][:min_len], recon_channels[1][:min_len]], axis=-1)
    else:
        wav_output = recon_channels[0]
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.wav))[0]
        output_path = os.path.join(hparams.log.test_result_dir, f'{base_name}_nuwave2.wav')
    
    # Save
    print(f"\nSaving to {output_path}...")
    swrite(output_path, hparams.audio.sampling_rate, wav_output)
    print("Done!")

