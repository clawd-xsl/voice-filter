#!/usr/bin/env python3
"""Filter voice lines from game-extracted audio files using Silero VAD.
Usage: python filter_voice.py <input_dir> <output_dir> [--min-dur 0.5] [--max-dur 20] [--threshold 0.5]
"""
import argparse, os, shutil, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess, json, tempfile

def get_duration(fpath):
    """Get audio duration via ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(fpath)],
            capture_output=True, text=True, timeout=10
        )
        info = json.loads(r.stdout)
        return float(info["format"]["duration"])
    except:
        return -1

def check_voice(args):
    """Check if file contains voice using Silero VAD. Returns (path, score)."""
    fpath, threshold = args
    try:
        import torch
        import torchaudio

        # Convert to 16kHz mono wav
        waveform, sr = torchaudio.load(str(fpath))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        
        # Load VAD model
        model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
        get_speech_timestamps, _, _, _, _ = utils
        
        speech_timestamps = get_speech_timestamps(waveform.squeeze(), model, sampling_rate=16000)
        
        if not speech_timestamps:
            return (str(fpath), 0.0)
        
        # Calculate speech ratio
        total_samples = waveform.shape[1]
        speech_samples = sum(t['end'] - t['start'] for t in speech_timestamps)
        ratio = speech_samples / total_samples
        
        return (str(fpath), ratio)
    except Exception as e:
        return (str(fpath), -1)

def main():
    parser = argparse.ArgumentParser(description="Filter voice lines from game audio")
    parser.add_argument("input_dir", help="Directory with audio files")
    parser.add_argument("output_dir", help="Directory to copy voice files to")
    parser.add_argument("--min-dur", type=float, default=3, help="Min duration in seconds (default: 3)")
    parser.add_argument("--max-dur", type=float, default=20, help="Max duration in seconds (default: 20)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Min speech ratio to count as voice (default: 0.3)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    parser.add_argument("--extensions", default=".wav,.ogg,.mp3,.flac,.wem,.opus", help="Audio extensions")
    parser.add_argument("--dry-run", action="store_true", help="Don't copy, just print results")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    exts = set(args.extensions.split(","))

    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist"); sys.exit(1)

    # Step 1: Collect files
    print("Scanning files...")
    all_files = [f for f in input_dir.rglob("*") if f.suffix.lower() in exts]
    print(f"Found {len(all_files)} audio files")

    # Step 2: Filter by duration
    print(f"Filtering by duration ({args.min_dur}s - {args.max_dur}s)...")
    candidates = []
    for i, f in enumerate(all_files):
        if i % 1000 == 0:
            print(f"  Duration check: {i}/{len(all_files)}")
        dur = get_duration(f)
        if args.min_dur <= dur <= args.max_dur:
            candidates.append(f)
    
    print(f"Duration filter: {len(all_files)} -> {len(candidates)} files")

    # Step 3: VAD check
    print(f"Running VAD on {len(candidates)} candidates...")
    voice_files = []
    
    # Process in batches to show progress
    batch_size = 100
    for batch_start in range(0, len(candidates), batch_size):
        batch = candidates[batch_start:batch_start+batch_size]
        print(f"  VAD: {batch_start}/{len(candidates)}")
        
        for f in batch:
            _, score = check_voice((f, args.threshold))
            if score >= args.threshold:
                voice_files.append((f, score))

    print(f"VAD filter: {len(candidates)} -> {len(voice_files)} voice files")

    # Step 4: Copy results
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        for f, score in voice_files:
            dest = output_dir / f.name
            # Handle duplicate names
            if dest.exists():
                dest = output_dir / f"{f.stem}_{hash(str(f)) % 10000}{f.suffix}"
            shutil.copy2(f, dest)
        print(f"Copied {len(voice_files)} files to {output_dir}")
    else:
        for f, score in voice_files[:20]:
            print(f"  {f.name} (speech: {score:.1%})")
        if len(voice_files) > 20:
            print(f"  ... and {len(voice_files)-20} more")

if __name__ == "__main__":
    main()
