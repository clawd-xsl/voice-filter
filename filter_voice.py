#!/usr/bin/env python3
"""Filter voice lines from game-extracted audio files using Silero VAD.
Usage: python filter_voice.py <input_dir> <output_dir> [--min-dur 3] [--max-dur 20] [--threshold 0.3]
"""
import argparse, os, shutil, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import subprocess, json

def get_duration(fpath):
    """Get audio duration via ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(fpath)],
            capture_output=True, text=True, timeout=30
        )
        info = json.loads(r.stdout)
        return (str(fpath), float(info["format"]["duration"]))
    except:
        return (str(fpath), -1)

def check_voice_batch(file_list, threshold):
    """Check a batch of files for voice using Silero VAD. Loads model once per process."""
    import torch
    import torchaudio
    
    model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
    get_speech_timestamps = utils[0]
    
    results = []
    for fpath in file_list:
        try:
            waveform, sr = torchaudio.load(str(fpath))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            
            speech_timestamps = get_speech_timestamps(waveform.squeeze(), model, sampling_rate=16000)
            
            if not speech_timestamps:
                results.append((str(fpath), 0.0))
                continue
            
            total_samples = waveform.shape[1]
            speech_samples = sum(t['end'] - t['start'] for t in speech_timestamps)
            ratio = speech_samples / total_samples
            results.append((str(fpath), ratio))
        except Exception as e:
            results.append((str(fpath), -1))
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Filter voice lines from game audio")
    parser.add_argument("input_dir", help="Directory with audio files")
    parser.add_argument("output_dir", help="Directory to copy voice files to")
    parser.add_argument("--min-dur", type=float, default=3, help="Min duration in seconds (default: 3)")
    parser.add_argument("--max-dur", type=float, default=20, help="Max duration in seconds (default: 20)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Min speech ratio to count as voice (default: 0.3)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers (default: 8)")
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
    total = len(all_files)
    print(f"Found {total} audio files")

    # Step 2: Filter by duration (parallel ffprobe)
    print(f"\n[Step 1/2] Filtering by duration ({args.min_dur}s - {args.max_dur}s)...")
    candidates = []
    t0 = time.time()
    done = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(get_duration, f): f for f in all_files}
        for future in as_completed(futures):
            fpath, dur = future.result()
            done += 1
            if done % 500 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  {done}/{total} ({done*100//total}%) | {rate:.0f} files/s | ETA {eta:.0f}s", flush=True)
            if args.min_dur <= dur <= args.max_dur:
                candidates.append(Path(fpath))
    
    print(f"  Duration filter: {total} -> {len(candidates)} files ({time.time()-t0:.1f}s)")

    # Step 3: VAD check (parallel, model loaded once per worker)
    print(f"\n[Step 2/2] Running VAD on {len(candidates)} candidates...")
    voice_files = []
    t0 = time.time()
    
    # Split into chunks for each worker
    chunk_size = max(1, len(candidates) // args.workers)
    chunks = [candidates[i:i+chunk_size] for i in range(0, len(candidates), chunk_size)]
    
    done = 0
    with ProcessPoolExecutor(max_workers=min(args.workers, len(chunks))) as pool:
        futures = {pool.submit(check_voice_batch, chunk, args.threshold): i for i, chunk in enumerate(chunks)}
        for future in as_completed(futures):
            results = future.result()
            for fpath, score in results:
                if score >= args.threshold:
                    voice_files.append((Path(fpath), score))
            done += len(results)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(candidates) - done) / rate if rate > 0 else 0
            print(f"  {done}/{len(candidates)} ({done*100//max(1,len(candidates))}%) | {rate:.0f} files/s | ETA {eta:.0f}s", flush=True)

    print(f"  VAD filter: {len(candidates)} -> {len(voice_files)} voice files ({time.time()-t0:.1f}s)")

    # Step 4: Copy results
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nCopying {len(voice_files)} files...")
        for i, (f, score) in enumerate(voice_files):
            dest = output_dir / f.name
            if dest.exists():
                dest = output_dir / f"{f.stem}_{hash(str(f)) % 10000}{f.suffix}"
            shutil.copy2(f, dest)
            if (i+1) % 100 == 0:
                print(f"  Copied {i+1}/{len(voice_files)}", flush=True)
        print(f"Done! Copied {len(voice_files)} files to {output_dir}")
    else:
        print(f"\n[Dry run] Top matches:")
        voice_files.sort(key=lambda x: x[1], reverse=True)
        for f, score in voice_files[:30]:
            print(f"  {f.name} (speech: {score:.1%})")
        if len(voice_files) > 30:
            print(f"  ... and {len(voice_files)-30} more")

if __name__ == "__main__":
    main()
