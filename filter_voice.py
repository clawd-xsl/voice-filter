#!/usr/bin/env python3
"""Filter voice lines from game-extracted audio files using Silero VAD.
Usage: python filter_voice.py <input_dir> <output_dir> [--min-dur 3] [--max-dur 20] [--threshold 0.3]
"""
import argparse, os, shutil, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import subprocess, json

def get_duration(fpath):
    """Get audio duration. Uses wave module for .wav, mutagen or ffprobe as fallback."""
    try:
        ext = Path(fpath).suffix.lower()
        if ext == ".wav":
            import wave
            with wave.open(str(fpath), 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate == 0:
                    return (str(fpath), -1)
                return (str(fpath), frames / rate)
        else:
            # Try mutagen for other formats
            try:
                import mutagen
                m = mutagen.File(str(fpath))
                if m and m.info:
                    return (str(fpath), m.info.length)
            except:
                pass
            # Fallback to ffprobe
            r = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(fpath)],
                capture_output=True, text=True, timeout=30
            )
            info = json.loads(r.stdout)
            return (str(fpath), float(info["format"]["duration"]))
    except:
        return (str(fpath), -1)

def load_wav_as_tensor(fpath):
    """Load wav file using wave module, return (tensor, sample_rate)."""
    import wave, struct, torch
    with wave.open(str(fpath), 'rb') as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        frames = wf.getnframes()
        data = wf.readframes(frames)
    
    if sw == 2:
        samples = torch.frombuffer(bytearray(data), dtype=torch.int16).float() / 32768.0
    elif sw == 4:
        samples = torch.frombuffer(bytearray(data), dtype=torch.int32).float() / 2147483648.0
    elif sw == 1:
        samples = torch.frombuffer(bytearray(data), dtype=torch.uint8).float() / 128.0 - 1.0
    else:
        raise ValueError(f"Unsupported sample width: {sw}")
    
    if ch > 1:
        samples = samples.view(-1, ch).mean(dim=1)
    
    return samples, sr

def resample_simple(waveform, orig_sr, target_sr):
    """Simple linear interpolation resampling (no torchaudio needed)."""
    import torch
    if orig_sr == target_sr:
        return waveform
    ratio = target_sr / orig_sr
    new_len = int(len(waveform) * ratio)
    indices = torch.arange(new_len, dtype=torch.float32) / ratio
    indices = indices.clamp(0, len(waveform) - 1)
    floor_idx = indices.long()
    frac = indices - floor_idx.float()
    ceil_idx = (floor_idx + 1).clamp(max=len(waveform) - 1)
    return waveform[floor_idx] * (1 - frac) + waveform[ceil_idx] * frac

def check_voice_batch(file_list, threshold, hub_dir=None):
    """Check a batch of files for voice using Silero VAD. Loads model once per process."""
    import torch
    
    # Load from local cache to avoid multi-process conflicts
    if hub_dir:
        model, utils = torch.hub.load(hub_dir, 'silero_vad', source='local', trust_repo=True)
    else:
        model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
    get_speech_timestamps = utils[0]
    
    results = []
    for fpath in file_list:
        try:
            samples, sr = load_wav_as_tensor(fpath)
            if sr != 16000:
                samples = resample_simple(samples, sr, 16000)
            
            speech_timestamps = get_speech_timestamps(samples, model, sampling_rate=16000)
            
            if not speech_timestamps:
                results.append((str(fpath), 0.0))
                continue
            
            total_samples = len(samples)
            speech_samples = sum(t['end'] - t['start'] for t in speech_timestamps)
            ratio = speech_samples / total_samples
            results.append((str(fpath), ratio))
        except Exception as e:
            results.append((str(fpath), -1))
    
    return results

STAGE_FILE = "filter_voice_stage.json"

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
    parser.add_argument("--resume", action="store_true", help="Resume from stage file (skip duration check)")
    parser.add_argument("--debug-dur", action="store_true", help="Print duration distribution and exit")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    exts = set(args.extensions.split(","))

    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist"); sys.exit(1)

    # Resume from stage file
    if args.resume:
        if not Path(STAGE_FILE).exists():
            print(f"Error: {STAGE_FILE} not found. Run without --resume first."); sys.exit(1)
        with open(STAGE_FILE) as f:
            stage = json.load(f)
        candidates = [Path(p) for p in stage["candidates"]]
        print(f"Resumed {len(candidates)} candidates from {STAGE_FILE}")
    else:
        # Step 1: Collect files
        print("Scanning files...")
        all_files = [f for f in input_dir.rglob("*") if f.suffix.lower() in exts]
        total = len(all_files)
        print(f"Found {total} audio files")

        # Duration check (parallel)
        print(f"\n[Step 1/2] Checking durations ({args.min_dur}s - {args.max_dur}s)...")
        durations = {}
        candidates = []
        t0 = time.time()
        done = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(get_duration, f): f for f in all_files}
            for future in as_completed(futures):
                fpath, dur = future.result()
                done += 1
                durations[fpath] = dur
                if dur < 0:
                    failed += 1
                elif args.min_dur <= dur <= args.max_dur:
                    candidates.append(Path(fpath))
                if done % 500 == 0 or done == total:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    print(f"  {done}/{total} ({done*100//total}%) | {rate:.0f} files/s | ETA {eta:.0f}s | failed: {failed}", flush=True)
        
        elapsed = time.time() - t0
        print(f"  Duration filter: {total} -> {len(candidates)} files ({elapsed:.1f}s, {failed} failed)")

        # Debug: show duration distribution
        if args.debug_dur or len(candidates) == 0:
            valid_durs = [d for d in durations.values() if d >= 0]
            if valid_durs:
                import statistics
                print(f"\n  Duration stats ({len(valid_durs)} valid files):")
                print(f"    Min: {min(valid_durs):.2f}s")
                print(f"    Max: {max(valid_durs):.2f}s")
                print(f"    Mean: {statistics.mean(valid_durs):.2f}s")
                print(f"    Median: {statistics.median(valid_durs):.2f}s")
                # Histogram
                buckets = [0]*10
                labels = ["<0.5s","0.5-1s","1-2s","2-3s","3-5s","5-10s","10-20s","20-60s","60-300s",">300s"]
                bounds = [0,0.5,1,2,3,5,10,20,60,300,float('inf')]
                for d in valid_durs:
                    for i in range(len(bounds)-1):
                        if bounds[i] <= d < bounds[i+1]:
                            buckets[i] += 1
                            break
                print(f"    Distribution:")
                for label, count in zip(labels, buckets):
                    bar = "#" * min(50, count * 50 // max(1, max(buckets)))
                    print(f"      {label:>8}: {count:>6} {bar}")
            else:
                print(f"\n  WARNING: All {total} files failed ffprobe! Check if ffprobe is in PATH.")
                # Show sample errors
                sample = list(durations.keys())[:3]
                for s in sample:
                    r = subprocess.run(
                        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", s],
                        capture_output=True, text=True, timeout=30
                    )
                    print(f"    {Path(s).name}: stdout={r.stdout[:200]} stderr={r.stderr[:200]}")
            
            if args.debug_dur:
                sys.exit(0)

        # Save stage
        with open(STAGE_FILE, "w") as f:
            json.dump({"candidates": [str(p) for p in candidates], "durations": {k: v for k, v in durations.items() if args.min_dur <= v <= args.max_dur}}, f)
        print(f"  Saved stage to {STAGE_FILE}")

    # Step 2: VAD check
    if not candidates:
        print("\nNo candidates to run VAD on. Use --debug-dur to check duration distribution.")
        sys.exit(0)

    print(f"\n[Step 2/2] Running VAD on {len(candidates)} candidates...")
    
    # Pre-download model in main process to avoid multi-process cache conflicts
    import torch
    print("  Loading Silero VAD model...", flush=True)
    hub_dir = torch.hub.get_dir()
    _ = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
    hub_cache = Path(hub_dir)
    silero_dirs = sorted(hub_cache.glob("snakers4_silero-vad*"))
    silero_local = str(silero_dirs[0]) if silero_dirs else None
    print(f"  Model cached at: {silero_local}", flush=True)
    # Quick sanity check on first candidate
    try:
        samples, sr = load_wav_as_tensor(str(candidates[0]))
        print(f"  Sanity check: {candidates[0].name} samples={len(samples)} sr={sr} min={samples.min():.4f} max={samples.max():.4f}", flush=True)
    except Exception as e:
        print(f"  WARNING: Could not read {candidates[0].name}: {e}", flush=True)
    del _
    
    voice_files = []
    t0 = time.time()
    
    chunk_size = max(1, len(candidates) // args.workers)
    chunks = [candidates[i:i+chunk_size] for i in range(0, len(candidates), chunk_size)]
    
    done = 0
    with ProcessPoolExecutor(max_workers=min(args.workers, len(chunks))) as pool:
        futures = {pool.submit(check_voice_batch, chunk, args.threshold, silero_local): i for i, chunk in enumerate(chunks)}
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

    # Step 3: Copy results
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
