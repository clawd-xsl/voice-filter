# voice-filter

Filter character voice lines from game-extracted audio files (e.g. Zenless Zone Zero).

Uses duration filtering + [Silero VAD](https://github.com/snakers4/silero-vad) to separate voice lines from music/SFX.

## Install

```bash
pip install torch torchaudio
```

## Usage

```bash
# Dry run (preview results, don't copy)
python filter_voice.py /path/to/extracted /path/to/output --dry-run

# Run for real
python filter_voice.py /path/to/extracted /path/to/voice_output

# Custom settings
python filter_voice.py ./audio ./voices --min-dur 3 --max-dur 20 --threshold 0.3
```

## How it works

1. Scan all audio files (wav/ogg/mp3/flac/wem/opus)
2. Filter by duration (default 3-20s) — removes music (too long) and SFX (too short)
3. Run Silero VAD on remaining files — keeps only files with >30% speech content
4. Copy matching files to output directory

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--min-dur` | 3 | Minimum duration in seconds |
| `--max-dur` | 20 | Maximum duration in seconds |
| `--threshold` | 0.3 | Minimum speech ratio (0-1) |
| `--workers` | 4 | Parallel workers |
| `--extensions` | .wav,.ogg,.mp3,.flac,.wem,.opus | Audio file extensions |
| `--dry-run` | false | Preview results without copying |

## Note

`.wem` files (Wwise) need to be converted first using [vgmstream](https://github.com/vgmstream/vgmstream) before processing.
