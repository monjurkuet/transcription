# Audio Transcription Script

Transcribe audio files using Groq Whisper API with automatic chunking for large files, round-robin API key rotation, and quota monitoring.

## Features

- **Round-Robin API Keys**: Automatically rotates through multiple API keys for higher throughput
- **Automatic Chunking**: Splits large audio files into 10-minute chunks with 5-second overlap
- **Quota Monitoring**: CLI tool to check remaining API quota
- **Timestamp Support**: Outputs both `.txt` (plain text) and `.json` (with segment timestamps)
- **Progress Tracking**: Visual progress bar with file-level and chunk-level tracking
- **Resume Support**: Skips already-transcribed files (checks for existing `.json`)
- **Rate Limiting**: Automatic retry with exponential backoff for API rate limits
- **Directory Mirroring**: Preserves source directory structure in output

## Requirements

### System Dependencies

- **Python 3.10+**
- **ffmpeg** (for audio chunking)

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Python Dependencies

Install via uv (recommended):

```bash
uv venv && uv pip install -r requirements.txt
```

Or with pip:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Transcribe single audio file
uv run main.py /path/to/audio.mp3

# Transcribe all files in directory
uv run main.py /path/to/audio/files

# Specify custom output directory
uv run main.py /path/to/audio/files --output /path/to/output
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `source` | Source file or directory containing audio files | (required) |
| `--output` | Output directory | `{source}_Transcript` or `{source}_Transcripts` |
| `--model` | Whisper model to use | `whisper-large-v3` |
| `--chunk-duration` | Chunk duration in seconds | `600` (10 min) |
| `--chunk-overlap` | Chunk overlap in seconds | `5` |

### Example Commands

```bash
# Transcribe single file
uv run main.py /root/test.mp3

# Transcribe all files in directory
uv run main.py /root/Audio_Transcriptions

# With custom output path
uv run main.py /root/Audio_Transcriptions --output /root/MyTranscripts

# Resume interrupted transcription (skips existing files)
uv run main.py /root/Audio_Transcriptions
```

## Quota Checker CLI

Check remaining API quota for your configured keys:

```bash
# Check quota status (text)
uv run python check_quota.py

# Check quota with JSON output
uv run python check_quota.py --json

# List configured API keys
uv run python check_quota.py --list-keys

# Force fresh quota check via API call
uv run python check_quota.py --check-headers
```

### Quota Output Example

```
============================================================
Groq API Quota Status
============================================================
Checked at: 2026-03-23T00:00:00+00:00
Total keys: 3

Key 1: gsk_xxx***abc123
  Requests: 950 / 1000 (daily limit) [95.0%]
  Resets: in 2h 30m
  Tokens: 11000 / 12000 (per minute)

Key 2: gsk_xxx***def456
  Requests: 980 / 1000 (daily limit) [98.0%]
  Resets: in 2h 30m
  Tokens: 11500 / 12000 (per minute)
```

## Configuration

### Environment Variables

Create a `.env` file in the project directory with your API keys:

```env
# Multiple keys (comma-separated) - recommended for higher throughput
GROQ_API_KEYS=gsk_key1,gsk_key2,gsk_key3

# Or single key (backward compatible)
GROQ_API_KEY=your_api_key_here
```

Get your API keys from: https://console.groq.com/keys

### How Round-Robin Works

1. All configured API keys are loaded at startup
2. Keys are rotated in sequence for each request
3. If a 429 (rate limit) response is received:
   - The key is marked as temporarily exhausted
   - The system automatically switches to the next available key
   - After the cooldown period, the key becomes available again
4. This maximizes throughput by keeping all keys active

## Supported Audio Formats

- `.wav` (recommended for chunking)
- `.mp3`
- `.mp4`
- `.flac`
- `.m4a`
- `.ogg`
- `.webm`
- `.mpeg`
- `.mpga`

## Groq API Limits

| Tier | Max File Size | Notes |
|------|---------------|-------|
| Free | 25 MB | Default |
| Dev | 100 MB | Requires billing |

The script automatically chunks files exceeding 25MB (configurable via `MAX_FILE_SIZE_MB`).

## Output Structure

### Single File
```
/root/test.mp3 → /root/test_Transcript/test.txt, test.json
```

### Directory
```
/path/to/Audio_Transcriptions/
├── 01 Beginners/
│   └── 01 Introduction.wav
├── 06 Fibonacci/
│   └── 01 Fibonacci Retracements.wav
└── ...

Output (/path/to/Audio_Transcriptions_Transcripts/):
├── 01 Beginners/
│   ├── 01 Introduction.txt
│   └── 01 Introduction.json
├── 06 Fibonacci/
│   ├── 01 Fibonacci Retracements.txt
│   └── 01 Fibonacci Retracements.json
└── ...
```

### Output Files

- **`.txt`**: Plain text transcription
- **`.json`**: Full metadata with segment timestamps

### JSON Output Format

```json
{
  "text": "Transcribed text content...",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 5.0,
      "text": " Hello everybody, hope that you're doing very well",
      "tokens": [50364, 479, 1277, ...],
      "temperature": 0.0,
      "avg_logprob": -0.15,
      "compression_ratio": 1.5,
      "no_speech_prob": 0.01
    }
  ]
}
```

## Chunking Configuration

### How Chunking Works

1. Audio files > 25MB are split into 10-minute segments
2. Each chunk has 5-second overlap with adjacent chunks
3. Each chunk is transcribed individually
4. Results are merged with adjusted timestamps
5. Duplicate content at chunk boundaries is removed

### Adjusting Chunk Settings

Edit `main.py` constants:

```python
CHUNK_DURATION_SEC = 600    # 10 minutes (increase for fewer API calls)
CHUNK_OVERLAP_SEC = 5        # 5 seconds (increase for better boundary handling)
MAX_FILE_SIZE_MB = 25        # Chunk files larger than this
```

Or use CLI flags:

```bash
uv run main.py /path/to/audio --chunk-duration 900 --chunk-overlap 10
```

## Troubleshooting

### "Organization has blocked API access because a spend alert threshold was met"

Your API key has reached its spending limit. Visit https://console.groq.com/settings/billing to manage alerts or add payment method.

### "Request Entity Too Large"

Your files exceed the API limit. The script should automatically chunk, but you may need to lower `MAX_FILE_SIZE_MB` or preprocess files with ffmpeg:

```bash
# Compress audio before transcription
ffmpeg -i input.wav -ar 16000 -ac 1 output.wav
```

### "file is empty" Error

Usually caused by corrupted or empty audio files. Check file integrity:

```bash
ffprobe -v error -show_entries format=duration input.wav
```

### Resume Interrupted Transcription

The script automatically skips files that already have `.json` output. To re-transcribe specific files, delete their `.txt` and `.json` outputs first.

## Logging

Logs are saved to `transcription.log` in the output directory. To change log location or level, edit `setup_logging()` in `main.py`.

## Cost Estimation

Using `whisper-large-v3`:

- **$0.111 per minute** of audio
- 10-minute chunk = ~$1.11 per chunk
- Average 90-minute file = ~10 chunks = ~$11

Using `whisper-large-v3-turbo` (faster, cheaper):

- **$0.04 per minute** of audio
- 10-minute chunk = ~$0.40 per chunk
- Average 90-minute file = ~10 chunks = ~$4

To switch models, edit `MODEL` in `main.py` or add `--model whisper-large-v3-turbo` to CLI.

## License

MIT License
