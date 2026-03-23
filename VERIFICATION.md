# Audio Transcription Verification Guide

This guide documents how to verify that all audio files have been successfully transcribed.

## Quick Verification Commands

### 1. Count Input vs Output Files

```bash
# Count input audio files
find /root/Audio_Transcriptions -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.mp4" \) | wc -l

# Count output .txt files
find /root/Audio_Transcriptions_Transcripts -type f -name "*.txt" | wc -l

# Count output .json files
find /root/Audio_Transcriptions_Transcripts -type f -name "*.json" | wc -l
```

Expected output: All three counts should be **equal** (e.g., 78).

### 2. Check for Empty or Small Files

```bash
# Check for empty files (0 bytes)
find /root/Audio_Transcriptions_Transcripts -type f \( -name "*.txt" -o -name "*.json" \) -size 0

# Check for files under 1KB (potential issues)
find /root/Audio_Transcriptions_Transcripts -type f \( -name "*.txt" -o -name "*.json" \) -size -1k

# Check for files under 10KB (may be incomplete)
find /root/Audio_Transcriptions_Transcripts -type f -name "*.txt" -size -10k
```

Expected: No output (no issues found).

### 3. Verify JSON Structure

Check that JSON files contain proper transcription data:

```bash
# Find a sample JSON file and check its content
find /root/Audio_Transcriptions_Transcripts -name "*.json" -type f | head -1 | xargs python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(f'Text: {len(d.get(\"text\",\"\"))} chars, Segments: {len(d.get(\"segments\",[]))}')" 
```

Expected: Text should have thousands of characters, segments should be in hundreds.

### 4. Check for Errors in Log

```bash
# Check for failed transcriptions
grep -i "fail\|error" /root/Audio_Transcriptions_Transcripts/transcription.log | grep -i "transcrib" | tail -20

# Check final summary
grep "Transcription complete" /root/Audio_Transcriptions_Transcripts/transcription.log
```

Expected: Should show "78 successful, 0 failed" (or matching count).

---

## Complete Verification Script

Run this single command for full verification:

```bash
echo "=== AUDIO TRANSCRIPTION VERIFICATION ===" && \
echo "" && \
echo "Input files:" && \
find /root/Audio_Transcriptions -type f \( -name "*.wav" -o -name "*.mp3" \) | wc -l && \
echo "" && \
echo "Output .txt files:" && \
find /root/Audio_Transcriptions_Transcripts -type f -name "*.txt" | wc -l && \
echo "" && \
echo "Output .json files:" && \
find /root/Audio_Transcriptions_Transcripts -type f -name "*.json" | wc -l && \
echo "" && \
echo "Empty files:" && \
find /root/Audio_Transcriptions_Transcripts -type f \( -name "*.txt" -o -name "*.json" \) -size 0 | wc -l && \
echo "" && \
echo "Files under 1KB:" && \
find /root/Audio_Transcriptions_Transcripts -type f \( -name "*.txt" -o -name "*.json" \) -size -1k | wc -l && \
echo "" && \
echo "=== Final Status ===" && \
grep "Transcription complete" /root/Audio_Transcriptions_Transcripts/transcription.log
```

---

## Verification Checklist

- [ ] Input audio count = Output .txt count
- [ ] Input audio count = Output .json count
- [ ] No empty files (0 bytes)
- [ ] No files under 1KB
- [ ] JSON files contain text and segments
- [ ] Log shows all files successful
- [ ] Log shows 0 failed

---

## Troubleshooting

### If counts don't match

1. Check which files are missing:
   ```bash
   # Find audio files without .txt output
   find /root/Audio_Transcriptions -name "*.wav" | while read f; do
     base=$(basename "$f" .wav)
     dir=$(dirname "$f" | sed 's|/root/Audio_Transcriptions|/root/Audio_Transcriptions_Transcripts|')
     if [ ! -f "$dir/$base.txt" ]; then
       echo "Missing: $f"
     fi
   done
   ```

2. Re-run transcription for missing files:
   ```bash
   # Delete .txt and .json for failed files, then re-run
   uv run main.py /root/Audio_Transcriptions
   ```

### If files are too small

Check the source audio file size. Small transcripts may indicate:
- Corrupted source audio
- API errors during transcription
- Very short audio content

Check log for specific errors:
```bash
grep -B5 "failed\|error" /root/Audio_Transcriptions_Transcripts/transcription.log | tail -30
```

---

## Directory Structure Reference

```
/root/Audio_Transcriptions/              # Source audio files
├── 01 Beginners/
│   └── 01 Introduction.wav
├── 06 Fibonacci/
│   └── 01 Fibonacci Retracements.wav
... (12 subdirectories, 78 filesudio_Transcriptions)

/root/A_Transcripts/   # Output transcripts
├── 01 Beginners/
│   ├── 01 Introduction.txt
│   └── 01 Introduction.json
├── 06 Fibonacci/
│   ├── 01 Fibonacci Retracements.txt
│   └── 01 Fibonacci Retracements.json
... (matching structure)
└── transcription.log
```
