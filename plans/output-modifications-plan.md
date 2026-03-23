# Plan: Output File Modifications

## Overview

Modify `main.py` to implement four changes:
1. **Remove `.txt` output** - Keep only JSON output files
2. **Add file metadata to JSON** - Include video/audio file metadata from ffprobe
3. **Remove tokens from JSON** - Strip token arrays from segment data
4. **Remove `text` field from JSON** - The `text` field is redundant (concatenation of segment texts)

---

## Current Behavior

### Output Generation (lines 448-457 in `process_file()`)
```python
# Save JSON with full metadata
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

# Save plain text
text = merged.get("text", "")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(text)

logger.info(f"Saved: {txt_path} and {json_path}")
```

### JSON Output Structure (from Groq API verbose_json)
```json
{
  "text": "Transcribed text...",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.0,
      "text": "Hello world",
      "tokens": [50364, 1124, 507, 1027, 28645],
      "temperature": 0.0,
      "avg_logprob": -0.5,
      "compression_ratio": 0.7,
      "no_speech_prob": 0.01
    }
  ]
}
```

---

## Implementation Steps

### Step 1: Remove `.txt` File Generation

**Files to modify:** `main.py`

**Changes:**
1. Delete `txt_path` variable on line 397 (it's unused after this change)
2. Remove `get_output_paths()` function or simplify it to return only `json_path`
3. Remove lines 452-455 that write the `.txt` file
4. Update line 457 logging from `"Saved: {txt_path} and {json_path}"` to `"Saved: {json_path}"`

**Simplified `get_output_paths()` (or remove entirely):**
```python
def get_output_paths(audio_rel_path: Path, output_dir: Path) -> Path:
    """Get output .json path for an audio file."""
    return output_dir / audio_rel_path.with_suffix(".json")
```

---

### Step 2: Add File Metadata to JSON Output

**Files to modify:** `main.py`

**New Function - `get_file_metadata()`**
Add a new function to extract comprehensive metadata using ffprobe:

```python
def get_file_metadata(file_path: Path) -> Dict[str, Any]:
    """Get comprehensive file metadata using ffprobe."""
    import json
    
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(file_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)
```

**Metadata to include in output JSON:**
- `file_metadata.filename` - Original filename
- `file_metadata.path` - Full path to source file
- `file_metadata.size_bytes` - File size in bytes
- `file_metadata.duration` - Duration in seconds (already calculated, reuse)
- `file_metadata.format` - Container format (mp4, mp3, etc.)
- `file_metadata.bit_rate` - Bitrate in bps
- `file_metadata.codec` - Audio codec name
- `file_metadata.sample_rate` - Sample rate (Hz)
- `file_metadata.channels` - Number of channels

**Changes in `process_file()` (lines 389-465):**
1. Call `get_file_metadata(audio_path)` after line 405
2. Build output structure that includes both `file_metadata` and `transcription`

**Modified output structure:**
```json
{
  "file_metadata": {
    "filename": "video.mp4",
    "path": "/path/to/video.mp4",
    "size_bytes": 1048576,
    "duration": 120.5,
    "format": "mp4",
    "bit_rate": 128000,
    "codec": "aac",
    "sample_rate": 44100,
    "channels": 2
  },
  "segments": [...]
}
```

---

### Step 3: Remove Tokens from JSON Segments

**Files to modify:** `main.py`

**New Function - `strip_metadata_fields()`**
Add a function to clean up API response by removing unnecessary fields:

```python
def strip_metadata_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    """Remove tokens and text field from transcription result."""
    cleaned = result.copy()
    # Remove tokens from each segment
    if "segments" in cleaned:
        cleaned["segments"] = [
            {k: v for k, v in seg.items() if k != "tokens"}
            for seg in cleaned["segments"]
        ]
    # Remove the text field (redundant - full text is concatenation of segment texts)
    cleaned.pop("text", None)
    return cleaned
```

**Changes in `process_file()`:**
1. Call `strip_metadata_fields(merged)` before saving to JSON
2. This should be applied to both `merged` (standard transcription) and after chunk merging

---

## Summary of Changes by Function

| Function | Lines | Change |
|----------|-------|--------|
| `get_output_paths()` | 382-386 | Simplify to return only JSON path |
| `get_file_metadata()` | NEW | Add new function for ffprobe metadata |
| `strip_metadata_fields()` | NEW | Add new function to remove tokens and text |
| `process_file()` | 389-465 | Remove txt save, add metadata, strip tokens/text |

---

## Files to Modify

- `main.py` - Primary implementation file

---

## Testing Considerations

1. Verify JSON output contains `file_metadata` section
2. Verify `tokens` field is absent from all segments
3. Verify `text` field is absent from root level
4. Verify no `.txt` files are created
5. Verify existing `.json` files are not overwritten (check logic on line 400)
