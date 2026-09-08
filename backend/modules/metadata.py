import subprocess
import json
import os

# FFprobe path as specified in the environment
FFPROBE_PATH = r"C:\Users\navee\Downloads\ffmpeg-9.0.1-essentials_build\ffmpeg-9.0.1-essentials_build\bin\ffprobe.exe"

def extract_metadata(video_path: str) -> dict:
    """
    Extracts metadata from the video file using ffprobe.
    """
    if not os.path.exists(FFPROBE_PATH):
        # Fallback if the path is wrong, try system ffprobe
        cmd_path = "ffprobe"
    else:
        cmd_path = FFPROBE_PATH
        
    cmd = [
        cmd_path,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(res.stdout)
        
        meta = {
            "duration": None,
            "resolution": None,
            "fps": None,
            "video_codec": None,
            "audio_codec": None,
            "bitrate": None,
            "file_size": None,
            "source": None
        }
        
        if "format" in data:
            d = data["format"].get("duration")
            meta["duration"] = f"{float(d):.2f}s" if d else None
            
            b = data["format"].get("bit_rate")
            meta["bitrate"] = f"{int(b)//1000} kbps" if b else None
            
        if "streams" in data:
            for stream in data["streams"]:
                if stream.get("codec_type") == "video":
                    meta["video_codec"] = stream.get("codec_name")
                    meta["resolution"] = f"{stream.get('width')}x{stream.get('height')}"
                    fps_str = stream.get("r_frame_rate", "0/1")
                    try:
                        num, den = map(int, fps_str.split('/'))
                        meta["fps"] = f"{num/den:.2f}" if den > 0 else None
                    except:
                        pass
                elif stream.get("codec_type") == "audio":
                    meta["audio_codec"] = stream.get("codec_name")
                    
        # Add file size and source
        try:
            size_bytes = os.path.getsize(video_path)
            meta["file_size"] = f"{size_bytes / (1024 * 1024):.2f} MB"
        except:
            meta["file_size"] = "Unknown"
        
        meta["source"] = "Local Upload"
                    
        return meta
    except Exception as e:
        print(f"Metadata extraction failed: {e}")
        return {}
