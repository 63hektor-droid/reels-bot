import os
import json
import subprocess
import requests
from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator

TOKEN = os.environ["BOT_TOKEN"]
CHANNEL = "@my_reels_fa"

# اکانت‌های اینستاگرامی که ویدیوهاشون بررسی میشه (خودت عوض کن)
ACCOUNTS = [
    "nasa",
    "natgeo",
    "9gag",
]

MAX_DURATION = 90  # ثانیه
POSTED_FILE = "posted.json"


def load_posted():
    if os.path.exists(POSTED_FILE):
        with open(POSTED_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_posted(posted):
    with open(POSTED_FILE, "w", encoding="utf-8") as f:
        json.dump(posted[-500:], f)


def ydl_opts(**extra):
    opts = {"quiet": False, "no_warnings": True, "ignoreerrors": False}
    if os.environ.get("IG_COOKIES"):
        with open("cookies.txt", "w", encoding="utf-8") as f:
            f.write(os.environ["IG_COOKIES"])
        opts["cookiefile"] = "cookies.txt"
    opts.update(extra)
    return opts


def pick_video(posted):
    best = None
    with YoutubeDL(ydl_opts(skip_download=True, playlistend=8)) as y:
        for acc in ACCOUNTS:
            try:
                info = y.extract_info(
                    f"https://www.instagram.com/{acc}/reels/", download=False
                )
            except Exception as e:
                print("skip", acc, e)
                continue
            for e in (info or {}).get("entries") or []:
                if not e or e.get("id") in posted:
                    continue
                if (e.get("duration") or 0) > MAX_DURATION:
                    continue
                if best is None or (e.get("view_count") or 0) > (
                    best.get("view_count") or 0
                ):
                    best = e
    return best


def download(entry):
    url = entry.get("webpage_url") or entry.get("url")
    with YoutubeDL(ydl_opts(outtmpl="video.%(ext)s", format="mp4/best")) as y:
        y.download([url])
    return "video.mp4"


def ts(sec):
    h = int(sec // 3600)
    m = int(sec % 3600 // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{int(s):02d},{int((s - int(s)) * 1000):03d}"


def make_srt(video):
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(video, vad_filter=True)
    tr = GoogleTranslator(source="auto", target="fa")
    lines = []
    n = 0
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        try:
            fa = tr.translate(text)
        except Exception as e:
            print("translate fail", e)
            continue
        n += 1
        lines.append(f"{n}\n{ts(seg.start)} --> {ts(seg.end)}\n{fa}\n")
    if n == 0:
        return False
    with open("sub.srt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return True


def burn(video):
    style = (
        "FontName=Noto Sans Arabic,FontSize=16,Outline=2,Shadow=0,"
        "Alignment=2,MarginV=60,PrimaryColour=&H00FFFFFF"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", video,
            "-vf", f"subtitles=sub.srt:force_style='{style}'",
            "-c:a", "copy", "out.mp4",
        ],
        check=True,
    )
    return "out.mp4"


def send(path):
    with open(path, "rb") as f:
        r = requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendVideo",
            data={"chat_id": CHANNEL, "supports_streaming": "true"},
            files={"video": f},
            timeout=300,
        )
    print(r.status_code, r.text[:200])
    r.raise_for_status()


def main():
    posted = load_posted()
    entry = pick_video(posted)
    if not entry:
        print("no video found")
        return
    vid = entry["id"]
    try:
        video = download(entry)
        if make_srt(video):
            send(burn(video))
        else:
            print("no speech, skipped")
    finally:
        posted.append(vid)
        save_posted(posted)


if __name__ == "__main__":
    import io, sys, contextlib, traceback

    buf = io.StringIO()

    class Tee:
        encoding = "utf-8"

        def write(self, s):
            buf.write(s)
            sys.__stdout__.write(s)

        def flush(self):
            pass

        def isatty(self):
            return False

    try:
        with contextlib.redirect_stdout(Tee()), contextlib.redirect_stderr(Tee()):
            main()
    except Exception:
        buf.write(traceback.format_exc())
    finally:
        text = buf.getvalue().replace(TOKEN, "***")
        with open("log.txt", "w", encoding="utf-8") as f:
            f.write(text[-6000:])
