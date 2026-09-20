#!/usr/bin/env python3
"""Reels Farsi bot.

Every run:
1. reads public Telegram channels from channels.txt (via the t.me/s/ web preview),
2. picks a recent video whose views are well above that channel's own average,
3. if the video has speech: transcribes it, translates to Persian, burns subtitles,
   otherwise posts the video as it is,
4. posts it to our channel and remembers it in posted.json.
"""
import json
import os
import re
import statistics
import subprocess
import tempfile
import textwrap
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ---------------- settings (can be overridden with environment variables) -------------
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TARGET = os.environ.get("CHANNEL_ID", "@my_reels_fa")
MIN_RATIO = float(os.environ.get("MIN_RATIO", "2.0"))      # views >= 2x channel median
MIN_VIEWS = int(os.environ.get("MIN_VIEWS", "1000"))       # ignore tiny numbers
MAX_AGE_DAYS = int(os.environ.get("MAX_AGE_DAYS", "7"))    # only recent videos
MAX_DURATION = int(os.environ.get("MAX_DURATION", "120"))  # seconds
MIN_DURATION = 3
MAX_MB = 45                                                # Telegram bot upload limit is 50 MB
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
STATE_FILE = Path("posted.json")
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}


class Skip(Exception):
    """Video is not usable (too long, too big, ...). It will not be retried."""


# ---------------- state ----------------------------------------------------------------
def load_state():
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text())
            return {"posted": list(data.get("posted", [])), "fails": dict(data.get("fails", {}))}
        except Exception:
            pass
    return {"posted": [], "fails": {}}


def save_state(state):
    state["posted"] = state["posted"][-3000:]
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=1))


# ---------------- reading channels -----------------------------------------------------
def parse_views(text):
    t = text.strip().upper().replace(",", "")
    m = re.match(r"^([\d.]+)\s*([KM]?)$", t)
    if not m:
        return 0
    mult = {"": 1, "K": 1_000, "M": 1_000_000}[m.group(2)]
    return int(float(m.group(1)) * mult)


def parse_page(html, channel):
    soup = BeautifulSoup(html, "html.parser")
    posts = []
    for msg in soup.select("div.tgme_widget_message[data-post]"):
        try:
            pid = int(msg["data-post"].split("/")[-1])
        except ValueError:
            continue
        views_el = msg.select_one(".tgme_widget_message_views")
        views = parse_views(views_el.get_text()) if views_el else 0
        video = msg.select_one("video.tgme_widget_message_video[src]")
        text_el = msg.select_one(".tgme_widget_message_text")
        date = None
        time_el = msg.select_one("time[datetime]")
        if time_el:
            try:
                date = datetime.fromisoformat(time_el["datetime"])
            except ValueError:
                pass
        posts.append({
            "channel": channel,
            "id": pid,
            "key": f"{channel}/{pid}",
            "views": views,
            "video_url": video["src"] if video else None,
            "text": text_el.get_text(" ", strip=True) if text_el else "",
            "date": date,
        })
    return posts


def fetch_posts(channel, pages=2):
    posts = []
    url = f"https://t.me/s/{channel}"
    for _ in range(pages):
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code != 200:
            break
        page = parse_page(r.text, channel)
        if not page:
            break
        posts.extend(page)
        url = f"https://t.me/s/{channel}?before={min(p['id'] for p in page)}"
        time.sleep(1)
    return posts


def find_candidates(channels, state):
    now = datetime.now(timezone.utc)
    done = set(state["posted"])
    found = []
    for ch in channels:
        try:
            posts = fetch_posts(ch)
        except Exception as e:  # network problem with one channel must not stop the rest
            print(f"[{ch}] fetch failed: {e}")
            continue
        views = [p["views"] for p in posts if p["views"] > 0]
        if len(views) < 8:
            print(f"[{ch}] not enough posts with views ({len(views)}), skipped")
            continue
        base = statistics.median(views)
        n = 0
        for p in posts:
            if not p["video_url"] or p["key"] in done:
                continue
            if p["date"] and now - p["date"] > timedelta(days=MAX_AGE_DAYS):
                continue
            if p["views"] < MIN_VIEWS or not base:
                continue
            ratio = p["views"] / base
            if ratio >= MIN_RATIO:
                p["ratio"] = ratio
                found.append(p)
                n += 1
        print(f"[{ch}] median views {int(base)}, candidates {n}")
    found.sort(key=lambda p: p["ratio"], reverse=True)
    return found


# ---------------- video helpers --------------------------------------------------------
def download(url, dest):
    size = 0
    with requests.get(url, headers=UA, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                size += len(chunk)
                if size > MAX_MB * 1024 * 1024:
                    raise Skip("file too big")
                f.write(chunk)


def video_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def transcribe(path, duration):
    """Returns (has_speech, [(start, end, text)], language)."""
    from faster_whisper import WhisperModel

    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        path, vad_filter=True, beam_size=1, condition_on_previous_text=False
    )
    segs = []
    for s in segments:
        text = s.text.strip()
        if text and s.no_speech_prob < 0.6:
            segs.append((s.start, s.end, text))
    speech_time = sum(e - s for s, e, _ in segs)
    chars = sum(len(t) for _, _, t in segs)
    has_speech = chars >= 25 and speech_time >= 0.15 * duration
    return has_speech, segs, info.language


def to_persian(texts, lang):
    if lang == "fa":
        return texts
    from deep_translator import GoogleTranslator

    tr = GoogleTranslator(source="auto", target="fa")
    out = []
    for t in texts:
        out.append(tr.translate(t) or t)
        time.sleep(0.2)
    return out


def srt_time(sec):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(path, segs, fa_texts):
    lines = []
    for i, ((start, end, _), fa) in enumerate(zip(segs, fa_texts), 1):
        wrapped = "\n".join(textwrap.wrap(fa, 32)) or fa
        lines.append(f"{i}\n{srt_time(start)} --> {srt_time(end)}\n{wrapped}\n")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def burn_subtitles(workdir):
    style = (
        "FontName=Vazirmatn,FontSize=18,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,Alignment=2,MarginV=30"
    )
    vf = f"scale=trunc(iw/2)*2:trunc(ih/2)*2,subtitles=subs.srt:force_style='{style}'"
    subprocess.run(
        ["ffmpeg", "-y", "-i", "in.mp4", "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "23", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "out.mp4"],
        cwd=workdir, check=True, capture_output=True,
    )


# ---------------- posting --------------------------------------------------------------
def clean_caption(text):
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:200] if len(text) >= 3 else ""


def build_caption(original_text, channel, lang):
    cap = ""
    if original_text:
        try:
            cap = to_persian([original_text], "auto")[0]
        except Exception:
            cap = ""
    source = f"منبع: @{channel}"
    return (cap + "\n\n" + source).strip()


def send_video(path, caption):
    with open(path, "rb") as f:
        r = requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendVideo",
            data={"chat_id": TARGET, "caption": caption[:1000], "supports_streaming": "true"},
            files={"video": f},
            timeout=300,
        )
    if not r.ok:
        raise RuntimeError(f"Telegram error: {r.text[:300]}")


def process(p):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "in.mp4"
        download(p["video_url"], src)
        dur = video_duration(src)
        if dur < MIN_DURATION or dur > MAX_DURATION:
            raise Skip(f"duration {dur:.0f}s out of range")

        has_speech, segs, lang = transcribe(str(src), dur)
        final = src
        if has_speech:
            fa = to_persian([t for _, _, t in segs], lang)
            write_srt(tmp / "subs.srt", segs, fa)
            burn_subtitles(tmp)
            final = tmp / "out.mp4"
            print(f"  speech found (lang={lang}), subtitles burned")
        else:
            print("  no speech, posting without subtitles")

        caption = build_caption(clean_caption(p["text"]), p["channel"], lang)
        send_video(final, caption)


def main():
    if not TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set")
    channels = [
        line.strip().lstrip("@")
        for line in Path("channels.txt").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    state = load_state()
    candidates = find_candidates(channels, state)
    print(f"{len(candidates)} candidate videos")

    for p in candidates[:5]:
        key = p["key"]
        print(f"trying {key} (views {p['views']}, {p['ratio']:.1f}x average)")
        try:
            process(p)
        except Skip as e:
            print(f"  skipped: {e}")
            state["posted"].append(key)
            continue
        except Exception as e:
            fails = state["fails"].get(key, 0) + 1
            state["fails"][key] = fails
            print(f"  failed ({fails}): {e}")
            if fails >= 3:
                state["posted"].append(key)
            continue
        state["posted"].append(key)
        print("  posted")
        break
    else:
        print("nothing posted this run")
    save_state(state)


if __name__ == "__main__":
    main()
