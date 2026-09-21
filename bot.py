#!/usr/bin/env python3
"""Reels Farsi bot, version 2.

Every run:
1. reads the public Telegram channels in channels.txt (t.me/s/<name> web preview),
2. keeps recent videos that have at least MIN_VIEWS views,
3. downloads the best one; if it has speech, transcribes it, translates to Persian
   and burns subtitles,
4. posts it to our channel and remembers it in posted.json.

The log says, for every channel, WHY posts were rejected.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ---------------- settings (environment variables override the defaults) ----------------
TOKEN = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
TARGET = (os.environ.get("CHANNEL_ID") or "@my_reels_fa").strip()
MIN_VIEWS = int(os.environ.get("MIN_VIEWS") or "10000")
MAX_AGE_DAYS = int(os.environ.get("MAX_AGE_DAYS") or "14")
PAGES = int(os.environ.get("PAGES") or "3")                 # pages of ~20 posts per channel
MAX_DURATION = int(os.environ.get("MAX_DURATION") or "120")  # seconds
MIN_DURATION = 3
MAX_MB = 45                                                  # bot upload limit is 50 MB
WHISPER_MODEL = os.environ.get("WHISPER_MODEL") or "small"
TRIES_PER_RUN = 5
STATE_FILE = Path("posted.json")
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}


class Skip(Exception):
    """Video is not usable (too long, too big, ...). It will not be retried."""


# ---------------- state ------------------------------------------------------------------
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


# ---------------- reading channels -------------------------------------------------------
def parse_views(text):
    t = (text or "").strip().upper().replace(",", "")
    m = re.match(r"^([\d.]+)\s*([KM]?)$", t)
    if not m:
        return 0
    mult = {"": 1, "K": 1_000, "M": 1_000_000}[m.group(2)]
    return int(float(m.group(1)) * mult)


def parse_duration(text):
    parts = re.findall(r"\d+", text or "")
    if not parts:
        return None
    sec = 0
    for x in parts:
        sec = sec * 60 + int(x)
    return sec


def parse_page(html, channel):
    soup = BeautifulSoup(html, "html.parser")
    posts = []
    for msg in soup.select("div.tgme_widget_message[data-post]"):
        try:
            pid = int(msg["data-post"].split("/")[-1])
        except ValueError:
            continue
        views_el = msg.select_one(".tgme_widget_message_views")
        video = msg.select_one("video[src]")
        has_player = bool(msg.select_one(
            ".tgme_widget_message_video_player, .tgme_widget_message_roundvideo_player"))
        dur_el = msg.select_one(".message_video_duration")
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
            "views": parse_views(views_el.get_text()) if views_el else 0,
            "video_url": video["src"] if video else None,
            "has_player": has_player,
            "duration": parse_duration(dur_el.get_text()) if dur_el else None,
            "text": text_el.get_text(" ", strip=True) if text_el else "",
            "date": date,
        })
    return posts


def fetch_posts(channel):
    """Returns (posts, http_status_of_first_page)."""
    posts, first_status = [], None
    url = f"https://t.me/s/{channel}"
    for _ in range(PAGES):
        r = requests.get(url, headers=UA, timeout=30)
        if first_status is None:
            first_status = r.status_code
        if r.status_code != 200:
            break
        page = parse_page(r.text, channel)
        if not page:
            break
        posts.extend(page)
        url = f"https://t.me/s/{channel}?before={min(p['id'] for p in page)}"
        time.sleep(1)
    return posts, first_status


def find_candidates(channels, state):
    now = datetime.now(timezone.utc)
    done = set(state["posted"])
    found = []
    for ch in channels:
        try:
            posts, status = fetch_posts(ch)
        except Exception as e:
            print(f"[{ch}] fetch failed: {e}")
            continue
        c = dict(videos=0, no_src=0, old=0, done=0, long=0, low=0, ok=0)
        best = 0
        newest = None
        for p in posts:
            if p["date"] and (newest is None or p["date"] > newest):
                newest = p["date"]
            if not p["video_url"]:
                if p["has_player"]:
                    c["no_src"] += 1  # video exists but the preview gives no direct link
                continue
            c["videos"] += 1
            best = max(best, p["views"])
            if p["key"] in done:
                c["done"] += 1
            elif p["date"] and now - p["date"] > timedelta(days=MAX_AGE_DAYS):
                c["old"] += 1
            elif p["duration"] and not (MIN_DURATION <= p["duration"] <= MAX_DURATION):
                c["long"] += 1
            elif p["views"] < MIN_VIEWS:
                c["low"] += 1
            else:
                c["ok"] += 1
                found.append(p)
        print(
            f"[{ch}] http={status} posts={len(posts)} videos={c['videos']} "
            f"(no_link={c['no_src']}) | rejected: old={c['old']} done={c['done']} "
            f"bad_length={c['long']} low_views={c['low']} | OK={c['ok']} | "
            f"best_video_views={best} newest={newest.date() if newest else '-'}"
        )
    found.sort(key=lambda p: p["views"], reverse=True)
    return found


# ---------------- video helpers ----------------------------------------------------------
def run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {(r.stderr or '')[-400:]}")
    return r.stdout


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
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "csv=p=0", str(path)]).strip()
    return float(out)


def transcribe(path, duration):
    """Returns (has_speech, [(start, end, text)], language)."""
    from faster_whisper import WhisperModel

    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        path, vad_filter=True, beam_size=1, condition_on_previous_text=False)
    segs = []
    for s in segments:
        text = s.text.strip()
        if text and s.no_speech_prob < 0.6:
            segs.append((s.start, s.end, text))
    speech_time = sum(e - s for s, e, _ in segs)
    chars = sum(len(t) for _, _, t in segs)
    has_speech = chars >= 25 and speech_time >= 0.15 * duration
    return has_speech, segs, info.language


def _chunks(texts, limit=4000):
    chunk, size = [], 0
    for t in texts:
        if chunk and size + len(t) + 1 > limit:
            yield chunk
            chunk, size = [], 0
        chunk.append(t)
        size += len(t) + 1
    if chunk:
        yield chunk


def _translate_chunk(chunk):
    """One single request for many lines (Google blocks many quick requests)."""
    from deep_translator import GoogleTranslator

    tr = GoogleTranslator(source="auto", target="fa")
    joined = "\n".join(" ".join(t.split()) for t in chunk)
    out = tr.translate(joined) or ""
    parts = [x.strip() for x in out.split("\n")]
    if len(parts) != len(chunk):
        raise ValueError(f"line count changed ({len(parts)} vs {len(chunk)})")
    return parts


def _translate_safe(chunk, delays):
    last = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return _translate_chunk(chunk)
        except ValueError as e:
            last = e
            break  # line count changed: translate line by line below
        except Exception as e:
            last = e
            print(f"  translate error, will retry: {str(e)[:90]}")
    from deep_translator import GoogleTranslator

    tr = GoogleTranslator(source="auto", target="fa")
    res = []
    for t in chunk:
        for attempt in range(len(delays)):
            try:
                res.append(tr.translate(t) or t)
                break
            except Exception as e:
                last = e
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"translation failed: {str(last)[:150]}")
        time.sleep(1.5)
    return res


def to_persian(texts, lang="auto", delays=(0, 10, 30, 60)):
    if lang == "fa":
        return texts
    out = []
    for chunk in _chunks(texts):
        out.extend(_translate_safe(chunk, delays))
        time.sleep(2)
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
    style = ("FontName=Vazirmatn,FontSize=18,PrimaryColour=&H00FFFFFF,"
             "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,Alignment=2,MarginV=30")
    vf = f"scale=trunc(iw/2)*2:trunc(ih/2)*2,subtitles=subs.srt:force_style='{style}'"
    run(["ffmpeg", "-y", "-i", "in.mp4", "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "23", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "out.mp4"],
        cwd=workdir)


# ---------------- posting ----------------------------------------------------------------
def clean_caption(text):
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:200] if len(text) >= 3 else ""


def build_caption(original_text, channel):
    cap = ""
    if original_text:
        try:
            cap = to_persian([original_text], "auto", delays=(0, 15))[0]
        except Exception:
            cap = ""
    return (cap + "\n\nمنبع: @" + channel).strip()


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

        send_video(final, build_caption(clean_caption(p["text"]), p["channel"]))


def main():
    if not TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set (check the secret name in reels.yml)")
    print(f"settings: MIN_VIEWS={MIN_VIEWS} MAX_AGE_DAYS={MAX_AGE_DAYS} PAGES={PAGES} target={TARGET}")
    channels = [
        line.strip().lstrip("@")
        for line in Path("channels.txt").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    state = load_state()
    candidates = find_candidates(channels, state)
    print(f"{len(candidates)} candidate videos")

    posted, errors = False, 0
    for p in candidates[:TRIES_PER_RUN]:
        key = p["key"]
        print(f"trying {key} ({p['views']} views)")
        try:
            process(p)
        except Skip as e:
            print(f"  skipped: {e}")
            state["posted"].append(key)
            continue
        except Exception as e:
            errors += 1
            fails = state["fails"].get(key, 0) + 1
            state["fails"][key] = fails
            print(f"  failed ({fails}): {e}")
            traceback.print_exc(file=sys.stdout)
            if fails >= 3:
                state["posted"].append(key)
            continue
        state["posted"].append(key)
        print("  posted")
        posted = True
        break
    if not posted:
        print("nothing posted this run")
    save_state(state)
    if candidates and not posted and errors:
        sys.exit(1)  # make the run red so the problem is visible


if __name__ == "__main__":
    main()
