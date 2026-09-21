#!/usr/bin/env python3
"""Reels Farsi bot, version 2.

Every run:
1. reads the public Telegram channels in channels.txt (t.me/s/<name> web preview),
2. keeps recent videos that have at least MIN_VIEWS views and (unless REQUIRE_LAUGH=0)
   a laughing emoji in the post text,
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
DOWNLOAD_TRIES = 3                                           # retries for temporary download errors
# Only post videos whose text contains a laughing emoji (funny videos).
REQUIRE_LAUGH = (os.environ.get("REQUIRE_LAUGH") or "1").strip() not in ("0", "false", "no")
# Channels that are allowed without a laughing emoji, e.g. LAUGH_EXEMPT="nature,tgrealnature"
LAUGH_EXEMPT = {c.strip().lstrip("@").lower()
                for c in (os.environ.get("LAUGH_EXEMPT") or "").split(",") if c.strip()}
LAUGH_RE = re.compile("[😂🤣😹😆😄😁😀😅😃😸😺😝😜🤪]")
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
        c = dict(videos=0, no_src=0, old=0, done=0, long=0, low=0, nolaugh=0, ok=0)
        need_laugh = REQUIRE_LAUGH and ch.lower() not in LAUGH_EXEMPT
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
            elif need_laugh and not LAUGH_RE.search(p["text"] or ""):
                c["nolaugh"] += 1
            else:
                c["ok"] += 1
                found.append(p)
        print(
            f"[{ch}] http={status} posts={len(posts)} videos={c['videos']} "
            f"(no_link={c['no_src']}) | rejected: old={c['old']} done={c['done']} "
            f"bad_length={c['long']} low_views={c['low']} no_laugh={c['nolaugh']} | OK={c['ok']} | "
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


def _download_once(url, dest):
    size = 0
    with requests.get(url, headers=UA, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                size += len(chunk)
                if size > MAX_MB * 1024 * 1024:
                    raise Skip("file too big")
                f.write(chunk)


def download(url, dest, tries=DOWNLOAD_TRIES):
    """Downloads with retries for temporary errors (5xx, timeouts, dropped connections)."""
    last = None
    for attempt in range(1, tries + 1):
        try:
            _download_once(url, dest)
            return
        except Skip:
            raise
        except requests.RequestException as e:
            last = e
            code = getattr(e.response, "status_code", 0) or 0
            print(f"  download error (try {attempt}/{tries}): {str(e)[:80]}")
            if 400 <= code < 500:  # expired link, forbidden, ...: retrying will not help
                break
            time.sleep(3 * attempt)
    raise last


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


class TranslateError(Exception):
    """All translation services failed. The video is NOT burned; next run tries again."""


def _chunks(texts, limit=3500):
    chunk, size = [], 0
    for t in texts:
        if chunk and size + len(t) + 1 > limit:
            yield chunk
            chunk, size = [], 0
        chunk.append(t)
        size += len(t) + 1
    if chunk:
        yield chunk


def _split_back(out, n):
    parts = [x.strip() for x in (out or "").strip().split("\n")]
    if len(parts) != n:
        parts = [x for x in parts if x]  # drop blank lines Google may add
    if len(parts) != n:
        raise ValueError(f"line count changed ({len(parts)} vs {n})")
    return parts


def _tr_gtx(lines, lang):
    """Google's public 'gtx' endpoint, one request for all lines."""
    joined = "\n".join(" ".join(t.split()) for t in lines)
    r = requests.get(
        "https://translate.googleapis.com/translate_a/single",
        params={"client": "gtx", "sl": lang or "auto", "tl": "fa", "dt": "t", "q": joined},
        headers=UA, timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    out = "".join(part[0] for part in data[0] if part and part[0])
    return _split_back(out, len(lines))


def _tr_mymemory(lines, lang):
    """MyMemory free API, one request per line."""
    src = lang if lang and lang != "auto" else "en"
    res = []
    for t in lines:
        t = " ".join(t.split())
        r = requests.get(
            "https://api.mymemory.translated.net/get",
            params={"q": t[:450], "langpair": f"{src}|fa"},
            headers=UA, timeout=30,
        )
        r.raise_for_status()
        out = (r.json().get("responseData") or {}).get("translatedText") or ""
        if not out or "MYMEMORY WARNING" in out.upper():
            raise RuntimeError(f"mymemory refused: {out[:80]}")
        res.append(out.strip())
        time.sleep(0.5)
    return res


def _tr_deep_google(lines, lang):
    """The library that failed before; kept as the last option."""
    from deep_translator import GoogleTranslator

    tr = GoogleTranslator(source="auto", target="fa")
    joined = "\n".join(" ".join(t.split()) for t in lines)
    return _split_back(tr.translate(joined), len(lines))


def translate_lines(lines, lang):
    errors = []
    for name, fn in (("gtx", _tr_gtx), ("mymemory", _tr_mymemory), ("google", _tr_deep_google)):
        for attempt in range(2):
            try:
                res = fn(lines, lang)
                print(f"  translated with {name}")
                return res
            except Exception as e:
                errors.append(f"{name}: {str(e)[:80]}")
                print(f"  translate error ({name}, try {attempt + 1}): {str(e)[:80]}")
                time.sleep(3)
    raise TranslateError("; ".join(errors[-3:]))


def to_persian(texts, lang="auto"):
    if lang == "fa":
        return texts
    out = []
    for chunk in _chunks(texts):
        out.extend(translate_lines(chunk, lang))
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
            cap = to_persian([original_text], "auto")[0]
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
    channels = []
    for line in Path("channels.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        name = parts[0].lstrip("@")
        channels.append(name)
        if "nolaugh" in [x.lower() for x in parts[1:]]:
            LAUGH_EXEMPT.add(name.lower())  # e.g. motivational channels
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
        except TranslateError as e:
            errors += 1
            print(f"  translation is down, will retry next run: {e}")
            break
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
