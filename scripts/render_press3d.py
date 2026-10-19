"""Render viz/press3d.html frame by frame into a smooth video.

    python scripts/render_press3d.py                    # figures/press3d.mp4 and the README GIF
    python scripts/render_press3d.py --pause 10 --out figures/press3d_video.mp4 --no-gif

Recording in the browser ("Record video") captures the page as it plays, so a
frame that takes too long to draw is simply missing and the video stutters.
Here every frame is drawn on its own, at exactly its moment in the clip, and
the frames are joined at a constant rate: the result is smooth however slow
the computer is.

Needs ``cache/press3d_clip.js`` (``scripts/export_pose_clip.py``), ffmpeg,
Google Chrome and Playwright, which drives it:

    pip install playwright

Your own Chrome is used, so no browser download is needed. Without Chrome,
``python -m playwright install chromium`` fetches Playwright's (where the
operating system is still supported) and ``--browser bundled`` uses it.
"""

from __future__ import annotations

import argparse
import base64
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CDN = "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js"


def encoder() -> list[str]:
    """The best H.264 encoder this ffmpeg has (conda's builds lack libx264)."""
    found = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True,
                           text=True).stdout
    if " libx264 " in found:
        return ["-c:v", "libx264", "-crf", "18", "-preset", "slow"]
    if " libopenh264 " in found:
        return ["-c:v", "libopenh264", "-b:v", "12M"]
    if " h264_videotoolbox " in found:
        return ["-c:v", "h264_videotoolbox", "-b:v", "12M"]
    raise SystemExit("this ffmpeg has no H.264 encoder; try: conda install -c conda-forge ffmpeg")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--pause", type=float, default=6.0, help="seconds held on the start frame")
    ap.add_argument("--slow", type=float, default=0.5, help="playback speed")
    ap.add_argument("--theme", choices=("dark", "light"), default="dark")
    ap.add_argument("--out", default="figures/press3d.mp4")
    ap.add_argument("--gif", default="figures/press_animation.gif", help="the README's GIF")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--browser", choices=("chrome", "bundled"), default="chrome",
                    help="your installed Google Chrome, or Playwright's own Chromium")
    ap.add_argument("--three", default=None,
                    help="a local three.module.js (0.160.0), to render without internet")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found; conda install -c conda-forge ffmpeg")
    if not (ROOT / "cache" / "press3d_clip.js").is_file():
        raise SystemExit("cache/press3d_clip.js not found; run scripts/export_pose_clip.py first")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit("pip install playwright")

    page_url = ((ROOT / "viz" / "press3d.html").as_uri()
                + f"?v=0&pause={args.pause}&slow={args.slow}&theme={args.theme}")
    frames = Path(tempfile.mkdtemp(prefix="press3d_"))
    with sync_playwright() as p:
        flags = ["--ignore-gpu-blocklist", "--enable-unsafe-swiftshader"]
        try:
            browser = p.chromium.launch(channel="chrome" if args.browser == "chrome" else None,
                                        args=flags)
        except Exception as exc:
            raise SystemExit(f"could not start the browser ({str(exc).splitlines()[0]}); "
                             "is Google Chrome installed? Or try --browser bundled")
        page = browser.new_page(viewport={"width": 1960, "height": 1200})
        if args.three:
            body = Path(args.three).read_bytes()
            page.route(CDN, lambda r: r.fulfill(body=body, content_type="application/javascript",
                                                headers={"Access-Control-Allow-Origin": "*"}))
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(page_url)
        page.wait_for_function("document.title === 'ready'", timeout=60_000)
        if errors:
            raise SystemExit(f"the page failed: {errors[0]}")
        duration = page.evaluate("window.__duration")
        n = int(duration * args.fps) + 1
        print(f"{duration:.1f} s at {args.fps} fps: {n} frames")
        for i in range(n):
            data = page.evaluate(
                "v => { window.__render(v); return document.getElementById('out').toDataURL('image/png'); }",
                i / args.fps)
            (frames / f"f{i:05d}.png").write_bytes(base64.b64decode(data.split(",", 1)[1]))
            if i % args.fps == 0:
                print(f"  {i / args.fps:4.0f} s", flush=True)
        browser.close()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(frames / "f%05d.png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(args.fps), "-i", pattern,
                    *encoder(), "-pix_fmt", "yuv420p", str(out)], check=True)
    print(f"written: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    if not args.no_gif:
        gif = ROOT / args.gif
        palette = ("fps=12,scale=960:-1:flags=lanczos,split[a][b];"
                   "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(args.fps), "-i", pattern,
                        "-vf", palette, str(gif)], check=True)
        print(f"written: {gif} ({gif.stat().st_size / 1e6:.1f} MB)")
    shutil.rmtree(frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
