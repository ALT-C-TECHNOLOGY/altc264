#!/usr/bin/env python3
"""Draws the README's benchmark chart (the 2D clip, altc264 against FFmpeg)
as light and dark SVGs in .github/assets/. The numbers are BENCHMARKS.md's.
    tools/bench/chart.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# The version shown in the legend: the workspace's, from Cargo.toml.
VERSION = re.search(r'^version = "([^"]+)"', (ROOT / "Cargo.toml").read_text(), re.M).group(1)

# CPU, (altc264, FFmpeg) frames per second at 1 thread, then at 8 threads.
ROWS_2D = [
    ("Apple M4 Pro", (132.5, 109.7), (578.1, 370.1)),
    ("Intel Core Ultra X9 388H", (108.7, 114.8), (360.3, 289.6)),
    ("AMD Ryzen AI 9 HX 370", (95.7, 108.6), (409.8, 351.0)),
    ("Qualcomm QCM6490", (51.8, 49.1), (191.2, 167.3)),
    ("WebAssembly (SIMD128)", (77.2, 62.3), (427.4, 272.0)),
]
CHARTS = [
    ("benchmark", "Big Buck Bunny 1080p: frames per second (higher is faster)", ROWS_2D),
]

THEMES = {
    "light": dict(bg="#ffffff", text="#0a0e12", muted="#5b6470", grid="#e3e6ea", ffmpeg="#b9c0c8"),
    "dark": dict(bg="#0d1117", text="#e6edf3", muted="#8b949e", grid="#21262d", ffmpeg="#4a525c"),
}

W, ROW_H, BAR_H, GAP = 960, 46, 15, 4
LABEL_W, PANEL_W, PANEL_GAP, TOP = 210, 330, 40, 116


def panel(x0, title, key, scale, t, ROWS):
    out = [f'<text x="{x0}" y="{TOP - 20}" class="h">{title}</text>']
    for i, row in enumerate(ROWS):
        vals = row[1] if key == 0 else row[2]
        y = TOP + i * ROW_H
        if vals is None:
            out.append(f'<text x="{x0}" y="{y + BAR_H + 4}" class="m">not supported yet</text>')
            continue
        a, f = vals
        wa, wf = a / scale * (PANEL_W - 95), f / scale * (PANEL_W - 95)
        ratio = a / f
        out.append(f'<rect x="{x0}" y="{y}" width="{wa:.1f}" height="{BAR_H}" rx="3" fill="url(#brand)"/>')
        out.append(f'<text x="{x0 + wa + 6:.1f}" y="{y + BAR_H - 3}" class="v">{a:.0f}</text>')
        out.append(f'<rect x="{x0}" y="{y + BAR_H + GAP}" width="{wf:.1f}" height="{BAR_H}" rx="3" fill="{t["ffmpeg"]}"/>')
        out.append(f'<text x="{x0 + wf + 6:.1f}" y="{y + 2 * BAR_H + GAP - 3}" class="m">{f:.0f}</text>')
        badge = "r" if ratio >= 1 else "m"
        out.append(f'<text x="{x0 + PANEL_W}" y="{y + BAR_H + 6}" text-anchor="end" class="{badge}">{ratio:.2f}×</text>')
    return out


def chart(t, title, ROWS):
    h = TOP + len(ROWS) * ROW_H + 40
    s1 = max(max(r[1]) for r in ROWS)
    s8 = max(max(r[2]) for r in ROWS if r[2])
    x1 = LABEL_W
    x8 = LABEL_W + PANEL_W + PANEL_GAP
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}" '
        'font-family="-apple-system, BlinkMacSystemFont, \'Segoe UI\', Helvetica, Arial, sans-serif">',
        '<defs><linearGradient id="brand" x1="0" x2="1" y1="0" y2="0">'
        '<stop offset="0" stop-color="#B7F542"/><stop offset="1" stop-color="#3EE0B8"/></linearGradient></defs>',
        f'<style>.t{{font-size:18px;font-weight:600;fill:{t["text"]}}}.h{{font-size:14px;font-weight:600;fill:{t["text"]}}}'
        f'.l{{font-size:13px;fill:{t["text"]}}}.v{{font-size:12px;font-weight:600;fill:{t["text"]}}}'
        f'.m{{font-size:12px;fill:{t["muted"]}}}.r{{font-size:13px;font-weight:700;fill:#2fbf8f}}</style>',
        f'<rect width="{W}" height="{h}" rx="12" fill="{t["bg"]}"/>',
        f'<text x="24" y="34" class="t">{title}</text>',
        f'<rect x="24" y="48" width="12" height="12" rx="2" fill="url(#brand)"/><text x="42" y="58" class="l">altc264 {VERSION}</text>',
        f'<rect x="150" y="48" width="12" height="12" rx="2" fill="{t["ffmpeg"]}"/><text x="168" y="58" class="l">FFmpeg 9.0.2</text>',
    ]
    for i, row in enumerate(ROWS):
        y = TOP + i * ROW_H
        out.append(f'<line x1="24" x2="{W - 24}" y1="{y + ROW_H - 6}" y2="{y + ROW_H - 6}" stroke="{t["grid"]}"/>')
        out.append(f'<text x="24" y="{y + BAR_H + 6}" class="l">{row[0]}</text>')
    out += panel(x1, "1 thread", 0, s1, t, ROWS)
    out += panel(x8, "8 threads", 1, s8, t, ROWS)
    out.append('</svg>')
    return "\n".join(out) + "\n"


def main():
    assets = ROOT / ".github" / "assets"
    for stem, title, rows in CHARTS:
        for name, t in THEMES.items():
            (assets / f"{stem}-{name}.svg").write_text(chart(t, title, rows))
            print(f"wrote {assets / f'{stem}-{name}.svg'}")


if __name__ == "__main__":
    main()
