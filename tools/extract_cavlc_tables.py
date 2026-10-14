#!/usr/bin/env python3
"""Extract the CAVLC variable-length code tables from the H.264 specification PDF.

Reads work/spec/h264.pdf (Rec. ITU-T H.264 (V16) (06/2026)) and writes
crates/altc264/src/cavlc_tables.rs containing:

  * COEFF_TOKEN                   - Table 9-5 (six nC columns),
  * TOTAL_ZEROS_4X4               - Tables 9-7 and 9-8,
  * TOTAL_ZEROS_CHROMA_DC_420/422 - Table 9-9 (a) and (b),
  * RUN_BEFORE                    - Table 9-10,
  * CBP_INTRA/INTER_420, _0_3     - Table 9-4 (a) and (b).

Written against the V16 (06/2026) edition. Only the Python standard library
is used, plus the poppler `pdftotext` tool (for word bounding boxes). The
output is deterministic; re-run with:

    python3 tools/extract_cavlc_tables.py [path/to/h264.pdf] [out.rs]

Method: every table is read from individual words with their coordinates.
A table body row is a text line that starts with integer row label(s) left
of the first code column, followed by codeword fragments ("0000", "0011",
"1", ...) or "-". Fragments that are separated by a single space (a small
horizontal gap) belong to one cell; wider gaps separate columns. Cells in a
column share their left edge, so the column of a cell is found by its left
x coordinate. The column identities come from the header row (integer
tzVlcIndex / zerosLeft values, aligned with the cells), or for Table 9-5
from the order of the nC header groups. After extraction every code table
is checked to be prefix-free with a Kraft sum <= 1.
"""

import html
import os
import re
import subprocess
import sys
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PDF = os.path.join(ROOT, "work", "spec", "h264.pdf")
DEFAULT_OUT = os.path.join(ROOT, "crates", "altc264", "src", "cavlc_tables.rs")
SPEC_EDITION = "Rec. ITU-T H.264 (V16) (06/2026)"

MINUS = "−"  # the spec uses U+2212 MINUS SIGN

WORD_RE = re.compile(
    r'<word xMin="([-\d.]+)" yMin="([-\d.]+)" xMax="([-\d.]+)" yMax="([-\d.]+)">([^<]*)</word>'
)
CAPTION_RE = re.compile(r"^Table (9-\d+) – (.*)$")
CODE_RE = re.compile(r"[01]+")
INT_RE = re.compile(r"\d+")

CELL_GAP = 5.0     # max horizontal gap (pt) between fragments of one cell
COL_TOL = 3.0      # tolerance (pt) for left edges of cells in one column
LABEL_MARGIN = 15.0  # row labels end this far left of the first code column


class Word:
    __slots__ = ("x0", "y0", "x1", "y1", "text")

    def __init__(self, x0, y0, x1, y1, text):
        self.x0, self.y0, self.x1, self.y1, self.text = x0, y0, x1, y1, text

    @property
    def xc(self):
        return (self.x0 + self.x1) / 2

    @property
    def yc(self):
        return (self.y0 + self.y1) / 2

    def __repr__(self):
        return f"Word({self.text!r} @ {self.x0:.1f},{self.yc:.1f})"


def fail(msg):
    sys.exit(f"extract_cavlc_tables: error: {msg}")


# ---------------------------------------------------------------------------
# PDF access
# ---------------------------------------------------------------------------

def load_pages(pdf):
    """Return a list (index = PDF page number - 1) of word lists."""
    out = subprocess.run(
        ["pdftotext", "-bbox-layout", "-enc", "UTF-8", pdf, "-"],
        check=True, stdout=subprocess.PIPE,
    ).stdout.decode("utf-8")
    pages = []
    for chunk in out.split("<page ")[1:]:
        words = []
        for m in WORD_RE.finditer(chunk):
            x0, y0, x1, y1 = (float(v) for v in m.groups()[:4])
            words.append(Word(x0, y0, x1, y1, html.unescape(m.group(5))))
        pages.append(words)
    return pages


def group_lines(words, tol=2.0):
    """Group words into text lines by vertical centre (sorted top to bottom)."""
    lines = []
    for w in sorted(words, key=lambda w: (w.yc, w.x0)):
        if lines and abs(lines[-1][0].yc - w.yc) <= tol:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [sorted(l, key=lambda w: w.x0) for l in lines]


def line_text(line):
    return " ".join(w.text for w in line)


def find_captions(pages):
    """Map table number ('9-5', ...) to a list of (page_index, caption_y0, region).

    A region runs from the caption line down to the next caption on the same
    page or the page footer ("Rec. ITU-T H.264 ..." line). Captions in the
    list of tables (with dot leaders) are skipped. Tables that continue on
    a following page with a repeated caption produce several regions.
    """
    tables = {}
    for pi, words in enumerate(pages):
        lines = group_lines(words)
        caps = []
        for line in lines:
            text = line_text(line)
            m = CAPTION_RE.match(text)
            if m and "....." not in text:
                caps.append((line[0].y0, m.group(1)))
        if not caps:
            continue
        footer = page_footer(words, lines, caps[-1][0])
        for i, (y, name) in enumerate(caps):
            y_end = caps[i + 1][0] if i + 1 < len(caps) else footer
            region = [w for w in words if y <= w.y0 < y_end]
            tables.setdefault(name, []).append((pi, y, region))
    return tables


def page_footer(words, lines, after_y):
    footer = max(w.y1 for w in words) + 1
    for line in lines:
        if "Rec. ITU-T H.264" in line_text(line) and line[0].y0 > after_y:
            footer = min(footer, line[0].y0)
    return footer


# ---------------------------------------------------------------------------
# Generic codeword grid
# ---------------------------------------------------------------------------

def split_cells(words):
    """Split the words of one line (left to right) into cells by gap width."""
    cells = []
    for w in words:
        if cells and w.x0 - cells[-1][-1].x1 <= CELL_GAP:
            cells[-1].append(w)
        else:
            cells.append([w])
    return cells


def cell_code(cell, where):
    """Codeword string of a cell, or None for '-'."""
    texts = [w.text for w in cell]
    if texts == ["-"]:
        return None
    if not all(CODE_RE.fullmatch(t) for t in texts):
        fail(f"{where}: unparseable cell {' '.join(texts)!r}")
    return "".join(texts)


def body_rows(lines, n_labels, first_col_x):
    """Table body rows: (labels, cells) for lines starting with n_labels
    integer row labels left of the first code column, followed only by
    codeword fragments and '-'."""
    rows = []
    limit = first_col_x - LABEL_MARGIN
    for line in lines:
        labels = [w for w in line if w.x1 < limit]
        rest = [w for w in line if w.x1 >= limit]
        if len(labels) != n_labels or not all(INT_RE.fullmatch(w.text) for w in labels):
            continue
        if not rest or not all(CODE_RE.fullmatch(w.text) or w.text == "-" for w in rest):
            continue
        rows.append(([int(w.text) for w in labels], split_cells(rest)))
    return rows


def column_starts(rows):
    """Cluster the left edges of all cells into column start positions."""
    xs = sorted(c[0].x0 for _, cells in rows for c in cells)
    clusters = []
    for x in xs:
        if clusters and x - clusters[-1][-1] <= COL_TOL:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return [sum(c) / len(c) for c in clusters]


def assign_column(cell, starts, where):
    x = cell[0].x0
    i = min(range(len(starts)), key=lambda i: abs(starts[i] - x))
    if abs(starts[i] - x) > COL_TOL:
        fail(f"{where}: cell at x={x:.1f} matches no column")
    return i


def numeric_grid(name, lines, col_values, col_texts=None):
    """Parse a grid whose header row lists the column values as integers
    (left-aligned with the cells) and whose single row label is the value.

    col_values: the expected column values, left to right. col_texts: the
    header texts if they differ from str(value) (e.g. '>6').
    Returns {column value: [(row value, code), ...]} in row order.
    """
    col_texts = col_texts or [str(v) for v in col_values]
    header = None
    for line in lines:
        texts = [w.text for w in line]
        if texts == col_texts:
            header = line
            break
    if header is None:
        fail(f"Table {name}: header row {col_texts} not found")
    hx = [w.x0 for w in header]
    rows = [r for r in body_rows([l for l in lines if l[0].yc > header[0].yc], 1, hx[0])]
    if not rows:
        fail(f"Table {name}: no body rows")
    starts = column_starts(rows)
    if len(starts) != len(hx) or any(abs(a - b) > COL_TOL for a, b in zip(starts, hx)):
        fail(f"Table {name}: cell columns {starts} do not match header {hx}")
    labels = [r[0][0] for r in rows]
    if labels != list(range(labels[0], labels[0] + len(labels))):
        fail(f"Table {name}: row labels out of sequence: {labels}")
    out = {v: [] for v in col_values}
    for (lab,), cells in rows:
        seen = set()
        for cell in cells:
            ci = assign_column(cell, starts, f"Table {name} row {lab}")
            if ci in seen:
                fail(f"Table {name} row {lab}: two cells in column {col_values[ci]}")
            seen.add(ci)
            code = cell_code(cell, f"Table {name} row {lab}")
            if code is not None:
                out[col_values[ci]].append((lab, code))
    return out


def split_subtables(lines, markers):
    """Split lines into sub-tables at lines starting with the given markers
    ('(a)', '(b)'). Returns {marker: lines}."""
    out, cur = {}, None
    for line in lines:
        if line[0].text in markers:
            cur = line[0].text
            out[cur] = []
        elif cur is not None:
            out[cur].append(line)
    if sorted(out) != sorted(markers):
        fail(f"sub-tables {markers} not found (got {sorted(out)})")
    return out


# ---------------------------------------------------------------------------
# Table 9-5: coeff_token
# ---------------------------------------------------------------------------

NC_HEADERS = [
    "0 <= nC < 2", "2 <= nC < 4", "4 <= nC < 8", "8 <= nC",
    f"nC = = {MINUS}1", f"nC = = {MINUS}2",
]


def parse_coeff_token(regions):
    cols = [dict() for _ in NC_HEADERS]  # (T1, TC) -> code
    order = []
    for pi, _, region in regions:
        lines = group_lines(region)
        # Header line: the one holding the nC column headers. Its words are
        # grouped into the column headers by gap width.
        hdr = next((l for l in lines if "nC" in [w.text for w in l]), None)
        if hdr is None:
            fail(f"Table 9-5 (PDF page {pi + 1}): no nC header")
        label_end = max(w.x1 for w in hdr if w.text in ("TrailingOnes", "TotalCoeff", "coeff_token"))
        groups = split_cells([w for w in hdr if w.x0 > label_end])
        texts = [line_text(g) for g in groups]
        if texts != NC_HEADERS:
            fail(f"Table 9-5 (PDF page {pi + 1}): nC headers {texts}")
        # Row labels sit below the label headers; code cells start right of them.
        rows = body_rows([l for l in lines if l[0].yc > hdr[0].yc], 2, label_end + LABEL_MARGIN)
        starts = column_starts(rows)
        if len(starts) != len(NC_HEADERS):
            fail(f"Table 9-5 (PDF page {pi + 1}): {len(starts)} cell columns")
        # Column k's cells start right of header k-1's end and left of header k's end.
        for k, s in enumerate(starts):
            lo = max(w.x1 for w in groups[k - 1]) if k else 0
            hi = max(w.x1 for w in groups[k])
            if not lo < s < hi:
                fail(f"Table 9-5: cell column {k} at x={s:.1f} not under its header")
        for (t1, tc), cells in rows:
            where = f"Table 9-5 TrailingOnes {t1} TotalCoeff {tc}"
            if not (0 <= t1 <= 3 and t1 <= tc <= 16):
                fail(f"{where}: impossible row")
            if (t1, tc) in order:
                fail(f"{where}: duplicate row")
            order.append((t1, tc))
            seen = set()
            for cell in cells:
                ci = assign_column(cell, starts, where)
                if ci in seen:
                    fail(f"{where}: two cells in column {ci}")
                seen.add(ci)
                code = cell_code(cell, where)
                frags = [w.text for w in cell]
                if code is not None and (any(len(f) != 4 for f in frags[:-1]) or len(frags[-1]) > 4):
                    fail(f"{where}: odd bit grouping {frags}")
                if code is not None:
                    cols[ci][(t1, tc)] = code
    expected = [(t1, tc) for tc in range(17) for t1 in range(4) if t1 <= tc]
    if sorted(order, key=lambda k: (k[1], k[0])) != expected:
        fail("Table 9-5: rows incomplete")
    out = []
    for ci, col in enumerate(cols):
        out.append([((t1 << 5) | tc, col[(t1, tc)]) for (t1, tc) in order if (t1, tc) in col])
    return out


# ---------------------------------------------------------------------------
# Tables 9-7 to 9-10
# ---------------------------------------------------------------------------

def single_region(tables, name):
    regs = tables.get(name)
    if not regs or len(regs) != 1:
        fail(f"Table {name}: expected exactly one region")
    return group_lines(regs[0][2])


def parse_total_zeros_4x4(tables):
    a = numeric_grid("9-7", single_region(tables, "9-7"), list(range(1, 8)))
    b = numeric_grid("9-8", single_region(tables, "9-8"), list(range(8, 16)))
    a.update(b)
    return [a[i] for i in range(1, 16)]


def parse_total_zeros_chroma(tables):
    subs = split_subtables(single_region(tables, "9-9"), ["(a)", "(b)"])
    a = numeric_grid("9-9 (a)", subs["(a)"], [1, 2, 3])
    b = numeric_grid("9-9 (b)", subs["(b)"], list(range(1, 8)))
    return [a[i] for i in range(1, 4)], [b[i] for i in range(1, 8)]


def parse_run_before(tables):
    g = numeric_grid("9-10", single_region(tables, "9-10"), list(range(1, 8)),
                     ["1", "2", "3", "4", "5", "6", ">6"])
    return [g[i] for i in range(1, 8)]


# ---------------------------------------------------------------------------
# Table 9-4: coded_block_pattern (continues over pages without a caption)
# ---------------------------------------------------------------------------

def parse_cbp(pages, tables):
    regs = tables.get("9-4")
    if not regs or len(regs) != 1:
        fail("Table 9-4: expected one caption")
    pi0, cap_y, _ = regs[0]
    sub = None
    data = {"(a)": {}, "(b)": {}}
    done = False
    for pi in range(pi0, pi0 + 4):
        lines = group_lines(pages[pi])
        footer = page_footer(pages[pi], lines, 0)
        for line in lines:
            if pi == pi0 and line[0].y0 < cap_y:
                continue
            if line[0].y0 >= footer:
                break
            text = line_text(line)
            if text.startswith("9.2 CAVLC parsing process"):
                done = True
                break
            if line[0].text in data:
                sub = line[0].text
                continue
            if sub and len(line) == 3 and all(INT_RE.fullmatch(w.text) for w in line):
                n, intra, inter = (int(w.text) for w in line)
                if n in data[sub]:
                    fail(f"Table 9-4 {sub}: codeNum {n} twice")
                data[sub][n] = (intra, inter)
        if done:
            break
    if not done:
        fail("Table 9-4: end of table not found")
    out = {}
    for sub, size, maxv in (("(a)", 48, 47), ("(b)", 16, 15)):
        d = data[sub]
        if sorted(d) != list(range(size)):
            fail(f"Table 9-4 {sub}: codeNum {sorted(d)}")
        for k in (0, 1):
            col = [d[n][k] for n in range(size)]
            if sorted(col) != list(range(maxv + 1)):
                fail(f"Table 9-4 {sub}: column {k} is not a permutation of 0..={maxv}")
            out[(sub, k)] = col
    # (b) for ChromaArrayType 0/3 carries the luma bits only (0..15).
    return out[("(a)", 0)], out[("(a)", 1)], out[("(b)", 0)], out[("(b)", 1)]


# ---------------------------------------------------------------------------
# Checks and output
# ---------------------------------------------------------------------------

def check_code_table(name, entries):
    """Prefix-freeness; returns the Kraft sum."""
    codes = sorted(c for _, c in entries)
    if len(set(v for v, _ in entries)) != len(entries):
        fail(f"{name}: duplicate value")
    for a, b in zip(codes, codes[1:]):
        if b.startswith(a):
            fail(f"{name}: {a} is a prefix of {b}")
    return sum(Fraction(1, 2 ** len(c)) for c in codes)


def vlc_lit(value, code):
    return f"Vlc {{ code: 0b{code}, len: {len(code)}, value: {value} }}"


def emit_tables(L, name, doc, tables, labels):
    L.extend(doc)
    L.append("#[rustfmt::skip]")
    L.append(f"pub const {name}: [&[Vlc]; {len(tables)}] = [")
    for label, entries in zip(labels, tables):
        L.append(f"    // {label}")
        L.append("    &[")
        for value, code in entries:
            L.append(f"        {vlc_lit(value, code)},")
        L.append("    ],")
    L.append("];")
    L.append("")


def emit_u8(L, name, doc, values):
    L.extend(doc)
    L.append("#[rustfmt::skip]")
    L.append(f"pub const {name}: [u8; {len(values)}] = [")
    for r in range(0, len(values), 8):
        chunk = values[r:r + 8]
        L.append(f"    {', '.join(f'{v:2}' for v in chunk)}, // codeNum {r}..={r + len(chunk) - 1}")
    L.append("];")
    L.append("")


def render(ct, tz, tz420, tz422, rb, cbp):
    L = []
    L.append("//! CAVLC variable-length code tables.")
    L.append("//!")
    L.append(f"//! Generated by tools/extract_cavlc_tables.py from {SPEC_EDITION}. Do not edit.")
    L.append("//!")
    L.append("//! Codewords are stored as their bit strings read MSB first: `code` holds")
    L.append("//! the `len` bits of the codeword. Entries the spec marks \"-\" are omitted.")
    L.append("")
    L.append("/// One codeword: `code` holds the `len` bits of the codeword (MSB first).")
    L.append("#[derive(Clone, Copy, Debug, PartialEq, Eq)]")
    L.append("pub struct Vlc {")
    L.append("    pub code: u32,")
    L.append("    pub len: u8,")
    L.append("    pub value: u8,")
    L.append("}")
    L.append("")
    emit_tables(L, "COEFF_TOKEN", [
        "/// Table 9-5 coeff_token. Index 0..=5 = columns 0<=nC<2, 2<=nC<4, 4<=nC<8, 8<=nC, nC==-1, nC==-2.",
        "/// value = (TrailingOnes << 5) | TotalCoeff.",
    ], ct, ["0 <= nC < 2", "2 <= nC < 4", "4 <= nC < 8", "8 <= nC", "nC == -1", "nC == -2"])
    emit_tables(L, "TOTAL_ZEROS_4X4", [
        "/// Table 9-7 and 9-8: total_zeros for 4x4 blocks, index tzVlcIndex - 1 (0..=14). value = total_zeros.",
    ], tz, [f"tzVlcIndex {i}" for i in range(1, 16)])
    emit_tables(L, "TOTAL_ZEROS_CHROMA_DC_420", [
        "/// Table 9-9 (a): chroma DC 2x2 (ChromaArrayType 1), index tzVlcIndex - 1 (0..=2).",
    ], tz420, [f"tzVlcIndex {i}" for i in range(1, 4)])
    emit_tables(L, "TOTAL_ZEROS_CHROMA_DC_422", [
        "/// Table 9-9 (b): chroma DC 2x4 (ChromaArrayType 2), index tzVlcIndex - 1 (0..=6).",
    ], tz422, [f"tzVlcIndex {i}" for i in range(1, 8)])
    emit_tables(L, "RUN_BEFORE", [
        "/// Table 9-10: run_before, index min(zerosLeft, 7) - 1 (0..=6, the last for zerosLeft > 6). value = run_before.",
    ], rb, [f"zerosLeft {i}" for i in range(1, 7)] + ["zerosLeft > 6"])
    intra420, inter420, intra03, inter03 = cbp
    emit_u8(L, "CBP_INTRA_420", [
        "/// Table 9-4: coded_block_pattern for each codeNum, for ChromaArrayType 1 or 2.",
        "/// Intra_4x4 / Intra_8x8 column.",
    ], intra420)
    emit_u8(L, "CBP_INTER_420", [
        "/// Table 9-4 (a), ChromaArrayType 1 or 2: Inter column.",
    ], inter420)
    emit_u8(L, "CBP_INTRA_0_3", [
        "/// Table 9-4: for ChromaArrayType 0 or 3 (codeNum 0..=15).",
        "/// Intra_4x4 / Intra_8x8 column.",
    ], intra03)
    emit_u8(L, "CBP_INTER_0_3", [
        "/// Table 9-4 (b), ChromaArrayType 0 or 3: Inter column.",
    ], inter03)
    return "\n".join(L)


def main():
    pdf = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF
    out_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    pages = load_pages(pdf)
    tables = find_captions(pages)

    regions = tables.get("9-5")
    if not regions:
        fail("Table 9-5 not found")
    ct = parse_coeff_token(regions)
    tz = parse_total_zeros_4x4(tables)
    tz420, tz422 = parse_total_zeros_chroma(tables)
    rb = parse_run_before(tables)
    cbp = parse_cbp(pages, tables)

    report = []
    for name, group, labels in (
        ("Table 9-5", ct, NC_HEADERS),
        ("Tables 9-7/9-8", tz, [f"tzVlcIndex {i}" for i in range(1, 16)]),
        ("Table 9-9 (a)", tz420, [f"tzVlcIndex {i}" for i in range(1, 4)]),
        ("Table 9-9 (b)", tz422, [f"tzVlcIndex {i}" for i in range(1, 8)]),
        ("Table 9-10", rb, [f"zerosLeft {i}" for i in range(1, 7)] + ["zerosLeft > 6"]),
    ):
        for label, entries in zip(labels, group):
            k = check_code_table(f"{name} {label}", entries)
            if k > 1:
                fail(f"{name} {label}: Kraft sum {k} > 1")
            note = "complete" if k == 1 else f"incomplete, {1 - k} of code space unused"
            report.append(f"{name} {label}: {len(entries)} codes, prefix-free, "
                          f"Kraft sum {k} ({note})")

    text = render(ct, tz, tz420, tz422, rb, cbp)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    for line in report:
        print(line)
    print(f"Table 9-4: {len(cbp[0])} + {len(cbp[2])} codeNum entries")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
