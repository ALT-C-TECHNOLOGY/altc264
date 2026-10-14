#!/usr/bin/env python3
"""Extract the CABAC data tables from the H.264 specification PDF.

Reads work/spec/h264.pdf (Rec. ITU-T H.264 (V16) (06/2026)) and writes
crates/altc264/src/cabac_tables.rs containing:

  * CTX_INIT_I / CTX_INIT_PB  - (m, n) context initialisation values from
    Tables 9-12 to 9-33,
  * RANGE_TAB_LPS             - Table 9-44,
  * TRANS_IDX_LPS / _MPS      - Table 9-45.

Written against the V16 (06/2026) edition. Only the Python standard library
is used, plus the poppler `pdftotext` tool (for word bounding boxes). The
output is deterministic; re-run with:

    python3 tools/extract_cabac_tables.py [path/to/h264.pdf] [out.rs]

Why bounding boxes: in the plain `-layout` text dump, narrow table cells
wrap, so e.g. "-46" is rendered as "-4" on one line and "6" on the line
below at the same x. We therefore work on individual words with their
coordinates, assign each word to a (row, column) cell geometrically, and
join the fragments of a cell from top to bottom.
"""

import html
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PDF = os.path.join(ROOT, "work", "spec", "h264.pdf")
DEFAULT_OUT = os.path.join(ROOT, "crates", "altc264", "src", "cabac_tables.rs")
SPEC_EDITION = "Rec. ITU-T H.264 (V16) (06/2026)"

MINUS = "−"  # the spec uses U+2212 MINUS SIGN for negative numbers

WORD_RE = re.compile(
    r'<word xMin="([-\d.]+)" yMin="([-\d.]+)" xMax="([-\d.]+)" yMax="([-\d.]+)">([^<]*)</word>'
)
CAPTION_RE = re.compile(r"^Table (9-\d+) – (.*)$")


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
        return f"Word({self.text!r} @ {self.xc:.1f},{self.yc:.1f})"


def fail(msg):
    sys.exit(f"extract_cabac_tables: error: {msg}")


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


def find_tables(pages):
    """Map table number ('9-12', ...) to a list of (page_index, words) regions.

    A region runs from the caption line down to the next caption on the same
    page or the page footer. Captions in the list of tables are skipped
    (they carry dot leaders and a page number). Tables that continue on a
    following page repeat their caption, so they simply produce more regions.
    """
    tables = {}
    for pi, words in enumerate(pages):
        lines = group_lines(words)
        caps = []
        for line in lines:
            text = " ".join(w.text for w in line)
            m = CAPTION_RE.match(text)
            if m and "....." not in text:
                caps.append((line[0].y0, m.group(1)))
        if not caps:
            continue
        # Footer: the line holding the running "Rec. ITU-T H.264" title.
        footer = max(w.y1 for w in words) + 1
        for line in lines:
            text = " ".join(w.text for w in line)
            if "Rec. ITU-T H.264" in text and line[0].y0 > caps[-1][0]:
                footer = min(footer, line[0].y0)
        for i, (y, name) in enumerate(caps):
            y_end = caps[i + 1][0] if i + 1 < len(caps) else footer
            region = [w for w in words if y <= w.y0 < y_end]
            tables.setdefault(name, []).append((pi, region))
    return tables


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------

def is_int(s):
    return re.fullmatch(r"[-" + MINUS + r"]?\d+", s) is not None


def to_int(s):
    return int(s.replace(MINUS, "-"))


def nearest(value, centres):
    """Index of the centre closest to value."""
    return min(range(len(centres)), key=lambda i: abs(centres[i] - value))


def join_cell(frags):
    """Join wrapped fragments of one cell, top to bottom (then left to right)."""
    return "".join(w.text for w in sorted(frags, key=lambda w: (round(w.yc, 0), w.x0)))


def parse_value(text, where):
    if text == "na":
        return None
    if not is_int(text):
        fail(f"unparseable cell {text!r} at {where}")
    return to_int(text)


def caption_range(region):
    """The ctxIdx ranges named in a caption, e.g. 'from 54 to 59, and 399 to 401'."""
    cap = " ".join(w.text for w in group_lines(region)[0])
    nums = [int(n) for n in re.findall(r"\d+", cap.split("ctxIdx", 1)[1])]
    idx = []
    for a, b in zip(nums[0::2], nums[1::2]):
        idx.extend(range(a, b + 1))
    return idx


# ---------------------------------------------------------------------------
# Layout A: ctxIdx across the top, m / n rows, optionally grouped by
# cabac_init_idc (Tables 9-12 to 9-17).
# ---------------------------------------------------------------------------

def parse_column_table(name, region, store, log):
    expected = caption_range(region)
    lines = group_lines(region)
    hdr_word = next(w for w in region if w.text == "ctxIdx" and w.y0 > lines[0][0].y1)
    # The header row is the first line below the "ctxIdx" label that has
    # (at least) as many integers to the right of the labels as the caption
    # names ctxIdx values.
    header = None
    for line in lines:
        if line[0].yc <= hdr_word.yc:
            continue
        ints = [w for w in line if is_int(w.text)]
        if len(ints) >= len(expected):
            header = ints[-len(expected):]
            break
    if header is None:
        fail(f"Table {name}: no ctxIdx header row")
    cols = [to_int(w.text) for w in header]
    if cols != expected:
        fail(f"Table {name}: header {cols} != caption {expected}")
    col_x = [w.xc for w in header]
    left_edge = header[0].x0 - 5
    header_y = header[0].yc

    # Row labels "m" / "n" sit left of the data; an optional group label
    # ("0", "1", "2" = cabac_init_idc, or "I" from "I slices") sits further
    # left on the first line of its group.
    below = [w for w in region if w.yc > header_y + 2]
    row_labels = sorted((w for w in below if w.x1 < left_edge and w.text in ("m", "n")),
                        key=lambda w: w.yc)
    m_x = min(w.x0 for w in row_labels)
    groups = sorted((w for w in below
                     if w.x1 < m_x and w.text in ("0", "1", "2", "I")), key=lambda w: w.yc)
    rows = []  # (label word, group key)
    for lab in row_labels:
        prior = [g for g in groups if g.yc <= lab.yc + 2]
        group = "all" if not groups else ("I" if prior[-1].text == "I" else int(prior[-1].text))
        rows.append((lab, group))
    row_y = [lab.yc for lab, _ in rows]

    # Assign each data fragment to the nearest column (by x) and the nearest
    # row label (by y). Wrapped fragments lie just above/below their row's
    # label line, much closer to it than to the next row's.
    cells = {}
    for w in below:
        if w.x0 < left_edge:
            continue
        key = (nearest(w.yc, row_y), nearest(w.xc, col_x))
        cells.setdefault(key, []).append(w)

    values = {}  # (group, ctxIdx) -> {"m": v, "n": v}
    for (ri, ci), frags in sorted(cells.items()):
        lab, group = rows[ri]
        text = join_cell(frags)
        v = parse_value(text, f"Table {name} ctxIdx {cols[ci]} {group}/{lab.text}")
        values.setdefault((group, cols[ci]), {})[lab.text] = v

    count = 0
    for (group, ctx), mn in values.items():
        if set(mn) != {"m", "n"}:
            fail(f"Table {name}: incomplete cell for ctxIdx {ctx} group {group}: {mn}")
        val = None if mn["m"] is None or mn["n"] is None else (mn["m"], mn["n"])
        targets = ["I", 0, 1, 2] if group == "all" else [group]
        for t in targets:
            store_value(store, t, ctx, val, name)
        count += 1
    groups_seen = sorted({g for g, _ in values}, key=str)
    for g in groups_seen:
        if {c for gg, c in values if gg == g} != set(expected):
            fail(f"Table {name}: group {g} does not cover all ctxIdx")
    log.append((name, expected, groups_seen))


# ---------------------------------------------------------------------------
# Layout B: one ctxIdx per row, two side-by-side halves, each with columns
# ctxIdx | I(and SI) m n | idc0 m n | idc1 m n | idc2 m n
# (Tables 9-18 to 9-33, some continuing over several pages).
# ---------------------------------------------------------------------------

ROW_GROUPS = ["I", "I", 0, 0, 1, 1, 2, 2]


def parse_row_table(name, regions, store, log):
    expected = caption_range(regions[0][1])
    got = {}
    for pi, region in regions:
        lines = group_lines(region)
        # Column header line: 16 words that are "m" / "n" (one 'N' typo
        # appears in Table 9-33, hence the case-insensitive test).
        mn_line = next((l for l in lines
                        if sum(w.text.lower() in ("m", "n") for w in l) == 16), None)
        if mn_line is None:
            fail(f"Table {name} (PDF page {pi + 1}): no m/n header line")
        mn = [w for w in mn_line if w.text.lower() in ("m", "n")]
        labels = [w.text.lower() for w in mn]
        if labels != ["m", "n"] * 8:
            fail(f"Table {name}: unexpected column header {labels}")
        cap_y1 = lines[0][0].y1  # skip the caption, which also says "ctxIdx"
        ctx_hdr = sorted((w for w in region if w.text == "ctxIdx" and w.y0 > cap_y1),
                         key=lambda w: w.x0)
        if len(ctx_hdr) != 2:
            fail(f"Table {name}: expected two ctxIdx column headers")
        # 18 columns: per half, the ctxIdx label column then 8 value columns.
        col_x = []
        col_role = []  # ("label", half) or (group, "m"/"n", half)
        for half in range(2):
            col_x.append(ctx_hdr[half].xc)
            col_role.append(("label", half))
            for k in range(8):
                w = mn[half * 8 + k]
                col_x.append(w.xc)
                col_role.append((ROW_GROUPS[k], labels[half * 8 + k], half))
        # The table body is the run of purely numeric lines below the m/n
        # header; the first line containing other text (running prose after
        # the table) ends it.
        body = []
        for line in lines:
            if line[0].y0 <= mn_line[0].y1:
                continue
            if not all(is_int(w.text) for w in line):
                break
            body.extend(line)
        by_col = {}
        for w in body:
            by_col.setdefault(nearest(w.xc, col_x), []).append(w)

        for half in range(2):
            lab_col = col_role.index(("label", half))
            row_labels = sorted(by_col.get(lab_col, []), key=lambda w: w.yc)
            idxs = [to_int(w.text) for w in row_labels]
            idxs = fix_label_sequence(name, pi, idxs)
            row_y = [w.yc for w in row_labels]
            cells = {}
            for ci, ws in by_col.items():
                role = col_role[ci]
                if role[0] == "label" or role[2] != half:
                    continue
                for w in ws:
                    cells.setdefault((nearest(w.yc, row_y), ci), []).append(w)
            for (ri, ci), frags in cells.items():
                group, which, _ = col_role[ci]
                ctx = idxs[ri]
                v = parse_value(join_cell(frags), f"Table {name} ctxIdx {ctx} {group}/{which}")
                got.setdefault((group, ctx), {})[which] = v

    for g in ("I", 0, 1, 2):
        ctxs = sorted(c for gg, c in got if gg == g)
        if ctxs != expected:
            missing = sorted(set(expected) - set(ctxs))
            extra = sorted(set(ctxs) - set(expected))
            fail(f"Table {name} group {g}: missing {missing} extra {extra}")
    for (group, ctx), mn in got.items():
        if set(mn) != {"m", "n"} or None in mn.values():
            fail(f"Table {name}: bad cell ctxIdx {ctx} group {group}: {mn}")
        store_value(store, group, ctx, (mn["m"], mn["n"]), name)
    log.append((name, expected, ["I", 0, 1, 2]))


WARNINGS = []


def fix_label_sequence(name, pi, idxs):
    """Row labels in one half-column must count up by one.

    The spec has at least one typo in a row label (Table 9-26 prints "641"
    where the sequence requires 541). A label that breaks the sequence while
    both of its neighbours agree on the value it should have is replaced by
    that value and reported; anything else is a hard error.
    """
    out = list(idxs)
    for i, v in enumerate(idxs):
        prev_ok = i == 0 or out[i - 1] + 1 == v
        next_ok = i + 1 == len(idxs) or idxs[i + 1] - 1 == v
        if prev_ok and next_ok:
            continue
        if 0 < i < len(idxs) - 1 and out[i - 1] + 2 == idxs[i + 1]:
            fixed = out[i - 1] + 1
            WARNINGS.append(f"Table {name} (PDF page {pi + 1}): row label {v} "
                            f"read as {fixed} (sequence {out[i - 1]}, ?, {idxs[i + 1]})")
            out[i] = fixed
        elif i == 0 and len(idxs) > 1 and idxs[1] - 1 != v:
            fail(f"Table {name}: cannot repair first row label {v}")
        elif i > 0 and not prev_ok:
            fail(f"Table {name}: row labels out of sequence near {v}: {idxs}")
    return out


def store_value(store, group, ctx, val, table):
    key = (group, ctx)
    if key in store:
        fail(f"ctxIdx {ctx} group {group} defined twice (Table {table})")
    store[key] = (val, table)


# ---------------------------------------------------------------------------
# Tables 9-44 and 9-45
# ---------------------------------------------------------------------------

def parse_range_tab(regions):
    """Table 9-44: two halves of rows 'pStateIdx v0 v1 v2 v3'."""
    pi, region = regions[0]
    table = {}
    for line in group_lines(region):
        ints = [to_int(w.text) for w in line if is_int(w.text)]
        if len(ints) == 10 and all(not w.text.isalpha() for w in line):
            for half in (ints[:5], ints[5:]):
                table[half[0]] = half[1:]
    if sorted(table) != list(range(64)):
        fail(f"Table 9-44: got pStateIdx {sorted(table)}")
    return [table[i] for i in range(64)]


def parse_trans(regions):
    """Table 9-45: blocks of three rows (pStateIdx / transIdxLPS / transIdxMPS)."""
    pi, region = regions[0]
    cur = None
    lps, mps = {}, {}
    for line in group_lines(region):
        head = line[0].text
        ints = [to_int(w.text) for w in line[1:] if is_int(w.text)]
        if head == "pStateIdx":
            cur = ints
        elif head in ("transIdxLPS", "transIdxMPS"):
            if cur is None or len(ints) != len(cur):
                fail("Table 9-45: misaligned row")
            (lps if head == "transIdxLPS" else mps).update(zip(cur, ints))
    if sorted(lps) != list(range(64)) or sorted(mps) != list(range(64)):
        fail("Table 9-45: incomplete")
    return [lps[i] for i in range(64)], [mps[i] for i in range(64)]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def ranges(nums):
    nums = sorted(nums)
    out = []
    for n in nums:
        if out and out[-1][1] + 1 == n:
            out[-1][1] = n
        else:
            out.append([n, n])
    return ", ".join(f"{a}..={b}" if a != b else f"{a}" for a, b in out)


def emit_ctx_array(lines, group, store, indent):
    run_table = None
    for ctx in range(1024):
        entry = store.get((group, ctx))
        if ctx == 276:
            lines.append(f"{indent}(0, 0), // 276: end_of_slice_flag / I_PCM terminate context (9.3.3.2.4), not initialised")
            run_table = None
            continue
        if entry is None or entry[0] is None:
            lines.append(f"{indent}(0, 0), // {ctx}: unused")
            run_table = None
            continue
        (m, n), table = entry
        note = f" Table {table}" if table != run_table else ""
        run_table = table
        lines.append(f"{indent}({m}, {n}), // {ctx}{note}")


def render(store, rtab, lps, mps, log):
    unused_i = [c for c in range(1024) if c != 276 and (store.get(("I", c)) or (None,))[0] is None]
    unused_pb = {g: [c for c in range(1024) if c != 276 and (store.get((g, c)) or (None,))[0] is None]
                 for g in (0, 1, 2)}
    L = []
    L.append(f"//! CABAC data tables.")
    L.append(f"//!")
    L.append(f"//! Generated by tools/extract_cabac_tables.py from {SPEC_EDITION}. Do not edit.")
    L.append(f"//!")
    L.append(f"//! * `CTX_INIT_I`, `CTX_INIT_PB`: (m, n) per ctxIdx, Tables 9-12 to 9-33.")
    L.append(f"//! * `RANGE_TAB_LPS`: Table 9-44, indexed [pStateIdx][qCodIRangeIdx].")
    L.append(f"//! * `TRANS_IDX_LPS`, `TRANS_IDX_MPS`: Table 9-45, indexed by pStateIdx.")
    L.append(f"//!")
    L.append(f"//! ctxIdx 276 (end_of_slice_flag and the I_PCM bin of mb_type) uses the")
    L.append(f"//! terminate decoding process and has no (m, n); it is stored as (0, 0).")
    L.append(f"//! Entries a slice-type class never uses are (0, 0):")
    L.append(f"//! * I and SI slices: ctxIdx {ranges(unused_i) or 'none'}.")
    for g in (0, 1, 2):
        L.append(f"//! * P, SP and B slices, cabac_init_idc {g}: ctxIdx {ranges(unused_pb[g]) or 'none'}.")
    L.append(f"//!")
    L.append(f"//! Tables giving one value set for all slice types (9-12, 9-17) are")
    L.append(f"//! replicated into both arrays.")
    L.append("")
    L.append("/// (m, n) context initialisation values for I and SI slices.")
    L.append("#[rustfmt::skip]")
    L.append("pub const CTX_INIT_I: [(i8, i8); 1024] = [")
    emit_ctx_array(L, "I", store, "    ")
    L.append("];")
    L.append("")
    L.append("/// (m, n) context initialisation values for P, SP and B slices,")
    L.append("/// indexed by cabac_init_idc (0..=2).")
    L.append("#[rustfmt::skip]")
    L.append("pub const CTX_INIT_PB: [[(i8, i8); 1024]; 3] = [")
    for g in (0, 1, 2):
        L.append(f"    // cabac_init_idc = {g}")
        L.append("    [")
        emit_ctx_array(L, g, store, "        ")
        L.append("    ],")
    L.append("];")
    L.append("")
    L.append("/// Table 9-44: rangeTabLPS[pStateIdx][qCodIRangeIdx].")
    L.append("#[rustfmt::skip]")
    L.append("pub const RANGE_TAB_LPS: [[u8; 4]; 64] = [")
    for i, row in enumerate(rtab):
        L.append(f"    [{', '.join(f'{v:3}' for v in row)}], // {i}")
    L.append("];")
    L.append("")
    for name, arr, doc in (("TRANS_IDX_LPS", lps, "transIdxLPS"), ("TRANS_IDX_MPS", mps, "transIdxMPS")):
        L.append(f"/// Table 9-45: {doc}(pStateIdx).")
        L.append("#[rustfmt::skip]")
        L.append(f"pub const {name}: [u8; 64] = [")
        for r in range(0, 64, 8):
            L.append(f"    {', '.join(f'{v:2}' for v in arr[r:r + 8])}, // {r}..={r + 7}")
        L.append("];")
        L.append("")
    return "\n".join(L)


def main():
    pdf = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF
    out_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    pages = load_pages(pdf)
    tables = find_tables(pages)

    store, log = {}, []
    for t in range(12, 34):
        name = f"9-{t}"
        regions = tables.get(name)
        if not regions:
            fail(f"Table {name} not found")
        # Row-per-ctxIdx tables are recognised by their 16-column m/n header.
        row_layout = any(sum(w.text.lower() in ("m", "n") for w in l) == 16
                         for l in group_lines(regions[0][1]))
        if row_layout:
            parse_row_table(name, regions, store, log)
        else:
            if len(regions) != 1:
                fail(f"Table {name}: unexpected continuation")
            parse_column_table(name, regions[0][1], store, log)

    rtab = parse_range_tab(tables["9-44"])
    lps, mps = parse_trans(tables["9-45"])

    text = render(store, rtab, lps, mps, log)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

    for name, expected, groups in log:
        pages_ = sorted({pi + 1 for pi, _ in tables[name]})
        print(f"Table {name}: {len(expected)} ctxIdx ({ranges(expected)}), "
              f"sets {groups}, PDF page(s) {pages_}")
    for w in WARNINGS:
        print("warning:", w)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
