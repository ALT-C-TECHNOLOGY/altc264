#!/usr/bin/env python3
"""Summarises a samply profile recorded with --unstable-presymbolicate.

usage: profile_summary.py PROFILE.json.gz [N] [--callers NAME]

Prints the top N functions by self and inclusive sample count, over all
threads, or with --callers the callers of functions whose name contains NAME.
"""
import bisect
import collections
import gzip
import json
import sys


def main():
    args = sys.argv[1:]
    callers_of = None
    if "--callers" in args:
        i = args.index("--callers")
        callers_of = args[i + 1]
        del args[i : i + 2]
    path = args[0]
    top = int(args[1]) if len(args) > 1 else 30
    callers = collections.Counter()
    prof = json.load(gzip.open(path) if path.endswith(".gz") else open(path))
    syms_path = path.removesuffix(".gz").removesuffix(".json") + ".json.syms.json"
    syms = json.load(open(syms_path))
    strings = syms["string_table"]
    tables = {}
    for lib in syms["data"]:
        entries = sorted(lib["symbol_table"], key=lambda e: e["rva"])
        tables[lib["debug_name"]] = ([e["rva"] for e in entries], entries)

    def lookup(lib_name, addr):
        t = tables.get(lib_name)
        if not t:
            return f"{lib_name}+{addr:#x}"
        rvas, entries = t
        i = bisect.bisect_right(rvas, addr) - 1
        if i >= 0 and addr < entries[i]["rva"] + entries[i]["size"]:
            return strings[entries[i]["symbol"]]
        return f"{lib_name}+{addr:#x}"

    self_c = collections.Counter()
    incl_c = collections.Counter()
    total = 0
    for t in prof["threads"]:
        st, ft, res = t["stackTable"], t["frameTable"], t["resourceTable"]
        libs = prof["libs"]
        names = []
        for i in range(ft["length"]):
            func = ft["func"][i]
            r = t["funcTable"]["resource"][func]
            lib = libs[res["lib"][r]]["debugName"] if r is not None and r >= 0 else "?"
            names.append(lookup(lib, ft["address"][i]))
        cache = {}

        def chain(s):
            if s in cache:
                return cache[s]
            out = []
            while s is not None:
                out.append(names[st["frame"][s]])
                s = st["prefix"][s]
            cache[s] = out
            return out

        for s in t["samples"]["stack"]:
            if s is None:
                continue
            c = chain(s)
            total += 1
            self_c[c[0]] += 1
            if callers_of:
                for k in range(len(c) - 1):
                    if callers_of in c[k]:
                        callers[c[k + 1]] += 1
                        break
            for n in set(c):
                incl_c[n] += 1
    print(f"{total} samples")
    if callers_of:
        for name, n in callers.most_common(top):
            print(f"{100 * n / total:6.2f}%  {name[:110]}")
        return
    for title, c in (("self", self_c), ("inclusive", incl_c)):
        print(f"\n-- {title}")
        for name, n in c.most_common(top):
            print(f"{100 * n / total:6.2f}%  {name[:110]}")


main()
