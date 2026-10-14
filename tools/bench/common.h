// Shared harness for the decode-only timing driver (ffmpeg-driver.c).
//
// Every driver loads the whole Annex B file and splits it into access units
// before the clock starts, so only decoding is timed. Output is one JSON line.
#ifndef BENCH_COMMON_H
#define BENCH_COMMON_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>

typedef struct { const uint8_t *p; size_t len; int type; } Nal;   // p points past the start code
typedef struct { size_t first, count; uint8_t *buf; size_t buf_len; } Au; // buf: AU re-joined with 4-byte start codes

typedef struct {
	const char *path;
	int threads;      // 0 = decoder default single-thread, -1 = auto
	int base_only;    // drop MVC NALs (14, 15, 20) so all decoders see the same 2D stream
	int hash;         // hash output pictures (slows decoding)
	const char *out[2]; // optional raw YUV dumps per view
} Args;

typedef struct {
	uint8_t *data; size_t size;
	Nal *nals; size_t n_nals;
	Au *aus; size_t n_aus;
} Stream;

typedef struct {
	uint64_t frames[2]; // output pictures per view
	uint64_t hash[2];
	FILE *out[2];
	double t0, c0, wall, cpu;
} Stats;

static void die(const char *msg) { fprintf(stderr, "error: %s\n", msg); exit(1); }

static Args parse_args(int argc, char **argv) {
	Args a = {0};
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--threads") && i + 1 < argc) a.threads = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--base-only")) a.base_only = 1;
		else if (!strcmp(argv[i], "--hash")) a.hash = 1;
		else if (!strcmp(argv[i], "--out") && i + 1 < argc) a.out[0] = argv[++i];
		else if (!strcmp(argv[i], "--out2") && i + 1 < argc) a.out[1] = argv[++i];
		else if (argv[i][0] != '-' && !a.path) a.path = argv[i];
		else die("usage: driver FILE.264 [--threads N] [--base-only] [--hash] [--out base.yuv] [--out2 dep.yuv]");
	}
	if (!a.path) die("no input file");
	return a;
}

// Types >= 0x100 are dependent-view PPS/SEI tagged by split_aus.
static int is_mvc_nal(int t) { return t == 14 || t == 15 || t == 20 || t >= 0x100; }
static int is_vcl(int t) { return t == 1 || t == 5 || t == 20 || t == 21; }

// first_mb_in_slice == 0 is coded as a single '1' bit right after the header.
static int starts_picture(const Nal *n) {
	if (n->type == 1 || n->type == 5) return n->len > 1 && (n->p[1] & 0x80);
	return 0;
}

static void split_nals(Stream *s) {
	size_t cap = 1024;
	s->nals = (Nal *)malloc(cap * sizeof(Nal));
	const uint8_t *d = s->data, *end = d + s->size;
	const uint8_t *cur = NULL;
	for (const uint8_t *p = d; p + 3 <= end; p++) {
		if (p[0] || p[1] || p[2] != 1) continue;
		if (cur) {
			const uint8_t *e = p;
			while (e > cur && e[-1] == 0) e--; // trailing zeros belong to the next start code
			if (s->n_nals == cap) s->nals = (Nal *)realloc(s->nals, (cap *= 2) * sizeof(Nal));
			s->nals[s->n_nals++] = (Nal){cur, (size_t)(e - cur), cur[0] & 31};
		}
		cur = p + 3;
		p += 2;
	}
	if (cur && cur < end) {
		if (s->n_nals == cap) s->nals = (Nal *)realloc(s->nals, (cap + 1) * sizeof(Nal));
		s->nals[s->n_nals++] = (Nal){cur, (size_t)(end - cur), cur[0] & 31};
	}
}

// Access unit boundaries per 7.4.1.2.3, tolerating the Blu-ray layout where the
// dependent view's SPS/PPS/SEI sit between the base and non-base slices.
static void split_aus(Stream *s, int base_only) {
	s->aus = (Au *)calloc(s->n_nals + 1, sizeof(Au));
	size_t start = 0, pending = (size_t)-1;
	int has_vcl = 0;
	for (size_t i = 0; i < s->n_nals; i++) {
		const Nal *n = &s->nals[i];
		int t = n->type;
		int boundary = 0;
		if (has_vcl) {
			// No SPS here: Blu-ray dependent views repeat it after the subset SPS.
			if (t == 9 || t == 14) boundary = 1;
			else if (starts_picture(n)) boundary = 1;
		}
		if (boundary) {
			size_t b = pending != (size_t)-1 ? pending : i;
			s->aus[s->n_aus++] = (Au){start, b - start, NULL, 0};
			start = b;
			has_vcl = 0;
			pending = (size_t)-1;
		}
		if (is_vcl(t)) { has_vcl = 1; pending = (size_t)-1; }
		else if (has_vcl && pending == (size_t)-1 && (t == 6 || t == 8 || t == 9 || t == 7 || t == 14)) pending = i;
	}
	if (start < s->n_nals) s->aus[s->n_aus++] = (Au){start, s->n_nals - start, NULL, 0};

	for (size_t a = 0; a < s->n_aus; a++) {
		Au *au = &s->aus[a];
		// In base-only mode, also drop the SPS/PPS/SEI that follow the base picture's
		// slices: in a plain AVC access unit nothing may sit there, so they belong
		// to the dependent view (Blu-ray layout) and confuse 2D-only decoders.
		int seen_base = 0;
		for (size_t k = 0; k < au->count; k++) {
			Nal *n = &s->nals[au->first + k];
			if (n->type == 1 || n->type == 5) seen_base = 1;
			else if (base_only && seen_base && (n->type == 6 || n->type == 7 || n->type == 8)) n->type |= 0x100;
		}
		size_t len = 0;
		for (size_t k = 0; k < au->count; k++) {
			const Nal *n = &s->nals[au->first + k];
			if (!(base_only && is_mvc_nal(n->type))) len += 4 + n->len;
		}
		au->buf = (uint8_t *)malloc(len + 64); // padding for decoders that over-read
		memset(au->buf + len, 0, 64);
		uint8_t *w = au->buf;
		for (size_t k = 0; k < au->count; k++) {
			const Nal *n = &s->nals[au->first + k];
			if (base_only && is_mvc_nal(n->type)) continue;
			memcpy(w, "\0\0\0\1", 4);
			memcpy(w + 4, n->p, n->len);
			w += 4 + n->len;
		}
		au->buf_len = len;
	}
}

static Stream load_stream(const Args *a) {
	Stream s = {0};
	FILE *f = fopen(a->path, "rb");
	if (!f) die("cannot open input");
	fseek(f, 0, SEEK_END);
	s.size = (size_t)ftell(f);
	fseek(f, 0, SEEK_SET);
	s.data = (uint8_t *)malloc(s.size + 64);
	if (fread(s.data, 1, s.size, f) != s.size) die("short read");
	memset(s.data + s.size, 0, 64);
	fclose(f);
	split_nals(&s);
	split_aus(&s, a->base_only);
	return s;
}

static double now_wall(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double now_cpu(void) {
	struct rusage ru;
	getrusage(RUSAGE_SELF, &ru);
	return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6 + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
}

static void stats_open(Stats *st, const Args *a) {
	memset(st, 0, sizeof(*st));
	st->hash[0] = st->hash[1] = 0xcbf29ce484222325ull;
	for (int v = 0; v < 2; v++)
		if (a->out[v] && !(st->out[v] = fopen(a->out[v], "wb"))) die("cannot open output");
}

static void stats_start(Stats *st) { st->t0 = now_wall(); st->c0 = now_cpu(); }
static void stats_stop(Stats *st) { st->wall = now_wall() - st->t0; st->cpu = now_cpu() - st->c0; }

// Hash and/or dump one cropped 8-bit 4:2:0 picture.
static void emit_picture(Stats *st, const Args *a, int view, const uint8_t *const planes[3],
                         const int strides[3], int w, int h) {
	st->frames[view]++;
	if (!a->hash && !st->out[view]) return;
	for (int c = 0; c < 3; c++) {
		int pw = c ? (w + 1) / 2 : w, ph = c ? (h + 1) / 2 : h;
		for (int y = 0; y < ph; y++) {
			const uint8_t *row = planes[c] + (size_t)y * strides[c];
			if (a->hash) {
				uint64_t x = st->hash[view];
				for (int i = 0; i < pw; i++) x = (x ^ row[i]) * 0x100000001b3ull;
				st->hash[view] = x;
			}
			if (st->out[view]) fwrite(row, 1, pw, st->out[view]);
		}
	}
}

static void report(const char *decoder, const Args *a, const Stream *s, Stats *st) {
	for (int v = 0; v < 2; v++) if (st->out[v]) fclose(st->out[v]);
	printf("{\"decoder\":\"%s\",\"file\":\"%s\",\"threads\":%d,\"base_only\":%s,"
	       "\"aus\":%zu,\"frames\":%llu,\"frames_dep\":%llu,\"wall_s\":%.6f,\"cpu_s\":%.6f,\"fps\":%.3f",
	       decoder, a->path, a->threads, a->base_only ? "true" : "false", s->n_aus,
	       (unsigned long long)st->frames[0], (unsigned long long)st->frames[1], st->wall, st->cpu,
	       st->wall > 0 ? st->frames[0] / st->wall : 0.0);
	if (a->hash) {
		printf(",\"hash\":\"%016llx\"", (unsigned long long)st->hash[0]);
		if (st->frames[1]) printf(",\"hash_dep\":\"%016llx\"", (unsigned long long)st->hash[1]);
	}
	printf("}\n");
}

#endif
