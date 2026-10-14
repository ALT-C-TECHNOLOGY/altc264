// Decode-only timing driver for FFmpeg's h264 decoder (libavcodec): one packet
// per access unit, file loaded before the clock starts. See BENCHMARKS.md.
#include "common.h"
#include <libavcodec/avcodec.h>

static void drain(AVCodecContext *ctx, AVFrame *fr, Stats *st, const Args *a) {
	while (avcodec_receive_frame(ctx, fr) == 0) {
		const uint8_t *planes[3] = {fr->data[0], fr->data[1], fr->data[2]};
		int strides[3] = {fr->linesize[0], fr->linesize[1], fr->linesize[2]};
		emit_picture(st, a, 0, planes, strides, fr->width, fr->height);
		av_frame_unref(fr);
	}
}

int main(int argc, char **argv) {
	Args a = parse_args(argc, argv);
	Stream s = load_stream(&a);
	Stats st;
	stats_open(&st, &a);
	const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_H264);
	AVCodecContext *ctx = avcodec_alloc_context3(codec);
	ctx->thread_count = a.threads < 0 ? 0 : a.threads == 0 ? 1 : a.threads; // 0 = auto in lavc
	ctx->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
	if (avcodec_open2(ctx, codec, NULL) < 0) die("avcodec_open2 failed");
	AVPacket *pkt = av_packet_alloc();
	AVFrame *fr = av_frame_alloc();
	stats_start(&st);
	for (size_t i = 0; i < s.n_aus; i++) {
		pkt->data = s.aus[i].buf;
		pkt->size = (int)s.aus[i].buf_len;
		if (!pkt->size) continue;
		while (avcodec_send_packet(ctx, pkt) == AVERROR(EAGAIN)) drain(ctx, fr, &st, &a);
		drain(ctx, fr, &st, &a);
	}
	avcodec_send_packet(ctx, NULL);
	drain(ctx, fr, &st, &a);
	stats_stop(&st);
	av_frame_free(&fr);
	av_packet_free(&pkt);
	avcodec_free_context(&ctx);
	report("ffmpeg", &a, &s, &st);
	return 0;
}
