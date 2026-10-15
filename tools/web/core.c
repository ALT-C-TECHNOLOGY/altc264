// The player's media core: FFmpeg demuxes, libaltc264 decodes the video
// (both MVC views side by side), FFmpeg decodes and resamples the audio.
// The page's worker calls the exports and supplies file reads (host.read).
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/stereo3d.h>
#include <libswresample/swresample.h>
#include <stdlib.h>

#define EXPORT(name) __attribute__((export_name(#name)))

__attribute__((import_module("host"), import_name("read"))) int host_read(double pos, int len, uint8_t *buf);

static AVFormatContext *fmt;
static AVCodecContext *vc, *ac;
static SwrContext *swr;
static AVPacket *pkt;
static AVFrame *frame, *audio;
static int vs = -1, as = -1, out_rate, eof;
static double pos, size;
static float *pcm;
static int pcm_cap;

// Out: the last video frame (planes, strides, size, pts, side by side) or
// audio chunk (interleaved stereo float, sample count, pts).
static struct {
	uint8_t *data[3];
	int stride[3], width, height, chroma_w, chroma_h, sbs;
	double pts;
	float *pcm;
	int samples;
	double audio_pts;
	int bits;
} out;

static int read_cb(void *opaque, uint8_t *buf, int n) {
	if (pos >= size) return AVERROR_EOF;
	int r = host_read(pos, n, buf);
	pos += r;
	return r > 0 ? r : AVERROR_EOF;
}

static int64_t seek_cb(void *opaque, int64_t to, int whence) {
	whence &= ~AVSEEK_FORCE;
	if (whence == AVSEEK_SIZE) return size;
	pos = whence == SEEK_SET ? to : whence == SEEK_CUR ? pos + to : size + to;
	return pos;
}

static double ts(AVStream *s, int64_t t) {
	return t == AV_NOPTS_VALUE ? -1 : t * av_q2d(s->time_base);
}

static void close_all(void) {
	if (!fmt) return;
	AVIOContext *pb = fmt->pb;
	avcodec_free_context(&vc);
	avcodec_free_context(&ac);
	swr_free(&swr);
	avformat_close_input(&fmt);
	av_freep(&pb->buffer);
	avio_context_free(&pb);
	av_packet_free(&pkt);
	av_frame_free(&frame);
	av_frame_free(&audio);
}

EXPORT(core_open) int core_open(double file_size, int threads, int rate) {
	close_all();
	// FFmpeg's H.264 parser logs MVC's subset SPS as errors while probing.
	av_log_set_level(AV_LOG_FATAL);
	size = file_size, pos = 0, eof = 0, out_rate = rate;
	fmt = avformat_alloc_context();
	fmt->pb = avio_alloc_context(av_malloc(1 << 18), 1 << 18, 0, NULL, read_cb, NULL, seek_cb);
	if (avformat_open_input(&fmt, NULL, NULL, NULL) < 0) return -1;
	avformat_find_stream_info(fmt, NULL);
	vs = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
	if (vs < 0 || fmt->streams[vs]->codecpar->codec_id != AV_CODEC_ID_H264) return -2;
	vc = avcodec_alloc_context3(avcodec_find_decoder_by_name("libaltc264"));
	avcodec_parameters_to_context(vc, fmt->streams[vs]->codecpar);
	av_opt_set_int(vc->priv_data, "altc264_threads", threads, 0);
	av_opt_set_int(vc->priv_data, "all_views", 1, 0);
	av_opt_set(vc->priv_data, "stereo_layout", "sbs", 0);
	if (avcodec_open2(vc, NULL, NULL) < 0) return -3;
	as = av_find_best_stream(fmt, AVMEDIA_TYPE_AUDIO, -1, vs, NULL, 0);
	const AVCodec *c = as >= 0 ? avcodec_find_decoder(fmt->streams[as]->codecpar->codec_id) : NULL;
	if (c) {
		ac = avcodec_alloc_context3(c);
		avcodec_parameters_to_context(ac, fmt->streams[as]->codecpar);
		if (avcodec_open2(ac, c, NULL) < 0) avcodec_free_context(&ac);
	}
	if (ac) {
		AVChannelLayout stereo = AV_CHANNEL_LAYOUT_STEREO;
		swr_alloc_set_opts2(&swr, &stereo, AV_SAMPLE_FMT_FLT, rate, &ac->ch_layout, ac->sample_fmt, ac->sample_rate, 0, NULL);
		// Downmixes without clipping.
		av_opt_set_double(swr, "rematrix_maxval", 1.0, 0);
		if (swr_init(swr) < 0) swr_free(&swr), avcodec_free_context(&ac);
	}
	pkt = av_packet_alloc();
	frame = av_frame_alloc();
	audio = av_frame_alloc();
	return 0;
}

EXPORT(core_duration) double core_duration(void) {
	return fmt->duration > 0 ? fmt->duration / 1e6 : 0;
}

EXPORT(core_fps) double core_fps(void) {
	AVRational r = fmt->streams[vs]->avg_frame_rate;
	return r.den ? av_q2d(r) : 0;
}

EXPORT(core_video_codec) const char *core_video_codec(void) {
	return vs < 0 ? "" : avcodec_get_name(fmt->streams[vs]->codecpar->codec_id);
}

EXPORT(core_audio_codec) const char *core_audio_codec(void) {
	return as < 0 ? "" : ac ? avcodec_get_name(ac->codec_id) : "unsupported";
}

static int take_video(void) {
	if (avcodec_receive_frame(vc, frame) < 0) return 0;
	out.width = frame->width, out.height = frame->height;
	const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(frame->format);
	out.chroma_w = AV_CEIL_RSHIFT(frame->width, d->log2_chroma_w);
	out.chroma_h = AV_CEIL_RSHIFT(frame->height, d->log2_chroma_h);
	for (int i = 0; i < 3; i++) out.data[i] = frame->data[i], out.stride[i] = frame->linesize[i];
	const AVFrameSideData *sd = av_frame_get_side_data(frame, AV_FRAME_DATA_STEREO3D);
	out.sbs = sd && ((AVStereo3D *)sd->data)->type == AV_STEREO3D_SIDEBYSIDE &&
		frame->width == 2 * fmt->streams[vs]->codecpar->width;
	out.bits = d->comp[0].depth;
	out.pts = ts(fmt->streams[vs], frame->best_effort_timestamp);
	return 1;
}

static int take_audio(void) {
	if (!ac || avcodec_receive_frame(ac, audio) < 0) return 0;
	int n = swr_get_out_samples(swr, audio->nb_samples);
	if (n * 2 > pcm_cap) pcm = realloc(pcm, (pcm_cap = n * 2) * sizeof(float));
	uint8_t *dst = (uint8_t *)pcm;
	out.samples = swr_convert(swr, &dst, n, (const uint8_t **)audio->extended_data, audio->nb_samples);
	out.pcm = pcm;
	out.audio_pts = ts(fmt->streams[as], audio->best_effort_timestamp);
	av_frame_unref(audio);
	return out.samples > 0;
}

// The next output: 1 video frame, 2 audio chunk, 0 end of stream.
EXPORT(core_next) int core_next(void) {
	av_frame_unref(frame);
	for (;;) {
		if (take_video()) return 1;
		if (take_audio()) return 2;
		if (eof) return 0;
		if (av_read_frame(fmt, pkt) < 0) {
			eof = 1;
			avcodec_send_packet(vc, NULL);
			if (ac) avcodec_send_packet(ac, NULL);
			continue;
		}
		if (pkt->stream_index == vs) avcodec_send_packet(vc, pkt);
		else if (ac && pkt->stream_index == as) avcodec_send_packet(ac, pkt);
		av_packet_unref(pkt);
	}
}

EXPORT(core_seek) void core_seek(double t) {
	av_seek_frame(fmt, -1, (int64_t)(t * AV_TIME_BASE), AVSEEK_FLAG_BACKWARD);
	avcodec_flush_buffers(vc);
	if (ac) avcodec_flush_buffers(ac), swr_init(swr);
	eof = 0;
}

EXPORT(core_out) void *core_out(void) {
	return &out;
}
