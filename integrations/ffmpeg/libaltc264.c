/*
 * H.264 / MVC decoding via libaltc264
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <altc264.h>

#include "libavutil/cpu.h"
#include "libavutil/imgutils.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/stereo3d.h"

#include "avcodec.h"
#include "codec_internal.h"
#include "decode.h"

/* Per-frame Blu-ray 3D plane offsets are exported as
 * AV_FRAME_DATA_SEI_UNREGISTERED with this UUID, followed by one byte per
 * offset sequence: bit 7 = plane_offset_direction_flag, bits 0-6 =
 * plane_offset_value. Subtitle tracks select a sequence by their 3d-plane
 * number. */
static const uint8_t plane_offsets_uuid[16] = {
    0xa1, 0x7c, 0x26, 0x4e, 0x3d, 0x2b, 0x4f, 0x0e,
    0x9b, 0x52, 0x6c, 0x21, 0x64, 0x70, 0x6c, 0x6f,
};

typedef struct LibAltc264Context {
    const AVClass *class;
    Altc264Decoder *dec;
    int threads;
    int all_views;
    int stereo_layout;
    int base_view;
    int base_right;
    int drained;
} LibAltc264Context;

static av_cold int libaltc264_init(AVCodecContext *avctx)
{
    LibAltc264Context *s = avctx->priv_data;
    Altc264Settings settings;
    int ret;

    altc264_default_settings(&settings);
    /* 0: the codec's thread count, or all CPUs. */
    if (s->threads > 0)
        settings.threads = s->threads;
    else if (avctx->thread_count > 0)
        settings.threads = avctx->thread_count;
    else
        settings.threads = av_cpu_count();
    settings.all_views     = s->all_views || s->stereo_layout != ALTC264_STEREO_SEPARATE;
    settings.stereo_layout = s->stereo_layout;
    if (s->base_view < 0) {
        /* Auto: the container's stereo info, e.g. Matroska block_rl. */
        const AVPacketSideData *sd = av_packet_side_data_get(avctx->coded_side_data,
                                                             avctx->nb_coded_side_data,
                                                             AV_PKT_DATA_STEREO3D);
        s->base_right = sd && (((const AVStereo3D *)sd->data)->flags & AV_STEREO3D_FLAG_INVERT);
    } else {
        s->base_right = s->base_view;
    }
    settings.base_view_right = s->base_right;
    ret = altc264_open(&s->dec, &settings);
    if (ret < 0)
        return AVERROR_EXTERNAL;
    if (avctx->extradata_size > 0)
        altc264_set_extradata(s->dec, avctx->extradata, avctx->extradata_size);
    av_log(avctx, AV_LOG_VERBOSE, "libaltc264 %s\n", altc264_version());
    return 0;
}

static av_cold int libaltc264_close(AVCodecContext *avctx)
{
    LibAltc264Context *s = avctx->priv_data;
    altc264_close(&s->dec);
    return 0;
}

static void libaltc264_flush(AVCodecContext *avctx)
{
    LibAltc264Context *s = avctx->priv_data;
    altc264_reset(s->dec);
    s->drained = 0;
}

/* Planar formats by layout (gray, 4:2:0, 4:2:2) and bit depth. 11 and 13
 * bit samples go in the next wider format. */
static int pick_pix_fmt(AVCodecContext *avctx, const Altc264Picture *p)
{
    static const enum AVPixelFormat fmts[3][7] = {
        { AV_PIX_FMT_GRAY8, AV_PIX_FMT_GRAY9, AV_PIX_FMT_GRAY10, AV_PIX_FMT_GRAY12,
          AV_PIX_FMT_GRAY12, AV_PIX_FMT_GRAY14, AV_PIX_FMT_GRAY14 },
        { AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV420P9, AV_PIX_FMT_YUV420P10, AV_PIX_FMT_YUV420P12,
          AV_PIX_FMT_YUV420P12, AV_PIX_FMT_YUV420P14, AV_PIX_FMT_YUV420P14 },
        { AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV422P9, AV_PIX_FMT_YUV422P10, AV_PIX_FMT_YUV422P12,
          AV_PIX_FMT_YUV422P12, AV_PIX_FMT_YUV422P14, AV_PIX_FMT_YUV422P14 },
    };
    int layout = !p->data[1] ? 0 : p->chroma_height == p->height ? 2 : 1;
    int depth = FFMAX(p->bit_depth_luma, p->data[1] ? p->bit_depth_chroma : 0);
    if (depth < 8 || depth > 14)
        return AVERROR_PATCHWELCOME;
    avctx->pix_fmt = fmts[layout][depth - 8];
    avctx->bits_per_raw_sample = depth;
    return 0;
}

static int output_picture(AVCodecContext *avctx, AVFrame *frame, Altc264Picture *p)
{
    LibAltc264Context *s = avctx->priv_data;
    const uint8_t *src[4] = { p->data[0], p->data[1], p->data[2], NULL };
    int src_stride[4] = { (int)p->stride[0], (int)p->stride[1], (int)p->stride[2], 0 };
    int ret;

    if ((ret = pick_pix_fmt(avctx, p)) < 0)
        return ret;
    if ((ret = ff_set_dimensions(avctx, p->width, p->height)) < 0)
        return ret;
    if ((ret = ff_get_buffer(avctx, frame, 0)) < 0)
        return ret;
    av_image_copy2(frame->data, frame->linesize, src, src_stride,
                   avctx->pix_fmt, p->width, p->height);
    frame->pts = p->has_pts ? p->pts : AV_NOPTS_VALUE;

    if (s->stereo_layout != ALTC264_STEREO_SEPARATE) {
        /* One packed frame per access unit. Frame packing has its gap
         * between the views, which AVStereo3D cannot describe. */
        int type = -1;
        switch (s->stereo_layout) {
        case ALTC264_STEREO_SIDE_BY_SIDE:
        case ALTC264_STEREO_SIDE_BY_SIDE_HALF:   type = AV_STEREO3D_SIDEBYSIDE; break;
        case ALTC264_STEREO_TOP_AND_BOTTOM:
        case ALTC264_STEREO_TOP_AND_BOTTOM_HALF: type = AV_STEREO3D_TOPBOTTOM;  break;
        }
        av_frame_remove_side_data(frame, AV_FRAME_DATA_STEREO3D);
        if (type >= 0) {
            AVStereo3D *stereo = av_stereo3d_create_side_data(frame);
            if (!stereo)
                return AVERROR(ENOMEM);
            stereo->type = type;
        }
    } else if (s->all_views) {
        AVFrameSideData *sd = av_frame_side_data_new(&frame->side_data, &frame->nb_side_data,
                                                     AV_FRAME_DATA_VIEW_ID, sizeof(int), 0);
        AVStereo3D *stereo;
        if (!sd)
            return AVERROR(ENOMEM);
        *(int *)sd->data = p->view_id;
        /* Replace the container's stream-level stereo info, if any. */
        av_frame_remove_side_data(frame, AV_FRAME_DATA_STEREO3D);
        stereo = av_stereo3d_create_side_data(frame);
        if (!stereo)
            return AVERROR(ENOMEM);
        /* Which eye the base view is comes from outside the elementary
         * stream (the Blu-ray playlist, carried by the container). */
        stereo->type = AV_STEREO3D_FRAMESEQUENCE;
        stereo->view = (p->view_id == 0) != s->base_right ? AV_STEREO3D_VIEW_LEFT
                                                          : AV_STEREO3D_VIEW_RIGHT;
    }

    if (p->num_plane_offsets > 0) {
        AVFrameSideData *sd = av_frame_new_side_data(frame, AV_FRAME_DATA_SEI_UNREGISTERED,
                                                     sizeof(plane_offsets_uuid) + p->num_plane_offsets);
        if (!sd)
            return AVERROR(ENOMEM);
        memcpy(sd->data, plane_offsets_uuid, sizeof(plane_offsets_uuid));
        for (int i = 0; i < p->num_plane_offsets; i++)
            sd->data[sizeof(plane_offsets_uuid) + i] =
                (p->plane_offsets[i].direction ? 0x80 : 0) | (p->plane_offsets[i].value & 0x7f);
    }
    return 0;
}

static int libaltc264_receive_frame(AVCodecContext *avctx, AVFrame *frame)
{
    LibAltc264Context *s = avctx->priv_data;
    Altc264Picture p;
    int ret;

    for (;;) {
        if (altc264_get_picture(s->dec, &p) == ALTC264_OK) {
            ret = output_picture(avctx, frame, &p);
            altc264_picture_unref(&p);
            return ret;
        }
        if (s->drained)
            return AVERROR_EOF;

        AVPacket *pkt = av_packet_alloc();
        if (!pkt)
            return AVERROR(ENOMEM);
        ret = ff_decode_get_packet(avctx, pkt);
        if (ret == AVERROR_EOF) {
            av_packet_free(&pkt);
            altc264_drain(s->dec);
            s->drained = 1;
            continue;
        }
        if (ret < 0) {
            av_packet_free(&pkt);
            return ret;
        }
        altc264_send_data(s->dec, pkt->data, pkt->size, pkt->pts, pkt->pts != AV_NOPTS_VALUE);
        av_packet_free(&pkt);
    }
}

#define OFFSET(x) offsetof(LibAltc264Context, x)
#define VD AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_DECODING_PARAM
static const AVOption libaltc264_options[] = {
    { "altc264_threads", "Decoding threads (0: auto)", OFFSET(threads), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 256, VD },
    { "all_views", "Decode and output every MVC view (frames carry view id side data)", OFFSET(all_views), AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, VD },
    { "stereo_layout", "Output both views of a stereo stream as one packed frame", OFFSET(stereo_layout), AV_OPT_TYPE_INT, { .i64 = ALTC264_STEREO_SEPARATE }, 0, 5, VD, .unit = "stereo_layout" },
        { "separate",      "One frame per view",                  0, AV_OPT_TYPE_CONST, { .i64 = ALTC264_STEREO_SEPARATE },            0, 0, VD, .unit = "stereo_layout" },
        { "sbs",           "Side by side, full resolution",       0, AV_OPT_TYPE_CONST, { .i64 = ALTC264_STEREO_SIDE_BY_SIDE },        0, 0, VD, .unit = "stereo_layout" },
        { "sbs_half",      "Side by side, half width",            0, AV_OPT_TYPE_CONST, { .i64 = ALTC264_STEREO_SIDE_BY_SIDE_HALF },   0, 0, VD, .unit = "stereo_layout" },
        { "tab",           "Top and bottom, full resolution",     0, AV_OPT_TYPE_CONST, { .i64 = ALTC264_STEREO_TOP_AND_BOTTOM },      0, 0, VD, .unit = "stereo_layout" },
        { "tab_half",      "Top and bottom, half height",         0, AV_OPT_TYPE_CONST, { .i64 = ALTC264_STEREO_TOP_AND_BOTTOM_HALF }, 0, 0, VD, .unit = "stereo_layout" },
        { "frame_packing", "HDMI 1.4 frame packing",              0, AV_OPT_TYPE_CONST, { .i64 = ALTC264_STEREO_FRAME_PACKING },       0, 0, VD, .unit = "stereo_layout" },
    { "base_view", "Eye of the MVC base view", OFFSET(base_view), AV_OPT_TYPE_INT, { .i64 = -1 }, -1, 1, VD, .unit = "base_view" },
        { "auto",  "From the container's stereo info", 0, AV_OPT_TYPE_CONST, { .i64 = -1 }, 0, 0, VD, .unit = "base_view" },
        { "left",  "Left eye",                         0, AV_OPT_TYPE_CONST, { .i64 = 0 },  0, 0, VD, .unit = "base_view" },
        { "right", "Right eye",                        0, AV_OPT_TYPE_CONST, { .i64 = 1 },  0, 0, VD, .unit = "base_view" },
    { NULL }
};

static const AVClass libaltc264_class = {
    .class_name = "libaltc264 decoder",
    .item_name  = av_default_item_name,
    .option     = libaltc264_options,
    .version    = LIBAVUTIL_VERSION_INT,
};

const FFCodec ff_libaltc264_decoder = {
    .p.name         = "libaltc264",
    CODEC_LONG_NAME("libaltc264 H.264 / MVC"),
    .p.type         = AVMEDIA_TYPE_VIDEO,
    .p.id           = AV_CODEC_ID_H264,
    .priv_data_size = sizeof(LibAltc264Context),
    .init           = libaltc264_init,
    .close          = libaltc264_close,
    .flush          = libaltc264_flush,
    FF_CODEC_RECEIVE_FRAME_CB(libaltc264_receive_frame),
    .p.capabilities = AV_CODEC_CAP_DELAY | AV_CODEC_CAP_OTHER_THREADS,
    .caps_internal  = FF_CODEC_CAP_NOT_INIT_THREADSAFE | FF_CODEC_CAP_AUTO_THREADS,
    .p.priv_class   = &libaltc264_class,
    .p.wrapper_name = "libaltc264",
};
