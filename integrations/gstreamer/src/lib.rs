//! GStreamer plugin with the `altc264dec` element: H.264 and MVC (stereo
//! 3D) decoding with altc264.
use gst::glib;

mod dec;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    dec::register(plugin)
}

gst::plugin_define!(
    altc264,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    env!("CARGO_PKG_VERSION"),
    "MIT/X11",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    "https://example.invalid/altc264",
    "2026-10-05"
);
