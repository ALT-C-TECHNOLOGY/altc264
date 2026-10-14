use gst::glib;
use gst::prelude::*;

mod imp;

glib::wrapper! {
    pub struct Altc264Dec(ObjectSubclass<imp::Altc264Dec>)
        @extends gst_video::VideoDecoder, gst::Element, gst::Object;
}

/// How the views of a stereo stream are output.
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Clone, Copy, glib::Enum, Default)]
#[repr(u32)]
#[enum_type(name = "GstAltc264StereoLayout")]
pub enum StereoLayout {
    #[enum_value(name = "Base view only (2D)", nick = "mono")]
    Mono = 0,
    #[default]
    #[enum_value(name = "Side by side, full resolution", nick = "side-by-side")]
    SideBySide = 1,
    #[enum_value(name = "Side by side, half width", nick = "side-by-side-half")]
    SideBySideHalf = 2,
    #[enum_value(name = "Top and bottom, full resolution", nick = "top-bottom")]
    TopBottom = 3,
    #[enum_value(name = "Top and bottom, half height", nick = "top-bottom-half")]
    TopBottomHalf = 4,
}

/// Which eye the MVC base view is.
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Clone, Copy, glib::Enum, Default)]
#[repr(u32)]
#[enum_type(name = "GstAltc264BaseView")]
pub enum BaseView {
    #[default]
    #[enum_value(name = "From the upstream multiview caps", nick = "auto")]
    Auto = 0,
    #[enum_value(name = "Left eye", nick = "left")]
    Left = 1,
    #[enum_value(name = "Right eye", nick = "right")]
    Right = 2,
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    StereoLayout::static_type().mark_as_plugin_api(gst::PluginAPIFlags::empty());
    BaseView::static_type().mark_as_plugin_api(gst::PluginAPIFlags::empty());
    gst::Element::register(
        Some(plugin),
        "altc264dec",
        gst::Rank::MARGINAL,
        Altc264Dec::static_type(),
    )
}
