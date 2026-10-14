//! Decoded picture sample buffers.

/// A field of a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Parity {
    Top = 0,
    Bottom = 1,
}

impl Parity {
    pub fn other(self) -> Self {
        match self {
            Parity::Top => Parity::Bottom,
            Parity::Bottom => Parity::Top,
        }
    }
}

/// A sample type: u8 for 8-bit video, u16 for bit depths 9 to 14. Planes
/// hold both buffers and use the one of their depth, so code generic over
/// the sample type picks its buffer with [`Pixel::of`].
pub trait Pixel: Copy + Default + PartialEq + Send + Sync + std::fmt::Debug + 'static {
    /// More than 8 bits per sample.
    const HIGH: bool;
    fn get(self) -> i32;
    /// A value already in range.
    fn new(v: i32) -> Self;
    /// Clips to 0..=max, where max is (1 << BitDepth) - 1.
    fn clip(v: i32, max: i32) -> Self;
    fn of(p: &Plane) -> &[Self];
    fn of_mut(p: &mut Plane) -> &mut [Self];
    fn vec_mut(p: &mut Plane) -> &mut Vec<Self>;
    /// The first sample of a view of samples of this type.
    fn view_ptr(v: &PlaneView) -> *const Self;
    /// The same samples as u8 (8-bit only) or u16 (high bit depth only),
    /// to reach the kernels of each depth.
    fn u8s(s: &[Self]) -> &[u8];
    fn u8s_mut(s: &mut [Self]) -> &mut [u8];
    fn u16s(s: &[Self]) -> &[u16];
    fn u16s_mut(s: &mut [Self]) -> &mut [u16];
}

impl Pixel for u8 {
    const HIGH: bool = false;
    #[inline(always)]
    fn get(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn new(v: i32) -> Self {
        v as u8
    }
    #[inline(always)]
    fn clip(v: i32, _max: i32) -> Self {
        v.clamp(0, 255) as u8
    }
    #[inline(always)]
    fn of(p: &Plane) -> &[u8] {
        &p.data
    }
    #[inline(always)]
    fn of_mut(p: &mut Plane) -> &mut [u8] {
        &mut p.data
    }
    #[inline(always)]
    fn vec_mut(p: &mut Plane) -> &mut Vec<u8> {
        &mut p.data
    }
    #[inline(always)]
    fn view_ptr(v: &PlaneView) -> *const u8 {
        v.ptr
    }
    #[inline(always)]
    fn u8s(s: &[u8]) -> &[u8] {
        s
    }
    #[inline(always)]
    fn u8s_mut(s: &mut [u8]) -> &mut [u8] {
        s
    }
    fn u16s(_: &[u8]) -> &[u16] {
        unreachable!("8-bit samples")
    }
    fn u16s_mut(_: &mut [u8]) -> &mut [u16] {
        unreachable!("8-bit samples")
    }
}

impl Pixel for u16 {
    const HIGH: bool = true;
    #[inline(always)]
    fn get(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn new(v: i32) -> Self {
        v as u16
    }
    #[inline(always)]
    fn clip(v: i32, max: i32) -> Self {
        v.clamp(0, max) as u16
    }
    #[inline(always)]
    fn of(p: &Plane) -> &[u16] {
        &p.data16
    }
    #[inline(always)]
    fn of_mut(p: &mut Plane) -> &mut [u16] {
        &mut p.data16
    }
    #[inline(always)]
    fn vec_mut(p: &mut Plane) -> &mut Vec<u16> {
        &mut p.data16
    }
    #[inline(always)]
    fn view_ptr(v: &PlaneView) -> *const u16 {
        v.ptr16
    }
    fn u8s(_: &[u16]) -> &[u8] {
        unreachable!("high bit depth samples")
    }
    fn u8s_mut(_: &mut [u16]) -> &mut [u8] {
        unreachable!("high bit depth samples")
    }
    #[inline(always)]
    fn u16s(s: &[u16]) -> &[u16] {
        s
    }
    #[inline(always)]
    fn u16s_mut(s: &mut [u16]) -> &mut [u16] {
        s
    }
}

/// Read access to samples: a whole plane, or one field of a frame plane
/// (every other row).
///
/// The plane may belong to a picture that another thread is still
/// decoding (frame threading): only rows it has finished may be read, and
/// so the view hands out slices of the rows asked for rather than of the
/// whole plane. Callers wait for those rows first (PicState::wait_rows).
#[derive(Clone, Copy)]
pub struct PlaneView<'a> {
    /// The first sample of the first row (8-bit or high bit depth plane;
    /// the other is dangling).
    ptr: *const u8,
    ptr16: *const u16,
    /// Samples from `ptr` (or `ptr16`) to the end of the buffer.
    len: usize,
    pub stride: usize,
    pub width: usize,
    pub height: usize,
    _plane: std::marker::PhantomData<&'a Plane>,
}

impl<'a> PlaneView<'a> {
    /// Samples `start..start + n` of the view.
    #[inline(always)]
    pub fn samples<T: Pixel>(&self, start: usize, n: usize) -> &'a [T] {
        assert!(start + n <= self.len);
        // SAFETY: in bounds of the buffer, which outlives 'a; the samples
        // are not being written (the caller waited for their rows).
        unsafe { std::slice::from_raw_parts(T::view_ptr(self).add(start), n) }
    }

    /// Sample `i` of the view.
    #[inline(always)]
    pub fn at<T: Pixel>(&self, i: usize) -> T {
        self.samples::<T>(i, 1)[0]
    }
}

/// A plane written through a raw pointer: by the thread decoding a picture
/// that other threads read (frame threading), or by several threads
/// writing disjoint parts of it (slices, deblocking rows).
pub(crate) struct RawPlane<T: Pixel = u8> {
    pub ptr: *mut T,
    pub len: usize,
    pub stride: usize,
}

impl<T: Pixel> Clone for RawPlane<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: Pixel> Copy for RawPlane<T> {}

// SAFETY: users keep to disjoint samples, ordered by their own
// synchronisation (see the users).
unsafe impl<T: Pixel> Send for RawPlane<T> {}
unsafe impl<T: Pixel> Sync for RawPlane<T> {}

impl<T: Pixel> RawPlane<T> {
    pub fn new(p: &mut Plane) -> Self {
        let stride = p.stride;
        let d = T::vec_mut(p);
        Self {
            ptr: d.as_mut_ptr(),
            len: d.len(),
            stride,
        }
    }

    /// The same plane as 8-bit samples (T is u8).
    #[inline(always)]
    pub fn bytes(&self) -> RawPlane<u8> {
        assert!(!T::HIGH);
        RawPlane {
            ptr: self.ptr.cast(),
            len: self.len,
            stride: self.stride,
        }
    }

    #[inline(always)]
    pub fn load<const N: usize>(&self, at: usize) -> [T; N] {
        assert!(at + N <= self.len);
        let mut v = [T::default(); N];
        // SAFETY: in bounds; see the type documentation.
        unsafe { std::ptr::copy_nonoverlapping(self.ptr.add(at), v.as_mut_ptr(), N) };
        v
    }

    #[inline(always)]
    pub fn store<const N: usize>(&self, at: usize, v: &[T; N]) {
        assert!(at + N <= self.len);
        // SAFETY: in bounds; see the type documentation.
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), self.ptr.add(at), N) };
    }

    /// `N` rows of `W` samples from `at`, `stride` apart: one bounds check
    /// for all of them.
    #[inline(always)]
    pub fn load_rows<const W: usize, const N: usize>(
        &self,
        at: usize,
        stride: usize,
    ) -> [[T; W]; N] {
        assert!(N > 0 && at + (N - 1) * stride + W <= self.len);
        // SAFETY: every row is in bounds (the last one is, asserted above);
        // see the type documentation.
        std::array::from_fn(|i| unsafe {
            std::ptr::read_unaligned(self.ptr.add(at + i * stride).cast::<[T; W]>())
        })
    }

    /// Stores rows as [`Self::load_rows`] loads them.
    #[inline(always)]
    pub fn store_rows<const W: usize, const N: usize>(
        &self,
        at: usize,
        stride: usize,
        rows: &[[T; W]; N],
    ) {
        assert!(N > 0 && at + (N - 1) * stride + W <= self.len);
        for (i, r) in rows.iter().enumerate() {
            // SAFETY: as load_rows.
            unsafe {
                std::ptr::write_unaligned(self.ptr.add(at + i * stride).cast::<[T; W]>(), *r)
            };
        }
    }

    /// Copies the samples at `at` to `dst`.
    #[inline(always)]
    pub fn read(&self, at: usize, dst: &mut [T]) {
        assert!(at + dst.len() <= self.len);
        // SAFETY: as load.
        unsafe { std::ptr::copy_nonoverlapping(self.ptr.add(at), dst.as_mut_ptr(), dst.len()) };
    }

    /// Copies `src` to `at`.
    #[inline(always)]
    pub fn write(&self, at: usize, src: &[T]) {
        assert!(at + src.len() <= self.len);
        // SAFETY: as store.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr.add(at), src.len()) };
    }

    /// The whole plane as a slice, for a user with exclusive access.
    ///
    /// # Safety
    /// No other thread may access the plane while the slice lives.
    pub unsafe fn exclusive<'a>(&self) -> &'a mut [T] {
        // SAFETY: the caller has exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

/// One colour plane: 8-bit samples in `data`, or samples of a higher bit
/// depth in `data16` (the other is empty).
#[derive(Clone)]
pub struct Plane {
    pub data: Vec<u8>,
    pub data16: Vec<u16>,
    pub stride: usize,
    pub width: usize,
    pub height: usize,
}

impl Plane {
    pub fn new(width: usize, height: usize) -> Self {
        Self::with_depth(width, height, false)
    }

    /// A plane of 16-bit samples when `high`.
    pub fn with_depth(width: usize, height: usize, high: bool) -> Self {
        // Rows padded to a multiple of 64 samples so SIMD kernels can over-read.
        let stride = width.next_multiple_of(64);
        let (data, data16) = if high {
            (Vec::new(), vec![0; stride * height])
        } else {
            (vec![0; stride * height], Vec::new())
        };
        Self {
            data,
            data16,
            stride,
            width,
            height,
        }
    }

    pub fn high(&self) -> bool {
        self.data.is_empty() && !self.data16.is_empty()
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.stride + x]
    }

    pub fn row(&self, y: usize) -> &[u8] {
        &self.data[y * self.stride..y * self.stride + self.width]
    }

    pub fn view(&self) -> PlaneView<'_> {
        self.view_from(0, self.stride, self.height)
    }

    /// The rows of one field of a frame plane.
    pub fn field(&self, parity: Parity) -> PlaneView<'_> {
        self.view_from(
            parity as usize * self.stride,
            2 * self.stride,
            self.height / 2,
        )
    }

    fn view_from(&self, o: usize, stride: usize, height: usize) -> PlaneView<'_> {
        // as_ptr does not create a reference to the samples, which another
        // thread may be writing (see PlaneView).
        let len = self.data.len().max(self.data16.len()).saturating_sub(o);
        PlaneView {
            ptr: self.data.as_ptr().wrapping_add(o),
            ptr16: self.data16.as_ptr().wrapping_add(o),
            len,
            stride,
            width: self.width,
            height,
            _plane: std::marker::PhantomData,
        }
    }
}

/// A decoded frame: luma plus two chroma planes (absent for 4:0:0).
#[derive(Clone)]
pub struct Picture {
    pub width_mbs: usize,
    pub height_mbs: usize,
    pub planes: Vec<Plane>,
    /// (MbWidthC, MbHeightC): chroma macroblock size, (0, 0) for 4:0:0.
    pub mb_c: (usize, usize),
    /// BitDepthY and BitDepthC. Samples are u16 when either exceeds 8.
    pub bit_depth: [u32; 2],
}

impl Picture {
    pub fn new(width_mbs: usize, height_mbs: usize, chroma_array_type: u8) -> Self {
        Self::with_depth(width_mbs, height_mbs, chroma_array_type, [8, 8])
    }

    /// A picture of the given luma and chroma bit depths.
    pub fn with_depth(
        width_mbs: usize,
        height_mbs: usize,
        chroma_array_type: u8,
        bit_depth: [u32; 2],
    ) -> Self {
        let high = bit_depth[0] > 8 || bit_depth[1] > 8;
        let (w, h) = (width_mbs * 16, height_mbs * 16);
        let mb_c = match chroma_array_type {
            1 => (8, 8),
            2 => (8, 16),
            3 => (16, 16),
            _ => (0, 0),
        };
        let mut planes = vec![Plane::with_depth(w, h, high)];
        if mb_c.0 > 0 {
            for _ in 0..2 {
                planes.push(Plane::with_depth(
                    width_mbs * mb_c.0,
                    height_mbs * mb_c.1,
                    high,
                ));
            }
        }
        Self {
            width_mbs,
            height_mbs,
            planes,
            mb_c,
            bit_depth,
        }
    }

    pub fn high(&self) -> bool {
        self.planes[0].high()
    }

    /// Mid-grey of plane `i`: 1 << (BitDepth - 1).
    pub fn grey(&self, i: usize) -> i32 {
        1 << (self.bit_depth[(i > 0) as usize] - 1)
    }

    /// The frame made of two field pictures; a missing field is grey (as
    /// JM outputs an unpaired field).
    pub fn interleave(top: Option<&Picture>, bottom: Option<&Picture>) -> Picture {
        let any = top.or(bottom).expect("a field");
        let high = any.high();
        let mut f = Picture {
            width_mbs: any.width_mbs,
            height_mbs: any.height_mbs * 2,
            planes: any
                .planes
                .iter()
                .map(|p| Plane::with_depth(p.width, p.height * 2, high))
                .collect(),
            mb_c: any.mb_c,
            bit_depth: any.bit_depth,
        };
        if high {
            f.interleave_t::<u16>(top, bottom);
        } else {
            f.interleave_t::<u8>(top, bottom);
        }
        f
    }

    fn interleave_t<T: Pixel>(&mut self, top: Option<&Picture>, bottom: Option<&Picture>) {
        let greys: Vec<i32> = (0..self.planes.len()).map(|i| self.grey(i)).collect();
        for (parity, field) in [(0, top), (1, bottom)] {
            for (i, dst) in self.planes.iter_mut().enumerate() {
                let grey = greys[i];
                let (w, stride) = (dst.width, dst.stride);
                for y in 0..dst.height / 2 {
                    let o = (2 * y + parity) * stride;
                    let row = &mut T::of_mut(dst)[o..o + w];
                    match field {
                        Some(src) => {
                            let s = &src.planes[i];
                            row.copy_from_slice(&T::of(s)[y * s.stride..y * s.stride + w]);
                        }
                        None => row.fill(T::new(grey)),
                    }
                }
            }
        }
    }
}
