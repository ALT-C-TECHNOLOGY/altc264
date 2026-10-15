#!/bin/sh
# Builds the player core: FFmpeg 9.0.2 with libaltc264, built with wasi-sdk,
# threaded (work/web/core.wasm, wasm32-wasip1-threads) and single-threaded
# for browsers without threads (work/web/core-st.wasm, wasm32-wasip1).
# Downloads wasi-sdk and FFmpeg on first run.
set -eu
cd "$(dirname "$0")/../.."
ROOT=$PWD
OUT=$ROOT/work/web
SDK=$OUT/wasi-sdk
FF=$OUT/ffmpeg
mkdir -p "$OUT"
case $(uname -sm) in
"Darwin arm64") HOST=arm64-macos ;;
"Darwin x86_64") HOST=x86_64-macos ;;
"Linux x86_64") HOST=x86_64-linux ;;
"Linux aarch64") HOST=arm64-linux ;;
esac
[ -d "$SDK" ] || {
	curl -sSfL "https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-34/wasi-sdk-34.0-$HOST.tar.gz" | tar xz -C "$OUT"
	mv "$OUT/wasi-sdk-34.0-$HOST" "$SDK"
}
[ -d "$FF/src" ] || git clone -q --depth 1 -b n9.0.2 https://git.ffmpeg.org/ffmpeg.git "$FF/src"
integrations/ffmpeg/apply.sh "$FF/src" > /dev/null
EMU="-D_WASI_EMULATED_PROCESS_CLOCKS -D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_MMAN"
EMULIBS="-lwasi-emulated-process-clocks -lwasi-emulated-signal -lwasi-emulated-mman"

# build TARGET THREADS(yes|no) OUTPUT
build() {
	target=$1 dir=$FF/$1 prefix=$OUT/prefix-$1
	if [ "$2" = yes ]; then
		cflags="--target=$target -pthread -msimd128 -O3 $EMU" threads=--enable-pthreads
		# The allocator wrapped as in .cargo/config.toml, whose flags the
		# library build below takes.
		wrap="-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free,--wrap=aligned_alloc,--wrap=posix_memalign,--wrap=sbrk"
		link="-Wl,--import-memory,--export-memory,--max-memory=4294967296 -Wl,--export=wasi_thread_start $wrap"
		rustflags=unset
	else
		cflags="--target=$target -msimd128 -O3 $EMU" threads=--disable-pthreads
		link="-Wl,--max-memory=4294967296"
		rustflags="-C target-feature=+simd128"
	fi
	# libaltc264, for FFmpeg's pkg-config check.
	if [ "$rustflags" = unset ]; then
		CARGO_TARGET_DIR=$OUT/cargo cargo build -q --release -p altc264-capi --target "$target"
	else
		CARGO_TARGET_DIR=$OUT/cargo RUSTFLAGS=$rustflags cargo build -q --release -p altc264-capi --target "$target"
	fi
	mkdir -p "$prefix/lib/pkgconfig" "$prefix/include" "$dir"
	cp "$OUT/cargo/$target/release/libaltc264.a" "$prefix/lib/"
	cp crates/altc264-capi/include/altc264.h "$prefix/include/"
	printf 'Name: altc264\nDescription: H.264/AVC and MVC video decoder\nVersion: 0.1.0\nLibs: -L%s -laltc264\nCflags: -I%s\n' \
		"$prefix/lib" "$prefix/include" > "$prefix/lib/pkgconfig/altc264.pc"
	# FFmpeg: demuxers, libaltc264 and the common audio decoders.
	cd "$dir"
	[ -f ffbuild/config.mak ] || PKG_CONFIG_PATH=$prefix/lib/pkgconfig ../src/configure \
		--enable-cross-compile --arch=wasm32 --target-os=none --cc="$SDK/bin/clang" --ld="$SDK/bin/clang" \
		--ar="$SDK/bin/llvm-ar" --nm="$SDK/bin/llvm-nm" --ranlib="$SDK/bin/llvm-ranlib" --strip="$SDK/bin/llvm-strip" \
		--pkg-config=pkg-config --pkg-config-flags=--static \
		--disable-everything --disable-programs --disable-doc --disable-network --disable-autodetect \
		--disable-asm --disable-inline-asm --disable-runtime-cpudetect --disable-debug \
		$threads --disable-w32threads --disable-os2threads --enable-swresample --enable-libaltc264 \
		--enable-decoder=libaltc264,aac,ac3,eac3,dca,truehd,mp3,flac,opus \
		--enable-demuxer=matroska,mov,mpegts --enable-parser=h264,aac,ac3,dca,mlp,mpegaudio,flac,opus \
		--extra-cflags="$cflags" --extra-ldflags="${cflags%% -msimd128*} $EMULIBS" > configure.log
	make -s -j"$(getconf _NPROCESSORS_ONLN)"
	cd "$ROOT"
	# The player core: a reactor module with its exports.
	"$SDK/bin/clang" $cflags -mexec-model=reactor -I"$dir" -I"$FF/src" tools/web/core.c -o "$OUT/$3" \
		"$dir/libavformat/libavformat.a" "$dir/libavcodec/libavcodec.a" \
		"$dir/libswresample/libswresample.a" "$dir/libavutil/libavutil.a" "$prefix/lib/libaltc264.a" \
		-lm $EMULIBS $link
	echo "built $OUT/$3"
}

build wasm32-wasip1-threads yes core.wasm
build wasm32-wasip1 no core-st.wasm
