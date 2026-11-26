# png-parser

Inspect and decode PNG files using only the Python standard library. The parser
exposes chunks, CRC status, image metadata and decoded pixels, so developers can
understand a file or investigate corruption without a separate image library.
It is an educational decoder and inspection tool, not a general image editor.

## Quickstart

Python 3.10 or newer is required. No package installation or network access is
needed to run the parser or its tests:

```bash
python pngparser.py --help
python -m unittest discover -s . -p "test_*.py" -v
python pngparser.py image.png --pixels 8
python pngparser.py image.png --pam image.pam
```

Replace `image.png` with a local PNG. PAM preserves the decoded alpha channel;
PPM exports RGB and drops alpha. Both exports write pixel data in blocks.

## Supported features

- Grayscale, RGB, indexed color, grayscale with alpha, and RGBA (types 0/2/3/4/6).
- Valid 1/2/4/8/16-bit combinations, all five scanline filters, and Adam7 interlace.
- Consecutive IDAT chunks, CRC32 checking, palette transparency (`tRNS`), and
  uncompressed Latin-1 text (`tEXt`).
- PPM (P6) and PAM (P7 RGBA) export; output samples are 8-bit per channel.
- Independent configurable limits for input bytes and pixel count.

## CLI

```bash
python pngparser.py image.png                  # Metadata, chunks and CRC status
python pngparser.py image.png --ppm image.ppm   # RGB export
python pngparser.py image.png --pam image.pam   # RGBA export
python pngparser.py image.png --pixels 8        # First eight RGBA pixels
python pngparser.py image.png --max-pixels 25000000
python pngparser.py image.png --max-input-bytes 134217728
python pngparser.py image.png --no-crc          # Diagnostic use only
```

A decode or file error returns exit code 1 and writes a diagnostic to stderr.
Invalid CLI arguments return 2. Resource-limit errors occur before any export
file is opened. Existing export files are otherwise overwritten; export writes
are not atomic.

## Python API

```python
from pngparser import load

image = load("image.png", max_pixels=16_000_000, max_input_bytes=67_108_864)
print(image.width, image.height, image.bit_depth, image.color_type_name)
print(next(image.iter_rgba()))
image.save_pam("image.pam")
```

`decode_png(data, ...)` accepts the same keyword limits for bytes already held by
the caller. `PngFormatError` indicates invalid PNG structure or data;
`PngUnsupportedError` includes files exceeding configured limits. Invalid
`max_input_bytes` values raise `ValueError`; it must be a positive integer.

## Resource and validation model

The defaults are **64 MiB of PNG input** and **16,000,000 pixels**. Small images
can contain large metadata chunks, so a pixel limit alone does not bound file
input. `load()` reads at most the byte limit plus one detection byte, even if a
file grows after opening. `decode_png()` rejects excess bytes before parsing or
copying chunks. Raise `max_input_bytes` or `--max-input-bytes` deliberately when
inspecting larger files. This is an application policy, not a PNG validity rule.

These limits do **not** cap total process memory or CPU time. The decoder retains
chunks and reconstructed image data; Python sample lists, especially Adam7 and
16-bit images, can consume substantially more memory than their input. The
low-level `read_chunks()` helper has no input limit. Use OS-level resource limits
or process isolation when inspecting untrusted files in a service.

CRC checks are enabled by default. The parser also checks chunk order, duplicate
palette/transparency chunks, unknown critical chunks, palette indices, scanline
lengths, and complete zlib streams. Disabling CRC does not disable resource
limits. Color management, compressed/international text, animation, and PNG
encoding are not implemented. Samples are inspected without color correction;
16-bit samples are reduced to their high byte for RGBA iteration/export.

## Architecture and tests

`pngparser.py` follows the file structure: signature → chunks/CRC → IHDR →
IDAT inflation → filter reconstruction → pixel iteration/export. Adam7 decodes
seven passes with a separate filter context per pass. See the
[W3C PNG specification](https://www.w3.org/TR/png-3/) for chunk and image layout.

`test_pngparser.py` contains a synthetic PNG encoder and offline tests for color
and bit-depth combinations, filters, Adam7, malformed files, resource limits,
and exports. Run the command in Quickstart. The [test workflow](.github/workflows/tests.yml)
runs the suite on Linux and Windows with Python 3.10, 3.12 and 3.14. Tests use
local generated fixtures; they are not an exhaustive PNG conformance suite.

## License

MIT; see [LICENSE](LICENSE).
