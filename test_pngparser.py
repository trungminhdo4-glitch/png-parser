"""Tests für pngparser — PNG-Dateiformat von Grund auf lesbar machen.

Die Tests erzeugen echte PNGs selbst (Mini-Encoder nach PNG-Spec 1.2,
nur stdlib), damit alles offline und reproduzierbar bleibt.
"""

from __future__ import annotations

import contextlib
import io
import os
import struct
import tempfile
import unittest
import zlib
from typing import ClassVar

import pngparser
from pngparser import (
    PngFormatError,
    decode_png,
    load,
)

SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def make_chunk(ctype: bytes, data: bytes) -> bytes:
    """Richtiges Chunk-Längenfeld + Typ + Daten + CRC32."""
    return (
        struct.pack(">I", len(data))
        + ctype
        + data
        + struct.pack(">I", _crc32(ctype + data))
    )


def encode_png(
    width: int,
    height: int,
    depth: int,
    ctype: int,
    *,
    rows: list[bytes] | None = None,
    filtered_rows: list[bytes] | None = None,
    palette: bytes | None = None,
    trns: bytes | None = None,
    texts: list[tuple[str, str | bytes]] | None = None,
    interlace: int = 0,
    extra_chunks: list[tuple[bytes, bytes]] | None = None,
) -> bytes:
    """Echte PNG-Datei bauen. rows = unfilterte Scanlines (ohne Filterbyte)."""
    ihdr = struct.pack(">IIBBBBB", width, height, depth, ctype, 0, 0, interlace)
    parts = [SIGNATURE, make_chunk(b"IHDR", ihdr)]
    for t, d in extra_chunks or []:
        parts.append(make_chunk(t, d))
    if palette is not None:
        parts.append(make_chunk(b"PLTE", palette))
    if trns is not None:
        parts.append(make_chunk(b"tRNS", trns))
    for key, value in texts or []:
        v = value.encode("latin-1") if isinstance(value, str) else value
        parts.append(make_chunk(b"tEXt", key.encode("latin-1") + b"\x00" + v))
    if filtered_rows is None:
        filtered_rows = [b"\x00" + row for row in (rows or [])]
    parts.append(make_chunk(b"IDAT", zlib.compress(b"".join(filtered_rows))))
    parts.append(make_chunk(b"IEND", b""))
    return b"".join(parts)


def pack_samples(samples: list[int], depth: int) -> bytes:
    """Test-eigene Sampleverpackung (MSB zuerst), Zeilenbytes rechts 0-gepolstert."""
    if depth == 8:
        return bytes(samples)
    if depth == 16:
        return b"".join(struct.pack(">H", s) for s in samples)
    out = bytearray()
    acc = 0
    filled = 0
    for s in samples:
        acc = (acc << depth) | s
        filled += depth
        while filled >= 8:
            filled -= 8
            out.append((acc >> filled) & 0xFF)
            acc &= (1 << filled) - 1
    if filled:
        out.append((acc << (8 - filled)) & 0xFF)
    return bytes(out)


def encode_png_interlaced(
    width: int,
    height: int,
    depth: int,
    ctype: int,
    sample_rows: list[list[int]],
    *,
    palette: bytes | None = None,
    trns: bytes | None = None,
    interlace_filter: int = 0,
) -> bytes:
    """Adam7-interlaced PNG: sample_rows = Hoehe x (Breite*Kanaele) Samples."""
    ch = pngparser.CHANNELS_PER_COLOR_TYPE[ctype]
    raw = bytearray()
    for y0, x0, ystep, xstep in pngparser.ADAM7_PASSES:
        pw = (width - x0 + xstep - 1) // xstep
        ph = (height - y0 + ystep - 1) // ystep
        if pw <= 0 or ph <= 0:
            continue
        stride = (pw * depth * ch + 7) // 8
        prev_raw = bytearray(stride)
        for j in range(ph):
            full_y = y0 + j * ystep
            pass_samples = [
                sample_rows[full_y][(x0 + i * xstep) * ch + c]
                for i in range(pw)
                for c in range(ch)
            ]
            raw_row = pack_samples(pass_samples, depth)
            if interlace_filter and j > 0:
                # Up: gefiltert = roh minus rohe Pass-Vorgaengerzeile
                filtered = bytes((r - p) & 0xFF for r, p in zip(raw_row, prev_raw))
            else:
                filtered = raw_row
            raw.append(0 if interlace_filter == 0 or j == 0 else interlace_filter)
            raw.extend(filtered)
            prev_raw = bytearray(raw_row)  # Prev-Kontext ist die ROHE Zeile
    ihdr = struct.pack(">IIBBBBB", width, height, depth, ctype, 0, 0, 1)
    parts = [SIGNATURE, make_chunk(b"IHDR", ihdr)]
    if palette is not None:
        parts.append(make_chunk(b"PLTE", palette))
    if trns is not None:
        parts.append(make_chunk(b"tRNS", trns))
    parts.append(make_chunk(b"IDAT", zlib.compress(bytes(raw))))
    parts.append(make_chunk(b"IEND", b""))
    return b"".join(parts)


class SignatureAndChunkTests(unittest.TestCase):
    """Signatur, Chunk-Rahmen, CRC — die rohen Formatregeln."""

    def test_rejects_bad_signature(self) -> None:
        data = b"\x89PNG\r\n\x20\x0a" + make_chunk(b"IHDR", b"\x00" * 13)
        with self.assertRaisesRegex(PngFormatError, "Signatur"):
            decode_png(data)

    def test_truncated_chunk_raises(self) -> None:
        data = encode_png(1, 1, 8, 2, rows=[b"\xff\x00\x00"])
        with self.assertRaisesRegex(PngFormatError, "Chunk"):
            decode_png(data[:-20])
        with self.assertRaisesRegex(PngFormatError, "Chunk"):
            decode_png(data[:-1])

    def test_crc_mismatch_raises(self) -> None:
        data = bytearray(encode_png(1, 1, 8, 2, rows=[b"\xff\x00\x00"]))
        crc_offset = len(data) - 4  # erstes CRC-Byte von IEND
        data[crc_offset] ^= 0x01
        with self.assertRaisesRegex(PngFormatError, "CRC"):
            decode_png(bytes(data))

    def test_ihdr_must_be_first(self) -> None:
        tail = make_chunk(b"tEXt", b"Key\x00Wert")
        parts = [
            SIGNATURE,
            tail,
            make_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)),
        ]
        with self.assertRaisesRegex(PngFormatError, "IHDR"):
            decode_png(b"".join(parts) + make_chunk(b"IEND", b""))

    def test_iend_must_be_last(self) -> None:
        parts = encode_png(1, 1, 8, 2, rows=[b"\xff\x00\x00"])
        with self.assertRaisesRegex(PngFormatError, "IEND"):
            decode_png(parts + make_chunk(b"tEXt", b"K\x00V"))

    def test_trailing_bytes_after_iend_raise(self) -> None:
        data = encode_png(1, 1, 8, 2, rows=[b"\xff\x00\x00"]) + b"JUNK"
        with self.assertRaisesRegex(PngFormatError, "IEND"):
            decode_png(data)

    def test_flags_of_chunk_types(self) -> None:
        img = decode_png(
            encode_png(1, 1, 8, 2, rows=[b"\xff\x00\x00"], texts=[("T", "V")])
        )
        by_name = {c.type_name: c for c in img.chunks}
        self.assertEqual(
            by_name["IHDR"].flags(), {"critisch", "oeffentlich", "nicht safe-to-copy"}
        )
        self.assertEqual(
            by_name["IDAT"].flags(), {"critisch", "oeffentlich", "nicht safe-to-copy"}
        )
        self.assertEqual(
            by_name["tEXt"].flags(), {"nebenbedeutend", "oeffentlich", "safe-to-copy"}
        )


class HeaderTests(unittest.TestCase):
    """IHDR: Masse, Bittiefe, Farbart — validiert gegen die Spec."""

    def test_ihdr_values_roundtrip(self) -> None:
        img = decode_png(encode_png(3, 2, 8, 2, rows=[b"\x00" * 9, b"\x00" * 9]))
        self.assertEqual(img.width, 3)
        self.assertEqual(img.height, 2)
        self.assertEqual(img.bit_depth, 8)
        self.assertEqual(img.color_type, 2)
        self.assertEqual(img.color_type_name, "RGB")

    def test_invalid_depth_color_combination(self) -> None:
        # Spec-Tabelle: Farbart 2 (RGB) darf nur depth 8 oder 16 haben.
        data = encode_png(1, 1, 4, 2, rows=[b"\x00"])
        with self.assertRaisesRegex(PngFormatError, "Bittiefe"):
            decode_png(data)

    def test_width_zero_raises(self) -> None:
        ihdr = struct.pack(">IIBBBBB", 0, 1, 8, 2, 0, 0, 0)
        parts = [
            SIGNATURE,
            make_chunk(b"IHDR", ihdr),
            make_chunk(b"IDAT", zlib.compress(b"")),
            make_chunk(b"IEND", b""),
        ]
        with self.assertRaisesRegex(PngFormatError, "Breite"):
            decode_png(b"".join(parts))


class PixelRoundtripTests(unittest.TestCase):
    """Deko&Unfilter: alle Farbtarten/Bittiefen schliessen auf die Originalpixel."""

    def test_rgb8_gradient(self) -> None:
        rows = [bytes((255, 0, 0, 0, 255, 0, 0, 0, 255))]
        img = decode_png(encode_png(3, 1, 8, 2, rows=rows))
        self.assertEqual(
            list(img.iter_rgba()),
            [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)],
        )

    def test_gray16_scales_down(self) -> None:
        row = struct.pack(">H", 0x1234)
        img = decode_png(encode_png(1, 1, 16, 0, rows=[row]))
        self.assertEqual(list(img.iter_rgba()), [(0x12, 0x12, 0x12, 255)])

    def test_gray16_trns_match(self) -> None:
        row = struct.pack(">H", 0x1234)
        img = decode_png(
            encode_png(1, 1, 16, 0, rows=[row], trns=struct.pack(">H", 0x1234))
        )
        self.assertEqual(list(img.iter_rgba()), [(0x12, 0x12, 0x12, 0)])

    def test_gray8(self) -> None:
        img = decode_png(encode_png(2, 1, 8, 0, rows=[bytes((0x40, 0x80))]))
        self.assertEqual(
            list(img.iter_rgba()),
            [(0x40, 0x40, 0x40, 255), (0x80, 0x80, 0x80, 255)],
        )

    def test_gray_alpha8(self) -> None:
        img = decode_png(encode_png(1, 1, 8, 4, rows=[bytes((0x80, 0x40))]))
        self.assertEqual(list(img.iter_rgba()), [(0x80, 0x80, 0x80, 0x40)])

    def test_gray_1bit_msb_first(self) -> None:
        # 3 Pixel in 1 Byte: 0b10100000 = 0xA0, value 1 -> weiss
        img = decode_png(encode_png(3, 1, 1, 0, rows=[b"\xa0"]))
        self.assertEqual(
            list(img.iter_rgba()),
            [(255, 255, 255, 255), (0, 0, 0, 255), (255, 255, 255, 255)],
        )

    def test_gray_2bit_scales(self) -> None:
        # Werte 0..3 -> skaliert auf *255 // 3 (0b00011011 = 0x1B)
        img = decode_png(encode_png(4, 1, 2, 0, rows=[b"\x1b"]))
        self.assertEqual(
            list(img.iter_rgba()),
            [
                (0, 0, 0, 255),
                (85, 85, 85, 255),
                (170, 170, 170, 255),
                (255, 255, 255, 255),
            ],
        )

    def test_gray_4bit_scales(self) -> None:
        img = decode_png(encode_png(2, 1, 4, 0, rows=[b"\x1f"]))
        self.assertEqual(
            list(img.iter_rgba()),
            [(17, 17, 17, 255), (255, 255, 255, 255)],
        )

    def test_rgba8(self) -> None:
        img = decode_png(encode_png(1, 1, 8, 6, rows=[bytes((1, 2, 3, 4))]))
        self.assertEqual(list(img.iter_rgba()), [(1, 2, 3, 4)])

    def test_palette_4bit(self) -> None:
        palette = bytes((16, 32, 48)) + bytes((240, 224, 208))
        img = decode_png(encode_png(2, 1, 4, 3, rows=[b"\x01"], palette=palette))
        self.assertEqual(
            list(img.iter_rgba()), [(16, 32, 48, 255), (240, 224, 208, 255)]
        )

    def test_palette_with_trns(self) -> None:
        palette = bytes((16, 32, 48)) + bytes((240, 224, 208))
        img = decode_png(
            encode_png(
                2, 1, 4, 3, rows=[b"\x01"], palette=palette, trns=bytes((0, 128))
            )
        )
        self.assertEqual(list(img.iter_rgba()), [(16, 32, 48, 0), (240, 224, 208, 128)])

    def test_palette_1bit(self) -> None:
        palette = bytes((9, 9, 9)) + bytes((250, 250, 250))
        # 2 Pixel à 1 Bit, MSB zuerst: 0b01000000 = 0x40 -> Indizes 0, 1
        img = decode_png(encode_png(2, 1, 1, 3, rows=[b"\x40"], palette=palette))
        self.assertEqual(list(img.iter_rgba()), [(9, 9, 9, 255), (250, 250, 250, 255)])


class FilterTests(unittest.TestCase):
    """Filter Reconstruction — pro Filter ein handgerechneter 2x2-Fall."""

    ROW0: ClassVar[bytes] = bytes((10, 20, 30, 40, 60, 80))
    ROW1: ClassVar[bytes] = bytes((12, 24, 36, 44, 66, 88))
    EXPECTED: ClassVar[list[tuple[int, int, int]]] = [
        (10, 20, 30),
        (40, 60, 80),
        (12, 24, 36),
        (44, 66, 88),
    ]

    def _decode_with_filter(
        self, filter_byte: int, filtered_row1: bytes
    ) -> list[tuple[int, int, int]]:
        filtered_rows = [b"\x00" + self.ROW0, bytes((filter_byte,)) + filtered_row1]
        img = decode_png(encode_png(2, 2, 8, 2, filtered_rows=filtered_rows))
        return [(r, g, b) for r, g, b, _a in img.iter_rgba()]

    def test_filter_none(self) -> None:
        got = self._decode_with_filter(0, self.ROW1)
        self.assertEqual(got, self.EXPECTED)

    def test_filter_sub(self) -> None:
        # Sub, bpp=3: erstes Pixel bleibt roh, danach Differenz auf gleiche
        # Kanalposition des PIXELS links: 44-12, 66-24, 88-36
        got = self._decode_with_filter(1, bytes((12, 24, 36, 32, 42, 52)))
        self.assertEqual(got, self.EXPECTED)

    def test_filter_up(self) -> None:
        # Up: oben subtrahieren: 12-10, 24-20, … gleiche Differenzen weil ROW0 klein
        got = self._decode_with_filter(2, bytes((2, 4, 6, 4, 6, 8)))
        self.assertEqual(got, self.EXPECTED)

    def test_filter_average(self) -> None:
        # Average, bpp=3: raw - floor((links-Pixel + oben) / 2);
        # x0-2: ohne links -> pred = 5/10/15; x3-5: a = 12/24/36
        got = self._decode_with_filter(3, bytes((7, 14, 21, 18, 24, 30)))
        self.assertEqual(got, self.EXPECTED)

    def test_filter_paeth(self) -> None:
        # Paeth, bpp=3: x0-2 ohne a/c -> p = b und |p-b| = 0 -> b gewinnt,
        # daher wie Up: 12-10, 24-20, 36-30; x3-5: b gewinnt Paeth-Wahl
        got = self._decode_with_filter(4, bytes((2, 4, 6, 4, 6, 8)))
        self.assertEqual(got, self.EXPECTED)

    def test_unknown_filter_raises(self) -> None:
        filtered_rows = [b"\x00" + self.ROW0, bytes((5,)) + self.ROW1]
        with self.assertRaisesRegex(PngFormatError, "Filter"):
            decode_png(encode_png(2, 2, 8, 2, filtered_rows=filtered_rows))

    def test_bpp_values(self) -> None:
        self.assertEqual(pngparser.bpp(1, 1), 1)  # 1-Bit-Grau: aufgerundet auf 1
        self.assertEqual(pngparser.bpp(8, 4), 4)  # 8-Bit-RGBA
        self.assertEqual(pngparser.bpp(16, 3), 6)  # 16-Bit-RGB
        self.assertEqual(pngparser.bpp(2, 1), 1)  # sub-byte: mind. 1


class ChunkOrderingTests(unittest.TestCase):
    """IDAT-Konkatenation, PLTE-Anforderungen, Unbekannte tolerieren."""

    def test_split_idat_ok(self) -> None:
        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        idat = zlib.compress(b"\x00" + b"\xff\x00\x00")
        mid = len(idat) // 2
        parts = [
            SIGNATURE,
            make_chunk(b"IHDR", ihdr),
            make_chunk(b"IDAT", idat[:mid]),
            make_chunk(b"IDAT", idat[mid:]),
            make_chunk(b"IEND", b""),
        ]
        img = decode_png(b"".join(parts))
        self.assertEqual(img.color_type, 2)

    def test_split_idat_gap_raises(self) -> None:
        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        idat = zlib.compress(b"\x00" + b"\xff\x00\x00")
        mid = len(idat) // 2
        parts = [
            SIGNATURE,
            make_chunk(b"IHDR", ihdr),
            make_chunk(b"IDAT", idat[:mid]),
            make_chunk(b"tEXt", b"K\x00V"),
            make_chunk(b"IDAT", idat[mid:]),
            make_chunk(b"IEND", b""),
        ]
        with self.assertRaisesRegex(PngFormatError, "IDAT"):
            decode_png(b"".join(parts))

    def test_unknown_ancillary_chunk_tolerated(self) -> None:
        data = encode_png(
            1, 1, 8, 2, rows=[b"\xff\x00\x00"], extra_chunks=[(b"abCd", b"\x01\x02")]
        )
        img = decode_png(data)
        types = [c.type_name for c in img.chunks]
        self.assertIn("abCd", types)

    def test_missing_palette_raises(self) -> None:
        data = encode_png(1, 1, 1, 3, rows=[b"\x80"], palette=None)  # ctype 3 ohne PLTE
        with self.assertRaisesRegex(PngFormatError, "PLTE"):
            decode_png(data)

    def test_palette_not_multiple_of_3(self) -> None:
        data = encode_png(1, 1, 1, 3, rows=[b"\x80"], palette=b"\x01\x02")
        with self.assertRaisesRegex(PngFormatError, "PLTE"):
            decode_png(data)

    def test_duplicate_plte_raises(self) -> None:
        # Spec: "There must not be more than one PLTE chunk."
        pal = bytes((9, 9, 9)) + bytes((250, 250, 250))
        data = encode_png(
            1, 1, 1, 3, rows=[b"\x80"], palette=pal, extra_chunks=[(b"PLTE", pal)]
        )
        with self.assertRaisesRegex(PngFormatError, "PLTE"):
            decode_png(data)

    def test_duplicate_trns_raises(self) -> None:
        data = encode_png(
            1,
            1,
            8,
            0,
            rows=[b"\x40"],
            trns=b"\x00\x40",
            extra_chunks=[(b"tRNS", b"\x00\x80")],
        )
        with self.assertRaisesRegex(PngFormatError, "tRNS"):
            decode_png(data)

    def test_palette_index_out_of_range_raises(self) -> None:
        # PLTE mit 1 Eintrag, Pixel nutzt Index 1 -> PngFormatError, kein IndexError
        data = encode_png(2, 1, 1, 3, rows=[b"\x40"], palette=bytes((9, 9, 9)))
        with self.assertRaisesRegex(PngFormatError, "Palette"):
            decode_png(data)

    def test_plte_on_grayscale_raises(self) -> None:
        data = encode_png(
            1, 1, 8, 0, rows=[b"\x40"], extra_chunks=[(b"PLTE", b"\x01\x02\x03")]
        )
        with self.assertRaisesRegex(PngFormatError, "PLTE"):
            decode_png(data)

    def test_corrupt_zlib_raises(self) -> None:
        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        parts = [
            SIGNATURE,
            make_chunk(b"IHDR", ihdr),
            make_chunk(b"IDAT", b"\x01\x02\x03\x04"),
            make_chunk(b"IEND", b""),
        ]
        with self.assertRaisesRegex(PngFormatError, "deflate"):
            decode_png(b"".join(parts))

    def test_idat_wrong_size_raises(self) -> None:
        # Erwartet 1 Zeile à (1 Filterbyte + 3 Bytes), geben nur 2 Bytes.
        parts = encode_png(1, 1, 8, 2, filtered_rows=[b"\x00\xff"])
        with self.assertRaisesRegex(PngFormatError, "Länge|Scanline"):
            decode_png(parts)

    def test_text_chunks_collected(self) -> None:
        img = decode_png(
            encode_png(
                1,
                1,
                8,
                2,
                rows=[b"\xff\x00\x00"],
                texts=[("Title", "Demo"), ("Author", "Ich")],
            )
        )
        self.assertEqual(img.text, {"Title": "Demo", "Author": "Ich"})

    def test_interlace_2_raises(self) -> None:
        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 2)
        parts = [
            SIGNATURE,
            make_chunk(b"IHDR", ihdr),
            make_chunk(b"IDAT", zlib.compress(b"\x00\x00\xff\x00")),
            make_chunk(b"IEND", b""),
        ]
        with self.assertRaisesRegex(PngFormatError, "Interlace"):
            decode_png(b"".join(parts))


class InterlaceTests(unittest.TestCase):
    """Adam7-Interlace: Pass-Tabelle, Pass-Entzerren, Pass-lokaler Filterkontext."""

    def test_adam7_table_matches_spec_grid(self) -> None:
        # RFC 2083, Abschnitt 2.6, Figur 8-1 (8x8-Raster der Pass-Nummern)
        spec_grid = (
            (1, 6, 4, 6, 2, 6, 4, 6),
            (7, 7, 7, 7, 7, 7, 7, 7),
            (5, 6, 5, 6, 5, 6, 5, 6),
            (7, 7, 7, 7, 7, 7, 7, 7),
            (3, 6, 4, 6, 3, 6, 4, 6),
            (7, 7, 7, 7, 7, 7, 7, 7),
            (5, 6, 5, 6, 5, 6, 5, 6),
            (7, 7, 7, 7, 7, 7, 7, 7),
        )
        covered: dict[tuple[int, int], int] = {}
        for pass_no, (y0, x0, ystep, xstep) in enumerate(
            pngparser.ADAM7_PASSES, start=1
        ):
            for y in range(y0, 8, ystep):
                for x in range(x0, 8, xstep):
                    cell = (y, x)
                    self.assertNotIn(
                        cell, covered, f"Zelle {cell} doppelt (Pass {pass_no})"
                    )
                    covered[cell] = pass_no
        self.assertEqual(len(covered), 64)
        expected = {(y, x): spec_grid[y][x] for y in range(8) for x in range(8)}
        self.assertEqual(covered, expected)

    def test_interlaced_gray8_roundtrip(self) -> None:
        # 9x9: alle 7 Paesse nichtleer, ungerade Kanten
        sample_rows = [[(y * 29 + x * 7) % 256 for x in range(9)] for y in range(9)]
        img = decode_png(encode_png_interlaced(9, 9, 8, 0, sample_rows))
        self.assertEqual(
            img.to_rgba_rows(),
            [[(v, v, v, 255) for v in row] for row in sample_rows],
        )

    def test_interlaced_rgb8_roundtrip(self) -> None:
        # flache Sampleliste: pro Pixel 3 Werte (r, g, b)
        sample_rows = [
            [v for x in range(9) for v in (x * 28, y * 28, 128)] for y in range(9)
        ]
        img = decode_png(encode_png_interlaced(9, 9, 8, 2, sample_rows))
        self.assertEqual(
            img.to_rgba_rows(),
            [
                [tuple(row[i : i + 3]) + (255,) for i in range(0, len(row), 3)]
                for row in sample_rows
            ],
        )

    def test_interlaced_rgba8_roundtrip(self) -> None:
        sample_rows = [
            [v for x in range(5) for v in (x * 40, y * 40, 10, x * y % 256)]
            for y in range(5)
        ]
        img = decode_png(encode_png_interlaced(5, 5, 8, 6, sample_rows))
        self.assertEqual(
            img.to_rgba_rows(),
            [
                [tuple(row[i : i + 4]) for i in range(0, len(row), 4)]
                for row in sample_rows
            ],
        )

    def test_interlaced_gray16_roundtrip(self) -> None:
        sample_rows = [
            [(y * 1000 + x * 333) % 65536 for x in range(3)] for y in range(3)
        ]
        img = decode_png(encode_png_interlaced(3, 3, 16, 0, sample_rows))
        self.assertEqual(
            img.to_rgba_rows(),
            [[((v >> 8), (v >> 8), (v >> 8), 255) for v in row] for row in sample_rows],
        )

    def test_interlaced_palette8_roundtrip(self) -> None:
        palette = (
            bytes((16, 32, 48))
            + bytes((240, 224, 208))
            + bytes((9, 9, 9))
            + bytes((7, 7, 7))
        )
        trns = bytes((0, 128, 200, 255))
        sample_rows = [[(x + y) % 4 for x in range(8)] for y in range(8)]
        img = decode_png(
            encode_png_interlaced(8, 8, 8, 3, sample_rows, palette=palette, trns=trns)
        )
        colors = [(16, 32, 48), (240, 224, 208), (9, 9, 9), (7, 7, 7)]
        expected = [
            [(colors[i][0], colors[i][1], colors[i][2], trns[i]) for i in row]
            for row in sample_rows
        ]
        self.assertEqual(img.to_rgba_rows(), expected)

    def test_interlaced_gray1bit_pass_row_padding(self) -> None:
        # 9 Pixel à 1 Bit: jede Pass-Zeile wird separat auf Ganzes Byte gepolstert
        sample_rows = [[(x + y) % 2 for x in range(9)] for y in range(9)]
        img = decode_png(encode_png_interlaced(9, 9, 1, 0, sample_rows))
        self.assertEqual(
            img.to_rgba_rows(),
            [
                [((v * 255), (v * 255), (v * 255), 255) for v in row]
                for row in sample_rows
            ],
        )

    def test_interlaced_pass_local_filter_context(self) -> None:
        # Filter 2 (Up) innerhalb der Paesse: Prev = Pass-Vorgaengerzeile,
        # nicht die volle Bildzeile darueber.
        sample_rows = [
            [v for x in range(9) for v in (x * 31, y * 17, 200)] for y in range(9)
        ]
        data = encode_png_interlaced(9, 9, 8, 2, sample_rows, interlace_filter=2)
        img = decode_png(data)
        self.assertEqual(
            img.to_rgba_rows(),
            [
                [tuple(row[i : i + 3]) + (255,) for i in range(0, len(row), 3)]
                for row in sample_rows
            ],
        )

    def test_interlaced_1x1_only_pass1(self) -> None:
        # 1x1: nur Pass 1 liefert eine Scanline, Pass 2-7 leer (keine Filterbytes)
        sample_rows = [[7, 200, 90]]
        img = decode_png(encode_png_interlaced(1, 1, 8, 2, sample_rows))
        self.assertEqual(img.to_rgba_rows(), [[(7, 200, 90, 255)]])

    def test_interlaced_idat_extra_bytes_raises(self) -> None:
        sample_rows = [
            [v for x in range(3) for v in (x * 3, y * 5, 90)] for y in range(3)
        ]
        base = encode_png_interlaced(3, 3, 8, 2, sample_rows)
        idat = next(c for c in pngparser.read_chunks(base) if c.type_name == "IDAT")
        raw = zlib.decompress(idat.data) + b"\x00"
        ihdr = struct.pack(">IIBBBBB", 3, 3, 8, 2, 0, 0, 1)
        parts = [
            SIGNATURE,
            make_chunk(b"IHDR", ihdr),
            make_chunk(b"IDAT", zlib.compress(raw)),
            make_chunk(b"IEND", b""),
        ]
        with self.assertRaisesRegex(PngFormatError, "Rohdaten"):
            decode_png(b"".join(parts))

    def test_interlaced_idat_truncated_pass_raises(self) -> None:
        sample_rows = [
            [v for x in range(3) for v in (x * 3, y * 5, 90)] for y in range(3)
        ]
        base = encode_png_interlaced(3, 3, 8, 2, sample_rows)
        idat = next(c for c in pngparser.read_chunks(base) if c.type_name == "IDAT")
        raw = zlib.decompress(idat.data)[:-2]
        ihdr = struct.pack(">IIBBBBB", 3, 3, 8, 2, 0, 0, 1)
        parts = [
            SIGNATURE,
            make_chunk(b"IHDR", ihdr),
            make_chunk(b"IDAT", zlib.compress(raw)),
            make_chunk(b"IEND", b""),
        ]
        with self.assertRaisesRegex(PngFormatError, "abgeschnitten"):
            decode_png(b"".join(parts))


class IoAndExportTests(unittest.TestCase):
    """Datei-I/O und PPM-Export (damit man das Ergebnis anschauen kann)."""

    def test_load_from_file(self) -> None:
        data = encode_png(1, 1, 8, 2, rows=[b"\xff\x00\x00"])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "demo.png")
            with open(path, "wb") as fh:
                fh.write(data)
            img = load(path)
            self.assertEqual(img.width, 1)

    def test_save_ppm(self) -> None:
        img = decode_png(encode_png(2, 1, 8, 2, rows=[bytes((255, 0, 0, 0, 255, 0))]))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.ppm")
            pngparser.write_ppm(img, path)
            with open(path, "rb") as fh:
                blob = fh.read()
        self.assertTrue(blob.startswith(b"P6\n2 1\n255\n"))
        pixel_bytes = blob.split(b"255\n", 1)[1]
        self.assertEqual(pixel_bytes, bytes((255, 0, 0, 0, 255, 0)))

    def test_save_ppm_rgba_keeps_color_drops_alpha(self) -> None:
        img = decode_png(encode_png(1, 1, 8, 6, rows=[bytes((1, 2, 3, 250))]))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.ppm")
            pngparser.write_ppm(img, path)
            with open(path, "rb") as fh:
                blob = fh.read()
        self.assertTrue(blob.endswith(bytes((1, 2, 3))))

    def test_main_missing_file_reports_clean_error(self) -> None:
        # Nur PngError zu fangen wäre zu wenig: auch OSError (z. B. fehlende Datei)
        # muss zu einer klaren Meldung statt Traceback führen.
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "gibts-nicht.png")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = pngparser.main([missing])
        self.assertEqual(code, 1)
        self.assertIn("FEHLER", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
