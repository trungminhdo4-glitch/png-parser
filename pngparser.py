"""PNG-Parser von Grund auf — liest PNG-Bilder ohne eine einzige Bildbibliothek.

Alles nach PNG-Spec 1.2, nur Python-Stdlib, komplett offline. Gebaut als
Reverse-Engineering-Uebung: das Dateiformat Byte fuer Byte erklaeren.

Nur Decoder (lesen). Unterstuetzt:
- Non-interlaced UND Adam7-Interlace (7 Pässe, pass-lokaler Filterkontext)
- Alle 5 Farbarten: 0=Graustufen, 2=RGB, 3=Palette, 4=Graustufen+Alpha, 6=RGBA
- Alle Bittiefen: 1, 2, 4, 8, 16
- Alle 5 Filter: None, Sub, Up, Average, Paeth
- Mehrere IDAT-Chunks (muessen konsekutiv sein), CRC32-Pruefung, tRNS, tEXt

Beispiel:
    img = decode_png(open("bild.png", "rb").read())
    print(img.width, img.height, img.color_type_name)
    for r, g, b, a in img.iter_rgba():
        ...
    img.save_ppm("bild.ppm")   # ohne Lib anschau- und editierbar machen

CLI:
    python pngparser.py bild.png [--ppm out.ppm] [--pixels] [--no-crc]
"""

from __future__ import annotations

import argparse
import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

SIGNATURE = b"\x89PNG\r\n\x1a\n"

#: Farbart -> Anzahl Samples pro Pixel (PNG-Spec, Tabelle 11.1)
CHANNELS_PER_COLOR_TYPE = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}

#: Farbart -> erlaubte Bittiefen (PNG-Spec, Tabelle 11.1)
VALID_BIT_DEPTHS = {
    0: {1, 2, 4, 8, 16},
    2: {8, 16},
    3: {1, 2, 4, 8},
    4: {8, 16},
    6: {8, 16},
}

COLOR_TYPE_NAMES = {
    0: "Graustufen",
    2: "RGB",
    3: "Palette",
    4: "Graustufen+Alpha",
    6: "RGBA",
}

#: Adam7-Paesse (RFC 2083, 2.6 / PNG-Spec 8.1): (Startzeile, Startspalte,
#: Zeilenschritt, Spaltenschritt). Aus dem 8x8-Grid des Specs abgeleitet und
#: per Property-Test gegen genau dieses Grid geprueft.
ADAM7_PASSES = (
    (0, 0, 8, 8),  # Pass 1: alle 8. Zeile/Spalte
    (0, 4, 8, 8),  # Pass 2
    (4, 0, 8, 4),  # Pass 3: Spaltenschritt 4, nicht 8
    (0, 2, 4, 4),  # Pass 4
    (2, 0, 4, 2),  # Pass 5
    (0, 1, 2, 2),  # Pass 6
    (1, 0, 2, 1),  # Pass 7: alle Spalten der ungeraden Zeilen
)


class PngError(Exception):
    """Basisklasse aller PNG-Fehler dieses Moduls."""


class PngFormatError(PngError):
    """Datei verletzt die PNG-Spec (Signatur, CRC, Chunk-Reihenfolge, …)."""


class PngUnsupportedError(PngError):
    """Gültige PNG-Datei, aber dieses Modul kann das Feature nicht lesen (z. B. Adam7)."""


def bpp(bit_depth: int, channels: int) -> int:
    """Bytes pro Pixel für die Filter-Rekonstruktion.

    Filter arbeiten auf Bytes, nicht auf Pixeln. Bei sub-byte-Bittiefen
    (1/2/4 Bit) teilen sich mehrere Pixel ein Byte -> mindestens 1 Byte.
    """
    return max(1, (bit_depth * channels + 7) // 8)


def _paeth(a: int, b: int, c: int) -> int:
    """Paeth-Predictor (PNG-Spec, 9.2): closest of a / b / c to p = a + b - c."""
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


@dataclass(frozen=True)
class Chunk:
    """Ein gelesener PNIL-Chunk mit seinem rohen Rahmen."""

    offset: int  # Position des Laengenfelds in der Datei
    type_name: str  # 4 ASCII-Buchstaben, z. B. "IHDR"
    data: bytes
    crc_stored: int
    crc_computed: int

    @property
    def is_critical(self) -> bool:
        return self.type_name[0].isupper()

    @property
    def is_ancillary(self) -> bool:
        return not self.is_critical

    @property
    def is_private(self) -> bool:
        return self.type_name[1].islower()

    @property
    def is_safe_to_copy(self) -> bool:
        return self.type_name[3].islower()

    @property
    def is_reserved_bit_violated(self) -> bool:
        # 3. Buchstabe muss Grossbuchstabe sein (reserviertes Bit = 0)
        return self.type_name[2].islower()

    def flags(self) -> set[str]:
        """Menschlich lesbare Chunk-Flags (fuehrende 4 Buchstaben decodiert)."""
        flags = {"critisch" if self.is_critical else "nebenbedeutend"}
        flags |= {"privat" if self.is_private else "oeffentlich"}
        flags |= {"safe-to-copy" if self.is_safe_to_copy else "nicht safe-to-copy"}
        if self.is_reserved_bit_violated:
            flags |= {"reserviert-bit-verletzt"}
        return flags

    def crc_ok(self) -> bool:
        return self.crc_stored == self.crc_computed


@dataclass(frozen=True)
class Ihdr:
    """Inhalt des IHDR-Chunks (immer der erste Chunk, 13 Bytes)."""

    width: int
    height: int
    bit_depth: int
    color_type: int
    compression: int
    filter_method: int
    interlace: int


@dataclass
class PngImage:
    """Dekodiertes PNG: Metadaten + unfilterte Scanlines + Auspack-Arbeit."""

    ihdr: Ihdr
    chunks: list[Chunk]
    palette: tuple[tuple[int, int, int], ...] | None
    trns: bytes | None
    text: dict[str, str]
    _rows: list[bytes]  # unfilterte Scanlines, Zeilen ohne Filterbyte (interlace 0)
    _samples: list[list[int]] | None = None  # Sample-Matrix (interlace 1)

    @property
    def width(self) -> int:
        return self.ihdr.width

    @property
    def height(self) -> int:
        return self.ihdr.height

    @property
    def bit_depth(self) -> int:
        return self.ihdr.bit_depth

    @property
    def color_type(self) -> int:
        return self.ihdr.color_type

    @property
    def color_type_name(self) -> str:
        return COLOR_TYPE_NAMES[self.color_type]

    @property
    def interlace(self) -> int:
        return self.ihdr.interlace

    @property
    def channels(self) -> int:
        return CHANNELS_PER_COLOR_TYPE[self.color_type]

    @property
    def stride(self) -> int:
        """Bytes einer Zeile (ohne Filterbyte), auf Ganzes Byte gerundet."""
        return (self.width * self.bit_depth * self.channels + 7) // 8

    def raw_rows(self) -> list[bytes]:
        """Unfilterte Scanlines (ohne Filterbyte); bei Adam7-Interlace leer."""
        return list(self._rows)

    def _iter_sample_rows(self) -> Iterator[list[int]]:
        """Je Bildzeile die Roh-Samples (Basis der Farbkonvertierung)."""
        if self._samples is not None:
            yield from self._samples
            return
        for row in self._rows:
            yield _expand_samples(row, self.width * self.channels, self.bit_depth)

    def iter_rgba(self) -> Iterator[tuple[int, int, int, int]]:
        """Alle Pixel in Rasterreihenfolge als (r, g, b, a) mit 8 Bit je Kanal.

        16-Bit-Bilder werden auf 8 Bit gekuerzt (>> 8), sub-byte-Bittiefen
        proportional skaliert (Wert * 255 // max). Palette wird ueber PLTE
        und tRNS aufgeloest, Graustufen/RGB-Einfarbtransparenz ueber tRNS.
        Funktioniert identisch fuer non-interlaced und Adam7-PNGs.
        """
        depth = self.bit_depth
        ch = self.channels

        def scale(v: int) -> int:
            if depth == 16:
                return v >> 8
            max_v = (1 << depth) - 1
            return v * 255 // max_v

        trns = self.trns
        for row_samples in self._iter_sample_rows():
            for px in range(self.width):
                s = row_samples[px * ch : (px + 1) * ch]
                ct = self.color_type
                if ct == 0:
                    g = scale(s[0])
                    a = 0 if trns and int.from_bytes(trns[:2], "big") == s[0] else 255
                    yield (g, g, g, a)
                elif ct == 2:
                    r, g, b = (scale(x) for x in s)
                    equal = trns and tuple(s) == struct.unpack(">3H", trns[:6])
                    yield (r, g, b, 0 if equal else 255)
                elif ct == 3:
                    assert self.palette is not None
                    r, g, b = self.palette[s[0]]
                    a = trns[s[0]] if trns and s[0] < len(trns) else 255
                    yield (r, g, b, a)
                elif ct == 4:
                    g = scale(s[0])
                    a = scale(s[1])
                    yield (g, g, g, a)
                else:
                    r, g, b, a = (scale(x) for x in s)
                    yield (r, g, b, a)

    def to_rgba_rows(self) -> list[list[tuple[int, int, int, int]]]:
        """Alle Zeilen als Liste von (r, g, b, a)-Tupeln."""
        it = iter(self.iter_rgba())
        return [[next(it) for _ in range(self.width)] for _ in range(self.height)]

    def save_ppm(self, path: str) -> None:
        """Als PPM (P6, RGB) schreiben — Alpha wird verworfen (PPM kennt kein Alpha)."""
        write_ppm(self, path)


# --------------------------------------------------------------------------
# Lesen: Signatur, Chunks, CRC
# --------------------------------------------------------------------------


def read_chunks(data: bytes, *, verify_crc: bool = True) -> list[Chunk]:
    """Alle Chunks bis IEND lesen (Signatur wird geprueft, Rahmen/CRC validiert)."""
    if len(data) < 8 or data[:8] != SIGNATURE:
        raise PngFormatError(
            "PNG-Signatur fehlt oder falsch: erwartet b'\\x89PNG\\r\\n\\x1a\\n'"
        )
    chunks: list[Chunk] = []
    pos = 8
    while True:
        if pos + 8 > len(data):
            raise PngFormatError(
                f"Chunk an Position {pos} abgeschnitten: {len(data) - pos} Byte "
                f"übrig, doch Chunkkopf braucht 8"
            )
        length = int.from_bytes(data[pos : pos + 4], "big")
        ctype = data[pos + 4 : pos + 8]
        if not all(65 <= byte <= 90 or 97 <= byte <= 122 for byte in ctype):
            raise PngFormatError(
                f"Chunk-Typ an Position {pos + 4} ist kein 4er-ASCII-Buchstabenwort: {ctype!r}"
            )
        if length > 0x7FFFFFFF:
            raise PngFormatError(f"Chunk-Länge {length} über der Spec-Grenze 2^31-1")
        chunk_end = pos + 8 + length + 4
        if chunk_end > len(data):
            raise PngFormatError(
                f"Chunk {ctype.decode('ascii', 'replace')} an Position {pos} "
                f"abgeschnitten: braucht {length + 12} Bytes, Datei endet nach "
                f"{len(data) - pos}"
            )
        chunk_data = data[pos + 8 : pos + 8 + length]
        crc_stored = int.from_bytes(data[chunk_end - 4 : chunk_end], "big")
        crc_computed = zlib.crc32(ctype + chunk_data) & 0xFFFFFFFF
        if verify_crc and crc_stored != crc_computed:
            raise PngFormatError(
                f"CRC32 passt nicht bei Chunk {ctype.decode('ascii', 'replace')} "
                f"an Position {pos}: gespeichert {crc_stored:08X}, berechnet {crc_computed:08X}"
            )
        chunks.append(
            Chunk(
                offset=pos,
                type_name=ctype.decode("latin-1"),
                data=chunk_data,
                crc_stored=crc_stored,
                crc_computed=crc_computed,
            )
        )
        pos = chunk_end
        if ctype == b"IEND":
            break
    if pos != len(data):
        raise PngFormatError(
            f"{len(data) - pos} Bytes Daten hinter IEND (IEND muss letzter Chunk sein)"
        )
    return chunks


def _parse_ihdr(chunk: Chunk) -> Ihdr:
    if len(chunk.data) != 13:
        raise PngFormatError(f"IHDR muss 13 Bytes Data haben, hat {len(chunk.data)}")
    width, height, depth, ct, compression, filter_method, interlace = struct.unpack(
        ">IIBBBBB", chunk.data
    )
    if width == 0 or height == 0:
        raise PngFormatError("Breite und Höhe müssen größer als 0 sein")
    if ct not in CHANNELS_PER_COLOR_TYPE:
        raise PngFormatError(f"Unbekannte Farbart {ct} (erlaubt: 0, 2, 3, 4, 6)")
    if depth not in VALID_BIT_DEPTHS[ct]:
        allowed = ", ".join(str(d) for d in sorted(VALID_BIT_DEPTHS[ct]))
        raise PngFormatError(
            f"Bittiefe {depth} ist für Farbart {COLOR_TYPE_NAMES[ct]} nicht erlaubt "
            f"(erlaubt: {allowed})"
        )
    if compression != 0:
        raise PngFormatError(
            f"Kompressionsmethode {compression} unbekannt (nur 0 = deflate)"
        )
    if filter_method != 0:
        raise PngFormatError(
            f"Filtermethode {filter_method} unbekannt (nur 0 = adaptiv)"
        )
    if interlace not in (0, 1):
        raise PngFormatError(f"Interlace-Methode {interlace} unbekannt (0 oder 1)")
    return Ihdr(width, height, depth, ct, compression, filter_method, interlace)


def _split_middle_chunks(
    chunks: list[Chunk], ihdr: Ihdr
) -> tuple[
    tuple[tuple[int, int, int], ...] | None, bytes | None, dict[str, str], list[bytes]
]:
    """Chunks zwischen IHDR und IEND einsammeln: PLTE, tRNS, tEXt, IDAT."""
    palette_raw: bytes | None = None
    trns: bytes | None = None
    text: dict[str, str] = {}
    idats: list[bytes] = []
    prev_name = "IHDR"
    idat_started = False
    for chunk in chunks[1:-1]:
        name = chunk.type_name
        if name == "IHDR":  # kann hier nicht vorkommen (read_chunks bricht bei IEND)
            raise PngFormatError("IHDR darf nur einmal direkt am Anfang stehen")
        if name == "PLTE":
            if palette_raw is not None:
                raise PngFormatError(
                    "PLTE darf nur einmal vorkommen (zweiter Chunk gefunden)"
                )
            if idat_started:
                raise PngFormatError("PLTE muss vor dem ersten IDAT kommen")
            if ihdr.color_type in (0, 4):
                raise PngFormatError(
                    f"PLTE ist bei Farbart {COLOR_TYPE_NAMES[ihdr.color_type]} verboten"
                )
            if ihdr.color_type in (2, 6):
                raise PngFormatError(
                    f"PLTE ist bei Farbart {COLOR_TYPE_NAMES[ihdr.color_type]} nicht erlaubt "
                    "(Truecolor braucht keine Palette)"
                )
            if len(chunk.data) == 0 or len(chunk.data) % 3 != 0:
                raise PngFormatError(
                    f"PLTE-Länge {len(chunk.data)} ist nicht (positives) Vielfaches von 3"
                )
            entries = len(chunk.data) // 3
            if entries > 2**ihdr.bit_depth:
                raise PngFormatError(
                    f"PLTE hat {entries} Einträge, mehr als {2**ihdr.bit_depth} "
                    f"(Bittiefe {ihdr.bit_depth}) erlaubt"
                )
            palette_raw = chunk.data
        elif name == "tRNS":
            if trns is not None:
                raise PngFormatError(
                    "tRNS darf nur einmal vorkommen (zweiter Chunk gefunden)"
                )
            if ihdr.color_type in (4, 6):
                raise PngFormatError(
                    f"tRNS ist bei Farbart {COLOR_TYPE_NAMES[ihdr.color_type]} verboten "
                    "(Kanäle existieren schon)"
                )
            if ihdr.color_type == 0 and len(chunk.data) != 2:
                raise PngFormatError(
                    f"tRNS für Graustufen braucht 2 Bytes, hat {len(chunk.data)}"
                )
            if ihdr.color_type == 2 and len(chunk.data) != 6:
                raise PngFormatError(
                    f"tRNS für RGB braucht 6 Bytes, hat {len(chunk.data)}"
                )
            if ihdr.color_type == 3:
                entries = len(palette_raw) // 3 if palette_raw else 0
                if len(chunk.data) > entries:
                    raise PngFormatError(
                        f"tRNS ({len(chunk.data)} Einträge) länger als PLTE ({entries} Einträge)"
                    )
            trns = chunk.data
        elif name == "tEXt":
            if b"\x00" in chunk.data:
                keyword, _, value = chunk.data.partition(b"\x00")
                text[keyword.decode("latin-1")] = value.decode("latin-1")
        elif name == "IDAT":
            if prev_name != "IDAT":
                if idat_started:
                    raise PngFormatError(
                        f"IDAT-Chunks müssen konsekutiv sein, dazwischen liegt {prev_name}"
                    )
                idat_started = True
            idats.append(chunk.data)
        elif name in ("IEND",):
            raise PngFormatError("IEND darf nur als letzter Chunk vorkommen")
        prev_name = name
    palette = None
    if palette_raw is not None:
        palette = tuple(
            (palette_raw[i], palette_raw[i + 1], palette_raw[i + 2])
            for i in range(0, len(palette_raw), 3)
        )
    return palette, trns, text, idats


# --------------------------------------------------------------------------
# Deko: Scanlines unfiltern und Samples auspacken
# --------------------------------------------------------------------------


def _unfilter_rows(raw: bytes, stride: int, byte_pp: int, height: int) -> list[bytes]:
    """Alle Scanlines rekonstruieren (Filter umdrehen)."""
    rows: list[bytes] = []
    prev = bytearray(stride)
    for y in range(height):
        base = y * (stride + 1)
        filter_type = raw[base]
        row = bytearray(raw[base + 1 : base + 1 + stride])
        if filter_type not in (0, 1, 2, 3, 4):
            raise PngFormatError(
                f"Unbekannter Filtertyp {filter_type} in Scanline {y} (erlaubt: 0-4)"
            )
        if filter_type == 1:  # Sub: links addieren
            for i in range(byte_pp, stride):
                row[i] = (row[i] + row[i - byte_pp]) & 0xFF
        elif filter_type == 2:  # Up: oben addieren
            for i in range(stride):
                row[i] = (row[i] + prev[i]) & 0xFF
        elif filter_type == 3:  # Average: Mitte von links und oben
            for i in range(stride):
                left = row[i - byte_pp] if i >= byte_pp else 0
                row[i] = (row[i] + (left + prev[i]) // 2) & 0xFF
        elif filter_type == 4:  # Paeth
            for i in range(stride):
                left = row[i - byte_pp] if i >= byte_pp else 0
                above = prev[i]
                above_left = prev[i - byte_pp] if i >= byte_pp else 0
                row[i] = (row[i] + _paeth(left, above, above_left)) & 0xFF
        rows.append(bytes(row))
        prev = row
    return rows


def _decode_adam7(raw: bytes, ihdr: Ihdr) -> list[list[int]]:
    """Adam7-Paesse entzerren -> Sample-Matrix (Hoehe x Breite*Kanaele).

    Jeder Pass ist ein eigenes Mini-Bild: eigener Zeilenschritt, eigene
    Filterbytes und ein Pass-lokaler Filterkontext (Prev-Zeile = vorherige
    Zeile desselben Passes). Leere Paesse tragen keine Filterbytes.
    """
    ch = CHANNELS_PER_COLOR_TYPE[ihdr.color_type]
    width, height, depth = ihdr.width, ihdr.height, ihdr.bit_depth
    matrix: list[list[int]] = [[0] * (width * ch) for _ in range(height)]
    pos = 0
    for pass_no, (y0, x0, ystep, xstep) in enumerate(ADAM7_PASSES, start=1):
        pw = (width - x0 + xstep - 1) // xstep
        ph = (height - y0 + ystep - 1) // ystep
        if pw <= 0 or ph <= 0:
            continue
        stride = (pw * depth * ch + 7) // 8
        needed = ph * (stride + 1)
        if pos + needed > len(raw):
            raise PngFormatError(
                f"IDAT-Rohdaten abgeschnitten: Pass {pass_no} braucht {needed} "
                f"Bytes, nur {len(raw) - pos} übrig"
            )
        pass_rows = _unfilter_rows(raw[pos : pos + needed], stride, bpp(depth, ch), ph)
        pos += needed
        for j, pass_row in enumerate(pass_rows):
            samples = _expand_samples(pass_row, pw * ch, depth)
            y = y0 + j * ystep
            for i in range(pw):
                x = x0 + i * xstep
                matrix[y][x * ch : (x + 1) * ch] = samples[i * ch : (i + 1) * ch]
    if pos != len(raw):
        raise PngFormatError(
            f"IDAT-Rohdaten: {len(raw) - pos} überschüssige Bytes nach Pass 7"
        )
    return matrix


def _expand_samples(row: bytes, count: int, depth: int) -> list[int]:
    """Bytes -> Naive Sampleliste (value pro Sample, sub-byte: MSB zuerst)."""
    if depth == 8:
        return list(row[:count])
    if depth == 16:
        return [int.from_bytes(row[2 * i : 2 * i + 2], "big") for i in range(count)]
    samples: list[int] = []
    mask = (1 << depth) - 1
    for byte in row:
        for shift in range(8 - depth, -1, -depth):
            samples.append((byte >> shift) & mask)
    return samples[:count]


# --------------------------------------------------------------------------
# Hohe Ebene: decode_png / load / write_ppm
# --------------------------------------------------------------------------


def decode_png(data: bytes, *, verify_crc: bool = True) -> PngImage:
    """PNG-Bytes komplett dekodieren (Format -> Bild)."""
    chunks = read_chunks(data, verify_crc=verify_crc)
    if chunks[0].type_name != "IHDR":
        raise PngFormatError(
            f"Erster Chunk ist {chunks[0].type_name}, aber muss IHDR sein"
        )
    if chunks[-1].type_name != "IEND":
        raise PngFormatError(
            f"Letzter Chunk ist {chunks[-1].type_name}, aber muss IEND sein"
        )
    ihdr = _parse_ihdr(chunks[0])
    palette, trns, text, idats = _split_middle_chunks(chunks, ihdr)
    if ihdr.color_type == 3 and palette is None:
        raise PngFormatError("Farbart Palette braucht einen PLTE-Chunk, keiner da")
    if not idats:
        raise PngFormatError(
            "Kein IDAT-Chunk: ohne Bild-Daten ist das kein lesbares PNG"
        )
    try:
        raw = zlib.decompress(b"".join(idats))
    except zlib.error as exc:
        raise PngFormatError(f"IDAT: deflate-Daten sind korrupt ({exc})") from exc

    ch = CHANNELS_PER_COLOR_TYPE[ihdr.color_type]
    if ihdr.interlace == 1:
        matrix = _decode_adam7(raw, ihdr)
        if ihdr.color_type == 3:
            _validate_palette_indices(palette, matrix)
        return PngImage(
            ihdr=ihdr,
            chunks=chunks,
            palette=palette,
            trns=trns,
            text=text,
            _rows=[],
            _samples=matrix,
        )
    stride = (ihdr.width * ihdr.bit_depth * ch + 7) // 8
    byte_pp = bpp(ihdr.bit_depth, ch)
    expected = ihdr.height * (stride + 1)
    if len(raw) != expected:
        raise PngFormatError(
            f"IDAT-Rohdaten falsch: {len(raw)} Bytes, erwartet {expected} "
            f"({ihdr.height} Scanlines à {stride + 1} Bytes mit Filterbyte)"
        )
    rows = _unfilter_rows(raw, stride, byte_pp, ihdr.height)
    if ihdr.color_type == 3:
        # Pixel-Indizes gegen die reale Palettenlaenge pruefen, sonst wuerde
        # erst iter_rgba() mit rohem IndexError abstuerzen.
        _validate_palette_indices(
            palette, [_expand_samples(row, ihdr.width, ihdr.bit_depth) for row in rows]
        )
    return PngImage(
        ihdr=ihdr,
        chunks=chunks,
        palette=palette,
        trns=trns,
        text=text,
        _rows=rows,
    )


def _validate_palette_indices(
    palette: tuple[tuple[int, int, int], ...] | None, sample_rows: list[list[int]]
) -> None:
    """Alle Pixel-Indizes gegen die reale Palettenlaenge pruefen, sonst wuerde
    erst iter_rgba() mit rohem IndexError abstuerzen."""
    assert palette is not None
    entries = len(palette)
    for y, row_samples in enumerate(sample_rows):
        for x, index in enumerate(row_samples):
            if index >= entries:
                raise PngFormatError(
                    f"Pixel-Index {index} in Zeile {y}, Spalte {x} zeigt hinter die "
                    f"Palette ({entries} Einträge)"
                )


def load(path: str, *, verify_crc: bool = True) -> PngImage:
    """PNG-Datei lesen und dekodieren."""
    with open(path, "rb") as fh:
        data = fh.read()
    return decode_png(data, verify_crc=verify_crc)


def write_ppm(img: PngImage, path: str) -> None:
    """Als PPM (P6, RGB) exportieren — Alpha geht verloren, PPM kennt kein Alpha."""
    with open(path, "wb") as fh:
        fh.write(b"P6\n%d %d\n255\n" % (img.width, img.height))
        fh.write(b"".join(bytes((r, g, b)) for r, g, b, _ in img.iter_rgba()))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PNG-Datei ohne Bildbibliothek lesen (Lern-Parser, nur stdlib)."
    )
    parser.add_argument("path", help="PNG-Datei")
    parser.add_argument("--ppm", metavar="ZIEL", help="Nach PPM (P6) exportieren")
    parser.add_argument(
        "--pixels",
        metavar="N",
        type=int,
        default=None,
        help="Erste N Pixel als (r, g, b, a) zeigen",
    )
    parser.add_argument(
        "--no-crc", action="store_true", help="CRC32-Prüfung abschalten"
    )
    args = parser.parse_args(argv)

    # argparse-Attribute sind Any -> hier einmal auf die deklarierten Typen
    # gussednet, danach fortan typ-clean.
    path = cast(str, args.path)
    no_crc = cast(bool, args.no_crc)
    pixels = cast("int | None", args.pixels)
    ppm = cast("str | None", args.ppm)

    try:
        img = load(path, verify_crc=not no_crc)
    except (PngError, OSError) as exc:
        print(f"FEHLER: {exc}")
        return 1
    print(f"Datei: {path}")
    print(f"  Maße: {img.width} x {img.height}")
    print(
        f"  Farbart: {img.color_type} = {img.color_type_name} ({img.channels} Kanäle)"
    )
    print(f"  Bittiefe: {img.bit_depth} Bit")
    if img.interlace:
        print("  Interlace: Adam7 (7 Pässe)")
    else:
        print(f"  Zeilenstruktur: {img.stride + 1} Bytes (1 Filterbyte + {img.stride})")
    if img.palette is not None:
        print(f"  Palette: {len(img.palette)} Einträge")
    if img.trns:
        print(f"  tRNS: {len(img.trns)} Bytes")
    if img.text:
        for key, value in img.text.items():
            print(f"  tEXt {key}: {value}")
    print(f"  Chunks: {len(img.chunks)}")
    for chunk in img.chunks:
        flags = " ".join(sorted(chunk.flags()))
        crc = "CRC ok" if chunk.crc_ok() else f"CRC FEHLER ({chunk.crc_stored:08X})"
        print(
            f"    @{chunk.offset:>8} {chunk.type_name:<5} {len(chunk.data):>7} B  {flags}  {crc}"
        )
    if pixels is not None:
        for index, pixel in enumerate(img.iter_rgba()):
            if index >= pixels:
                break
            print(f"  Pixel {index}: {pixel}")
    if ppm:
        write_ppm(img, ppm)
        print(f"  Exportiert nach {ppm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
