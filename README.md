# png-parser — PNG von Hand durchleuchten (nur stdlib)

Ein PNG-**Decoder** von Grund auf: Signatur → Chunks → CRC32 → zlib →
Scanlines entzerren (Filter) → Pixel. Keine Bildbibliothek, 0 Dependencies,
offline — gebaut als Reverse-Engineering-Übung für das eigene Formatverständnis.

| Datei | Inhalt |
| --- | --- |
| `pngparser.py` | Parser + PPM-Export + CLI (`python pngparser.py bild.png`) |
| `test_pngparser.py` | 57 unittests mit eigenem Mini-Encoder als Fixture-Generator |
| `pyrightconfig.json` | basedpyright-Konfig (zwei Strikt-Regeln gedämpft) |
| `LICENSE` | MIT |

## Konzeptide

```python
img = decode_png(data)                        # oder load("bild.png")
img.width, img.height, img.color_type_name    # z. B. ("RGB", "Palette", …)
next(img.iter_rgba())                         # (r, g, b, a) in Rasterreihenfolge
img.raw_rows()                                # unfilterte Scanlines für Roh-Analyse
img.save_ppm("out.ppm")                       # ohne Lib sichtbar machen
```

Unterstützt: Farbarten 0/2/3/4/6, Bittiefen 1/2/4/8/16, alle 5 Filtertypen,
**Adam7-Interlace** (7 Pässe, pass-lokaler Filterkontext), mehrfache IDATs,
tRNS, tEXt, CRC32-Prüfung.

## CLI

```
python pngparser.py bild.png                 # Info + Chunktabelle + CRC-Status
python pngparser.py bild.png --ppm out.ppm   # in PPM (P6) exportieren
python pngparser.py bild.png --pixels 8      # erste 8 Pixel zeigen
python pngparser.py bild.png --no-crc        # CRC aus (Fehlersuche)
```

## Das Know-how — PNG-Dateistruktur Byte je Byte

**1. Signatur (8 Bytes): `89 50 4E 47 0D 0A 1A 0A`.**
`0x89` = oberstes Bit gesetzt (ausgedachter Magie-Wert, fängt 7-Bit-Kanäle),
`PNG` als ASCII, `\r\n` fängt DOS-Text/Binary-Verwechslung ab,
`\x1a` hält Unix `cat` an, `\n` fängt Unix-Roundtrip-Verwechslung ab.

**2. Chunks: `Länge (4 BE) + Typ (4 ASCII) + Daten + CRC32`.**
Big-Endian, Länge zählt nur die Daten. Der Typ sind 4 ASCII-Buchstaben;
je Bit 5 (Wert 1 = Kleinbuchstabe) ist ein Flag:

| Buchstabe | Bedeutung |
| --- | --- |
| 1. | groß = kritisch (Pflicht-Chunks), klein = nebengeordnet |
| 2. | groß = öffentlich, klein = privat (nutzerspezifisch) |
| 3. | muss groß sein — reserviert |
| 4. | klein = safe-to-copy (darf bei Feature-Sicherung erhalten bleiben) |

Beispiel `tEXt`: `t` klein → nebengeordnet, `E` groß → öffentlich,
`X` groß → reserviert ✓, `t` klein → safe-to-copy.

**3. IHDR (13 Bytes): `<IIBBBBB` = Breite, Höhe, Bittiefe, Farbart,
Kompression, Filtermethode, Interlace.** Erlaubte Bittiefen je Farbart
(Spec-Tabelle 11.1). Kanäle pro Pixel: 0→1, 2→3, 3→1, 4→2, 6→4.

**4. IDATs:** konsekutive Chunk-Daten konkatenieren, `zlib.decompress`.
Erwartete Größe: `Höhe × (1 + stride)` mit `stride = ⌈Breite·Bittiefe·Kanäle / 8⌉` —
vor jeder Zeile sitzt 1 Filterbyte.

**5. Filter (Spec 9.2) — die Falle, 3× abgegangen:**
Filter arbeiten auf **Bytes**, aber „links" meint das **Pixel** links, d. h.
`x − bpp` mit `bpp = max(1, ⌈Bittiefe·Kanäle / 8⌉)` — das gleiche Kanal-Byte
des Nachbarpixels, NICHT das Nachbar-Byte!

| Typ | Name | Recon = Filt + … |
| --- | --- | --- |
| 0 | None | — |
| 1 | Sub | `Raw(x − bpp)` (Links-Pixel, 0 wenn x < bpp) |
| 2 | Up | `Prior(x)` (Pixel oben, 0 in Zeile 0) |
| 3 | Average | `⌊(Raw(x − bpp) + Prior(x)) / 2⌋` |
| 4 | Paeth | nächster von a/b/c zu `p = a + b − c` |

Paeth-Wahlregel präzise: `pa = |p−a|, pb = |p−b|, pc = |p−c|`
→ `a` wenn `pa ≤ pb` **und** `pa ≤ pc`, sonst `b` wenn `pb ≤ pc`, sonst `c`.
Lern-Fall: vorletztes Byte `a=0, b=10, c=0` — `p=10`, `pb=0` minimal → **b** gewinnt!

**6. Sub-byte-Bittiefen (1/2/4):** mehrere Pixel teilen sich ein Byte,
**MSB zuerst**, Zeile auf Ganzes Byte aufgefüllt. Für Anzeige skalieren:
`Wert × 255 // max` (max = 2^Bittiefe − 1); 16 Bit → `>> 8`.

**7. tRNS:** Grau (2 B) / RGB (6 B) = Schlüsselfarbe auf **RAW-Ebene**
(vor Skalierung) vergleichen; Palette = Alpha-Bytes je Palette-Index (≤ PLTE-Größe).

**8. Adam7-Interlace (RFC 2083, 2.6):** 7 Pässe über ein 8×8-Raster:

| Pass | (Startz, Starts, Zeilenschritt, Spaltenschritt) |
| --- | --- |
| 1 | (0, 0, 8, 8) |
| 2 | (0, 4, 8, 8) |
| 3 | (4, 0, 8, 4) — Spaltenschritt **4**, nicht 8 |
| 4 | (0, 2, 4, 4) |
| 5 | (2, 0, 4, 2) |
| 6 | (0, 1, 2, 2) |
| 7 | (1, 0, 2, 1) — *alle* Spalten der ungeraden Zeilen |

Jeder Pass ist ein eigenes Mini-Bild: eigene Pass-Breite/-Höhe (⌈-Formeln),
eigene Filterbytes, **pass-lokaler Filterkontext** (Prev-Zeile = vorherige
Zeile desselben Passes), und Pass-Zeilen werden separat auf Ganzes Byte
gepolstert. Leere Pässe (unter 5 Spalten/Zeilen) tragen keine Filterbytes.
Der Decoder prüft die Pass-Tabelle per Property-Test gegen das 8×8-Grid
aus der Spec (jede Zelle genau einmal).

## Fehlerkanäle (alle mit deutlicher deutscher Meldung)

- falsche Signatur, abgeschnittene Chunks, CRC-Mismatch (mit gespeichert/berechnet)
- IHDR nicht am Anfang / IEND nicht am Ende / Daten nach IEND
- IDAT: nicht konsekutiv, deflate korrupt, falsche Rohdatenlänge
- unlautere Farbart/Bittiefe-Kombi, fehlerhafte PLTE (kein Vielfaches von 3,
  zu viele Einträge, verboten bei Graustufen)
- doppelte PLTE/tRNS (Spec: darf nur einmal vorkommen)
- Paletten-Index hinter der PLTE-Länge (bricht mit Meldung, nicht mit IndexError)
- überschüssige Bytes nach dem zlib-Stream im IDAT (stdlib verwirft sie still)

## 3 Fallen, die dieser Testlauf gelehrt hat (TESTED)

1. **`bytes((x))` ≠ `bytes((x,))`** — ohne Komma ist `bytes(int)` ein
   *Längen*-Konstruktor: `bytes(160)` = 160 Null-Bytes! Kostete 4 Errors.
2. **bpp-Filter-Falle** oben („links" = Pixel, nicht Byte) — Sub/Average/
   Paeth-Fixtures waren zunächst falsch handgerechnet.
3. **Paeth-Gleichstand:** `a` gewinnt (`pa ≤ pb` bindet), nicht `b`.

## Übungsaufgaben (nächste Ebene)

- [x] Adam7-Interlace (IMPLEMENTED + TESTED: Property-Test gegen Spec-Grid,
  Roundtrips für Farbarten/Bittiefen, pass-lokaler Filterkontext)
- [ ] PNG-**Writer** mit Filter-Optimalwahl (beste Filtertypen pro Zeile messen)
- [ ] Chunk-Writer (tEXt anpassen, ohne Bildbits zu berühren); BMP-Export
- [ ] Fuzzing: Byte-Flip-Loop automatisieren (Kaputt-Muster aus `kaputt.png`)
- [ ] Gegenprobe gegen Pillow über 1000 Zufalls-PNGs (wenn offline-Vorinstall)
- [ ] Weiter in der RE-Liste: ZIP-Dateiformat als nächste Übung (Nr. 2)

## Status (FACT / IMPLEMENTED / TESTED)

- IMPLEMENTED: Decoder (alle Farbarten/Bittiefen, non-interlaced UND Adam7),
  CRC-Check, tRNS/tEXt, PPM-Export, CLI, Review-Fixes (doppelte PLTE/tRNS
  abgelehnt, Paletten-Index-Bounds, CLI fängt auch OSError).
- TESTED: 58/58 unittests (`python -m unittest test_pngparser`), ruff clean,
  basedpyright 0 Errors / 0 Warnungen, CLI-Feldtest mit generiertem sowie
  gezielt korruptiertem PNG (CRC-Fehler-Fall: Exit 1, Fehler auf stderr).
