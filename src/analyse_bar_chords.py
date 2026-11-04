"""
Analyse chord progression by bar using music21.
Prints the chord for each measure and compares with Hooktheory reference.
"""
import sys
from music21 import converter, chord, note, roman, key as m21key

def analyse_midi(midi_path, expected_key=None, max_bars=16):
    print(f"\n{'='*60}")
    print(f"File: {midi_path.split('/')[-1]}")
    score = converter.parse(midi_path)

    # detect key
    detected_key = score.analyze('key')
    print(f"Detected key: {detected_key}")
    if expected_key:
        print(f"Expected key: {expected_key}")

    k = detected_key

    # chordify the whole score, then iterate by measure
    chordified = score.chordify()

    bar_chords = []
    # chordify returns a Score; get its first part
    from music21 import stream
    if isinstance(chordified, stream.Score):
        part = chordified.parts[0]
    else:
        part = chordified
    for i, measure in enumerate(part.getElementsByClass('Measure')):
        if i >= max_bars:
            break

        # collect all chords in this measure, weighted by duration
        best_chord = None
        best_dur = 0
        for el in measure.flatten().getElementsByClass('Chord'):
            if float(el.duration.quarterLength) > best_dur and len(el.pitches) >= 2:
                best_dur = float(el.duration.quarterLength)
                best_chord = el

        if best_chord is None:
            bar_chords.append(f"m{i+1:2d}: ---")
            continue

        try:
            rn = roman.romanNumeralFromChord(best_chord, k)
            chord_name = best_chord.commonName
            root = best_chord.root().name
            bar_chords.append(f"m{i+1:2d}: {rn.figure:6s} ({root} {chord_name[:12]})")
        except Exception as e:
            bar_chords.append(f"m{i+1:2d}: ?  ({e})")

    for entry in bar_chords:
        print(f"  {entry}")

# Hartmann's Youkai Girl - Hooktheory says A minor: VII-i-i-VI-VII-i-VII (Intro), VI-VII-i-VII-VI-VII (Chorus)
analyse_midi(
    "data/touhou_midi_collection/1 - Shooter Games/11 - Subterranean Animism/15. Hartmann's Youkai Girl (akira).mid",
    expected_key="a minor",
    max_bars=16
)

# U.N. Owen was Her - check a second song
analyse_midi(
    "data/touhou_midi_collection/1 - Shooter Games/6 - The Embodiment of Scarlet Devil/15. U.N. Owen was her (ZUN).mid",
    max_bars=16
)
