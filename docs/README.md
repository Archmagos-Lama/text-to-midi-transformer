# Project materials

- [Portfolio (PDF)](Touhou_Music_Generation_Portfolio.pdf): project implementation, experiments and limitations.
- [BARPOS feature distributions](results/barpos/distributions.png).
- Results for `baseline`, `barpos`, `harmrole` and `harmrole_v2` are stored under `results/`. Each directory includes the original per-file metrics and evaluation plots.
- `samples/` contains seeds 0 and 1 from each unconditional generation variant. These fixed seeds are illustrative outputs, not a quality-ranked selection or the full evaluation set. MIDI playback requires a synthesizer or soundfont.

The evaluation script uses SciPy's `jensenshannon`, which returns Jensen-Shannon **distance**. Existing plot labels say JS; these should be interpreted as distance rather than divergence. The eight features describe statistical similarity, not overall musical quality or long-range structure.

Original MIDI arrangements, augmented training data, checkpoints, cached annotations and prompt continuations are not included in this update. The source music and arrangements belong to their respective creators; this repository does not grant rights to the source corpus.
