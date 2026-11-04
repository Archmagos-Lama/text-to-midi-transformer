
Temporal quantization is performed in beat space by mapping each note
independently to a fixed-resolution beat grid, rather than normalizing
over the total duration or total number of beats of the piece.

A 1/32-beat quantization strikes a balance between temporal resolution and
model complexity, preserving expressive timing in slow-tempo pieces while
avoiding spurious micro-timing noise.

Although the MIDI file contains nine instrument tracks, they are mapped to
six semantic roles to reflect musical function rather than DAW-specific
track separation.

Malformed or non-standard MIDI files were skipped during vocabulary
construction, following common practice in large-scale symbolic music
datasets such as Lakh.