# Slide 75 phase-sensitivity redesign

## Goal

Replace the unclear three-orbit sketch on slide 75 with a visual that lets a newcomer immediately understand the central iPRC idea: the same state-space kick can advance, barely change, or delay an oscillator depending on the phase at which it arrives.

## Scope

- Modify only the opening teaching frame in `notes/technical_presentation/chapters/09_prc_dprc.tex` unless a small shared-style adjustment is required by compilation.
- Preserve the Madrid theme, JaxCont palette, current-main badge, footer, and surrounding Chapter 9 narrative.
- Keep the final PDF at one assembled deck; page numbering may remain unchanged.

## Approved visual design

The frame will use intuition first and geometry second:

1. A shared reference oscillation/timing cue establishes the unperturbed rhythm and three possible kick phases: early, middle, and late.
2. Three compact before/after comparisons apply the same orange kick at those phases and show three qualitatively distinct timing outcomes:
   - phase advance;
   - little or approximately zero phase change;
   - phase delay.
3. A compact mathematical summary explains the picture:

   \[
   \Delta\phi \approx Z(\phi)^{\mathsf T}\delta x.
   \]

   The kick `\delta x` is unchanged, while the phase-dependent sensitivity `Z(\phi)` changes.

## Visual language

- Blue: unperturbed/reference oscillation or event time.
- Teal: perturbed oscillation or shifted event time.
- Orange: the identical state-space kick.
- Use explicit words—`advance`, `little change`, and `delay`—instead of asking viewers to infer the meaning from arc length or direction.
- Include a small legend or direct labels so the diagram is understandable without narration.
- Avoid overlapping annotations and avoid three nearly identical orbit circles.

## Scientific boundaries

- Describe a sufficiently small kick and a first-order phase response.
- Do not imply that every phase necessarily realizes exactly these three outcomes for every oscillator; present them as representative outcomes illustrating phase dependence.
- Do not imply a finite reset, basin crossing, or change to the periodic orbit itself.
- Retain the current-main/planned-for-v0.4 status.

## Acceptance criteria

- A newcomer can answer “what stayed the same?” and “what changed?” from the picture alone.
- The same kick is visibly reused in all three comparisons.
- Advance, near-zero response, and delay are visually distinguishable and textually labeled.
- The equation visibly connects the three outcomes to phase-dependent `Z(\phi)`.
- The rebuilt slide has no clipping, overlap, stretched content, or LaTeX layout warnings.
- `make verify` and the relevant presentation checks pass.
