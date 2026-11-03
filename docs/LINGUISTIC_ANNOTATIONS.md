# Linguistic Annotation System

## Purpose

Schema 2.0.0 replaces selected hide candidates with neutral linguistic annotations. The Agent identifies concepts and maps their appearances in source subtitles. It does not choose playback visibility, learner-specific difficulty policy, or a hide subset.

## Selective annotation policy

The model looks for meaningful units with practical language-learning value and avoids redundant constituent entries when a complete expression conveys the same concept. It may annotate useful basic vocabulary; ease is not an exclusion rule. Ordinary function words without independent learning value, irrelevant proper names, isolated morphemes/fragments, and redundant parts of expressions are generally excluded. Overlap is allowed only for genuinely different concepts.

Coverage is measured over reconstructed semantic sentences, not subtitle cues or concept count: a sentence is covered when at least one validated occurrence is linked to it. The Agent aims for 30% coverage as a soft goal. If the initial result is below 30%, it asks Gemini for one supplemental review of unannotated sentences, without requiring a candidate where none is worthwhile. Python caps final coverage at 60%, keeping sentences with higher-priority concepts first and warning for excluded sentences. The report includes the final numerator, denominator, ratio, whether the supplemental review ran, and any shortfall or cap exclusions. This episode-wide coverage policy does not change the existing maximum of two word occurrences per semantic sentence.

Analysis and occurrence prompts may include source-language-specific guidance. Spanish and French have dedicated additions selected from the normalized source-language code; other source languages use the common prompt. French guidance distinguishes lexical words from agreement, pronoun/clitic, tense/aspect, negation, and register patterns, and asks for exact observed spellings including accents, elisions, and hyphens. Python searches the model-supplied exact forms and retains exact source offsets.

The taxonomy is fixed:

- `noun`
- `verb`
- `adjective`
- `adverb`
- `fixed_expression`
- `idiom`
- `collocation`
- `grammar_pattern`
- `other`

### Difficulty rubric

| Level | Approximate classification | Guidance |
|---|---|---|
| 1 | A1 | Basic vocabulary and structures |
| 2 | A2 | Elementary everyday language |
| 3 | B1–B2 | Intermediate vocabulary and structures |
| 4 | C1–C2 | Advanced or specialized language |

These are approximate learning labels, not official CEFR certifications. The rubric considers semantic abstraction, structural complexity, register, and learner familiarity. It is not based on subtitle timing, visibility, story importance, or learner performance, and frequency alone does not set the level. The rubric is included in the Gemini prompt and validated as an integer from 1 through 4. Category and difficulty are separate fields.

## Data model

### Concept (`vocabulary.json` → `concepts[]`)

Each concept is represented once:

- `concept_id`: deterministic ID derived from normalized category, lemma, and `meaning_key`.
- `lemma`: canonical word or canonical expression.
- `meaning_key`: compact target-language sense key used to keep polysemous concepts distinct and merge equivalent concepts across batches.
- `category`: one taxonomy value above.
- `frequency`: count of mapped linguistic instances in this subtitle file. This is episode frequency, not an external corpus rating.
- `difficulty`: rubric level 1–4.
- `selection_priority`: `high`, `medium`, or `low` pedagogical priority used to rank word candidates when a semantic sentence contains more than two word occurrences.
- `selection_reason`: concise explanation of the concept's independent learning value.
- `translation`, `explanation`: concept-level meaning and concise learning note, stored once.
- `occurrence_ids`: cue-local occurrence records linked to the concept.
- `surface_forms`: exact source forms used during location.

Same lemma/category/meaning concepts get the same ID even when surface forms differ grammatically. Different meaning keys produce distinct IDs for different senses.

### Occurrence (`subtitles.json` → `linguistic_occurrences[]`)

Each cue-local piece contains:

- `occurrence_id`: deterministic unique ID for concept + cue + offsets.
- `instance_id`: groups cue-local pieces that together express one linguistic instance; cross-cue expression pieces share this ID.
- `concept_id`: the owning concept.
- `cue_id`, `text`, `start_char`, `end_char`: cue-local source fragment and its exact half-open span.
- `sentence_ids`: semantic sentence references supported by the occurrence mapping.

Offsets are calculated against the unmodified `display_cues[].source_text` using Python Unicode code-point indexing; `end_char` is exclusive. A phrase spanning cues yields a separate exact substring occurrence in each spoken cue, linked by the same `instance_id`. The mapper may bridge a recognized bracketed non-linguistic cue (such as `[Aplausos]`), but it never annotates the sound cue. Python handles NFC-equivalent Unicode, case, whitespace, and hyphen separators while retaining offsets into the original text. Nouns and adjectives get conservative Spanish gender/number variants; verb forms must be explicitly supplied from observed subtitle text. Frequency counts distinct `instance_id` values, so a cross-cue phrase counts once.

Display cues keep original IDs, sequence, timing, and source segmentation. Their `linguistic_occurrence_ids` link to cue-local occurrence records. Sentences use `concept_ids`; quiz questions use Python-resolved `concept_ids` and supporting `sentence_ids`.

Each quiz container carries one `trigger_time` indicating when the quiz appears. An in-video checkpoint stores its appearance time on the checkpoint object; the `final_quiz` object uses the video duration. Individual questions inherit the timing of their containing quiz. Culture items carry `trigger_time` derived from the earliest start time of their associated subtitle cues.

For quiz generation, Gemini returns sentence IDs and a text concept hint (`concept_lemma` and/or `concept_meaning`), never an opaque concept ID. Python restricts candidates to concepts attached to the cited sentences and writes the official `concept_id` only for a unique match. Missing or ambiguous matches leave `concept_ids` empty and create a `quiz_warnings` entry for human review. Content-comprehension questions may intentionally have an empty concept list.

Word categories (`noun`, `verb`, `adjective`, and `adverb`) are capped at two published occurrences per semantic sentence. Expressions and grammar patterns are not counted against this word cap. AI proposes word candidates and priorities; Python locates occurrences and applies the cap, retaining the strongest candidates and recording excluded candidates as `single_word_sentence_cap` warnings. There is no episode-wide word ratio cap in the Agent output.

### Translation alignment (`subtitles.json` → `translation_alignments[]`)

After cue translations are finalized, Gemini aligns each source occurrence to zero or more target-language texts in the same display cue. Gemini does not provide character offsets. Python searches each returned text in the final translation and writes a stable `alignment_id`, source `occurrence_id`, `cue_id`, exact translation `text`, half-open Unicode code-point offsets, `alignment_type` (`direct`, `paraphrase`, or `split`), and confidence. Multiple records may point to one source occurrence when its meaning is split across target text. The source concept difficulty remains authoritative; target spans do not receive an independently inferred level. A consumer can apply the source occurrence's selected visibility to every linked target span.

An omitted, ambiguous, or uncertain translation match remains unaligned and produces a warning. A target meaning that crosses cue boundaries is also left unaligned for review. Repeated target text is assigned left-to-right using the model's occurrence order only when that makes the mapping deterministic. The model proposes semantic correspondence; Python locates target text, assigns offsets and IDs, and checks references. It does not guess missing spans.

## Processing flow

1. Gemini analyzes each translated semantic-sentence batch and proposes new concepts using the taxonomy, difficulty rubric, separate word/expression selection guidance, and source-language-specific additions where available. Known concepts are supplied to later batches to discourage duplicate definitions.
2. Python normalizes concept identity and merges repeated category/lemma/meaning keys. The first definition is retained; differences in later metadata are warnings rather than silently replacing the concept explanation.
3. Gemini makes a separate occurrence-mapping pass over every sentence batch, using the complete concept list. It returns concept IDs, observed surface forms, and sentence IDs, without regenerating concept definitions.
4. Python searches original cue text within each referenced sentence context, locates supplied surface forms plus conservative noun/adjective variants, and calculates exact offsets. It can join spoken cue text over recognized non-linguistic interruptions and emits one local piece per contributing spoken cue.
5. Python applies the two-word-per-semantic-sentence cap to located word occurrences, then computes episode frequency, occurrence IDs, sentence/cue references, and logging summaries. It reports a genuinely unmatched concept/sentence context separately from an alternative form that was absent where another form mapped. Concepts with no located occurrence are excluded from the published vocabulary and appear in `annotation_warnings` with `zero_occurrence_concept_not_published`.
6. Cue translations receive full-sentence and neighboring-cue context. A conservative lexical check flags cue translation words absent from the linked sentence translation; these are review warnings, not semantic proof. Gemini review also checks cue/sentence alignment.
7. Gemini aligns the already located source occurrences to target-language texts in finalized cue translations. Python searches the final translations, calculates exact target offsets, generates stable alignment IDs, and records missing or uncertain matches.
8. Structural validation checks IDs, references, taxonomy, difficulty, frequency, offsets, duplicate spans, source-to-translation links, quiz and culture appearance times, non-linguistic cross-cue gaps, quiz evidence/option consistency, and independently validates all five JSON documents against the declared Draft 2020-12 schemas before ZIP export.

The model must report every concept and occurrence context it can identify in each batch. Python maps all reliably locatable matches within those reported contexts; it cannot independently prove that Gemini listed every semantically relevant concept or context. It does not invent missing dialogue. Unsupported zero-occurrence proposals are excluded from the vocabulary and remain visible as warnings. Cultural items include `needs_review` and `verification_note` so uncertain factual context can be flagged. Raw subtitle text and prompts remain out of event logs.

## Validation and logs

Validation checks each package document against its JSON Schema, unique concept/occurrence/alignment IDs, allowed categories and levels, exact source and translation Unicode offsets, cue/sentence/concept/quiz/alignment links, `frequency == distinct mapped instance count`, reciprocal occurrence IDs, duplicate occurrences, and valid cross-cue groups with only recognized non-linguistic gaps. Published concepts must have located occurrences. Quiz evidence must exactly quote a cited sentence, and option sets must be unique and consistently formatted. Distinct concepts may overlap; identical spans attached to multiple concepts are retained and reported for review.

Each run reports published concept count, cue-local occurrence count, repeated-concept count, category and difficulty distributions, genuine unmatched contexts, alternative-form mismatches, zero-occurrence exclusions, cue-translation warnings, ambiguity warnings, Gemini token usage when reported, retries, and processing failures. API secrets are not included.

## Breaking change from schema 1.0.0

- `subtitles.json` removes cue `hide_spans` and adds `linguistic_occurrences`, `annotation_warnings`, and cue/sentence concept references.
- `vocabulary.json` changes `items[]` to concept-level `concepts[]`.
- Quiz questions replace `annotation_ids` with `concept_ids`.
- `subtitles.json` adds optional `translation_warnings`; culture items may include `needs_review` and `verification_note`.
- Vocabulary concepts with no located occurrence are no longer published; their IDs and lemmas remain in annotation warnings.
- JSON `schema_version` changes to `2.0.0`.
- Schema-v1 resume caches cannot be resumed; use a fresh run after upgrading.
