# Fade Content Agent

A local Python 3.12 command-line agent that turns an SRT/VTT subtitle file or YouTube video URL into a validated bilingual learning-content ZIP. It uses TranscriptAPI to retrieve YouTube captions and Google Gemini for language processing. It does not download video or transcribe audio.

## Setup

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp .env.example .env
```

Set `GEMINI_API_KEY` and optionally `GEMINI_MODEL` (`gemini-3.5-flash-lite` by default) in `.env` or the environment. Set `TRANSCRIPT_API_KEY` to enable YouTube URL input. Never commit `.env`.

## Run

### HTTP API

The optional API wraps the existing synchronous pipeline in an asynchronous
job interface. Keep `GEMINI_API_KEY` and `TRANSCRIPT_API_KEY` on the server;
the browser only receives an analysis id and the completed schema 2.0 package.
Set `FADE_AGENT_API_KEY` on the API host. All analysis, polling, and result
routes require this value as a Bearer token; `/health` remains public for
platform health checks. Configure the same value only in the protected
frontend's server-side proxy, never in browser code or an unrestricted frontend.

```sh
uvicorn fade_content_agent.api:app --host 0.0.0.0 --port 8080
```

Use `POST /analyses` with `youtube_url`, `source_language`, and
`target_language`, then poll `GET /analyses/{analysis_id}`. When the status is
`completed`, fetch `GET /analyses/{analysis_id}/result`. The API process can be
containerized with the included `Dockerfile`.

```sh
fade-content-agent episode.srt \
  --source-language es --target-language en \
  --video-id extra-episode-1 \
  --title "Extra en Español — Episode 1" --output-dir ./fade-output
```

YouTube input uses the same command and does not require a source language, video ID, or duration:

```sh
fade-content-agent "https://www.youtube.com/watch?v=VIDEO_ID" --target-language en
```

TranscriptAPI's default language selection is used unless `--language` is supplied. The actual returned language is stored in the ZIP metadata and used throughout the pipeline. For SRT/VTT input, the source language remains required and processing duration is taken from the final cue end time.

Optional glossary is a JSON file mapping source terms to preferred translations. Resume a schema-v2 run with `--resume <run-id>` using the same inputs and output directory. Schema-v1 run caches and ZIPs are not compatible; start a fresh run after upgrading. One successful run creates `outputs/<video-id>-<run-id>.zip` with exactly six root files: `manifest.json`, `subtitles.json`, `vocabulary.json`, `culture.json`, `quizzes.json`, and `review-report.md`. Per-run state and logs are kept outside the ZIP under `runs/<run-id>/`. YouTube runs also keep the raw TranscriptAPI response at `runs/<run-id>/input/transcriptapi-response.json` for comparison with normalized cues.

## Linguistic annotations (schema 2.0.0)

The Agent selects meaningful learning concepts, not hide candidates. `vocabulary.json` stores each located concept once (`concept_id`, canonical `lemma`, category, episode frequency, difficulty 1–4, translation, explanation, and occurrence references). Concepts with no located occurrence are excluded and reported as warnings. `subtitles.json` stores cue-local `linguistic_occurrences` with exact Unicode code-point offsets, cue/sentence references, and an `instance_id` that groups a multi-cue expression, including expressions interrupted by recognized bracketed sound cues. After cue translations are finalized, Gemini aligns each source occurrence to zero, one, or multiple target texts in that cue's translation; Python searches the final translation, calculates exact Unicode offsets, and creates stable `translation_alignment_ids`. Each target span references its source `occurrence_id`, allowing a consumer to synchronize visibility across both languages. Low-confidence, omitted, ambiguous, cross-cue, or unresolvable alignments remain explicit review warnings. `display_cues` retain their original timing and text. Each quiz container includes one `trigger_time`: checkpoint quiz timing is stored on its checkpoint object, and the `final_quiz` timing is the video duration. Individual questions do not carry timing because they are displayed as part of their containing quiz. Culture notes include `trigger_time` from their associated subtitle cues. Quiz questions reference concepts through `concept_ids`. No field chooses or changes playback visibility.

Concepts are discovered in batches, then a separate occurrence-mapping pass sends the stable concept list across all batches. Python locates observed forms, conservative noun/adjective inflections, and minor Unicode/spacing variants, then calculates offsets against unchanged cue text. Frequency is the number of reliably mapped linguistic instances in this subtitle file, not a corpus-frequency estimate. Category uses the taxonomy in [the annotation guide](docs/LINGUISTIC_ANNOTATIONS.md); difficulty uses an approximate, non-certifying A1/A2/B1–B2/C1–C2 rubric. Genuine missing contexts and alternative-form mismatches are reported separately. Cue translations use whole-sentence context and receive alignment warnings when content is absent from the sentence translation. A separate alignment pass only maps already confirmed source occurrences to target text; it does not analyze target-language difficulty or change source annotations. Python searches the final cue translation, assigns offsets, and uses the model's occurrence order to resolve repeated target text when the mapping is sufficiently determined.

This is a breaking output change from schema 1.0.0: `hide_spans` and `hide_eligible` are removed; `vocabulary.json.items` becomes `vocabulary.json.concepts`; quiz `annotation_ids` becomes `concept_ids`; and JSON files declare schema version `2.0.0`.

## Pipeline and reliability

The Agent distinguishes word concepts (`noun`, `verb`, `adjective`, `adverb`) from expressions and grammar patterns. Python publishes at most two word occurrences per semantic sentence, while expressions and grammar patterns are not counted against that cap. AI supplies each concept's `selection_priority` and `selection_reason`; Python applies the cap after exact occurrence mapping and records excluded word candidates as warnings. Word occurrences have no episode-wide ratio cap; overall semantic-sentence annotation coverage has a soft 30% goal and a 60% upper cap. Coverage is measured as semantic sentences with at least one validated occurrence divided by all semantic sentences. If initial coverage is below 30%, Gemini reviews unannotated sentences once; the goal never justifies inventing weak learning items.

Language-specific analysis guidance is selected from the original/source language. Spanish (`es`, including regional variants) and French (`fr`, including regional variants) have separate additions for language-element analysis and occurrence mapping; other languages use the shared guidance. French rules cover useful vocabulary, agreement, tense/aspect, pronoun/clitic patterns, negation, and register, while preserving exact observed accents and elisions for location.

Quiz generation follows the same separation of responsibilities: Gemini returns evidence sentence IDs and text hints for the intended concept, while Python resolves those hints against concepts attached to the cited sentences. Unique matches receive official `concept_id` values; missing or ambiguous matches remain empty and are recorded in `quizzes.quiz_warnings` for human review. Content-comprehension questions may have no concept ID.

Batches default to 50 cues with three cues of contextual overlap. Gemini calls request JSON, pace requests to stay below common free-tier per-minute quotas, and retry transient/invalid responses up to three times (quota errors wait at least 60 seconds). The run state is updated after validated stages, and failed runs retain their state, events, usage summary, and report. Subtitle text is treated as untrusted content. Logs contain counts, errors, model and token metadata when available; they do not include prompts or subtitle bodies. Costs are not estimated because a maintained pricing table is not bundled.

The ZIP is created only after cross-reference validation, independent JSON Schema validation against `schemas/*.schema.json`, and a separate Gemini semantic review. Semantic review findings, occurrence-mapping warnings, cue translation warnings, and cultural review flags are included in `review-report.md`; they do not replace deterministic structural validation. Run reports also summarize published concepts, cue-local occurrences, category/difficulty distributions, repeats, missing/alternative forms, zero-occurrence exclusions, and Gemini usage.

## Tests

```sh
python -m unittest discover -s tests -v
```

Tests use fake Gemini responses for deterministic workflow checks and do not require an API key. The complete-episode acceptance run needs the episode subtitle file and a working Gemini API key.
