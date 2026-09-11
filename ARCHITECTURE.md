# VideoSearch GUI — Architecture

A minimal video editor / search UI: static front-end + a single-file Python
backend (`server.py`), plus a small `models/` package it calls into for
frame extraction, FAISS index building, and relevance feedback. See
[requirements.txt](requirements.txt) for the Python dependencies — the
backend is no longer standard-library-only now that it builds CLIP-based
FAISS datasets and, for Explain, runs a captioning VLM.

## Files

| File | Role |
|---|---|
| [index.html](index.html) | Page structure: player, controls, timeline, search panel, result actions, Explain modal |
| [style.css](style.css) | Dark theme, layout (player top half, search strip below), result-action buttons, modal |
| [app.js](app.js) | All front-end behavior — player, timeline/hover-preview, pins, keyboard shortcuts, search wiring (incl. the locked/"New search" box state), Create FAISS button, Select frames/Feedback/Explain wiring |
| [server.py](server.py) | Serves the static files, stores/reuses uploaded videos under `database/`, exposes `/api/search`, `/api/feedback`, `/api/explain`, `/api/videos`, `/api/extract-frames`, `/api/pins` |
| [models/frame_extractor.py](models/frame_extractor.py) | Reads a video's `*_shot_boundaries_datamodel.json` and saves one `.jpg` per shot-boundary `dimension_idx` into `frames/` next to the video |
| [models/write_faiss_index.py](models/write_faiss_index.py) | Encodes a folder of frames with a VLM (CLIP by default) and writes/loads the resulting FAISS index; also the standalone `write_faiss_index.py` CLI for building an index over an arbitrary image folder |
| [models/search.py](models/search.py) | `search_frames()` embeds a text query and searches a video's FAISS index for the `k` most similar frames (cosine similarity); `feedback_search_frames()` re-runs that search after folding relevance feedback into the query; `explain_negative_feedback()` captions a feedback round's negative frames for the Explain modal |
| [models/relevance_feedback.py](models/relevance_feedback.py) | `RocchioUpdate` (the `alpha*query + beta*positive - gamma*negative` query-update math); `ImageEmbeddingRelevanceFeedback` (averages retrieval-space image embeddings — what `feedback_search_frames` actually uses to update the query); `CaptionVLMRelevanceFeedback` (captions images with a VLM and embeds the captions — used by `explain_negative_feedback` and the checkpoint, not by the live query update); `whole_image_box()` helper for treating a whole image as one fragment; `ImageBasedVLMRelevanceFeedback` (extracts annotated image fragments — defined but not currently wired into any flow) |
| [models/configs.py](models/configs.py), [models/clip.py](models/clip.py), [models/siglip.py](models/siglip.py), [models/llava.py](models/llava.py), [models/vlm_wrapper.py](models/vlm_wrapper.py), [models/utils.py](models/utils.py) | VLM model/processor wrappers (CLIP, SigLIP, LLaVA) that `write_faiss_index.py` and `search.py` pick between via `model_family` |
| [checkpoints/vlm_checkpoint.py](checkpoints/vlm_checkpoint.py), [checkpoints/relevance_feedback_checkpoint.py](checkpoints/relevance_feedback_checkpoint.py) | Standalone, offline-image scripts proving the embedding/captioning/Rocchio pipeline actually works (cosine similarity between matching text/images; relevance feedback closing the gap between images marked Relevant vs Irrelevant) — not part of the running app |
| `requirements.txt` | pip dependencies for `server.py` and the `models/` package (torch, transformers, faiss-cpu, opencv-python-headless, ...) |
| `database/<video name>/<file>` | On-disk reference copy of each opened video (created on first open, reused after) |
| `database/<video name>/<video name>.faiss`, `..._id_map.json` | FAISS dataset built by the Create FAISS button — index + id→path lookup, saved next to the video |
| `database/<video name>/frames/` | Shot-boundary frames extracted from the video, named `frame_<dimension_idx>.jpg` |

## Component overview

```mermaid
graph TD
    subgraph Browser
        HTML[index.html] --> CSS[style.css]
        HTML --> JS[app.js]
    end

    subgraph "Python process (server.py)"
        STATIC["Static file handler\n(index.html / style.css / app.js)"]
        RANGE["Range/206 handler\n(video seeking)"]
        SEARCH["GET /api/search\nhandle_search()"]
        FEEDBACK["POST /api/feedback\nhandle_feedback()"]
        EXPLAIN["POST /api/explain\nhandle_explain()"]
        UPLOAD["POST /api/videos\nfind_existing_video() / save"]
        EXTRACT["POST /api/extract-frames\nhandle_extract_frames()"]
    end

    subgraph "models/"
        FE["frame_extractor.py\nextract_frames_from_video()"]
        WFI["write_faiss_index.py\nbuild_index_for_frames()"]
        SRCH["search.py\nsearch_frames() / feedback_search_frames()\nexplain_negative_feedback()"]
        RFB["relevance_feedback.py\nRocchioUpdate\nImageEmbeddingRelevanceFeedback\nCaptionVLMRelevanceFeedback"]
    end

    DB[("database/<video name>/\nvideo, json, frames/, .faiss")]

    JS -- "GET /" --> STATIC
    JS -- "GET video src (Range)" --> RANGE
    JS -- "GET /api/search?q=&video_name=" --> SEARCH
    JS -- "POST /api/feedback\n(Feedback click)" --> FEEDBACK
    JS -- "POST /api/explain\n(Explain click)" --> EXPLAIN
    JS -- "POST /api/videos" --> UPLOAD
    JS -- "POST /api/extract-frames\n(Create FAISS click)" --> EXTRACT
    RANGE --> DB
    UPLOAD --> DB
    EXTRACT --> FE
    EXTRACT --> WFI
    SEARCH --> SRCH
    FEEDBACK --> SRCH
    EXPLAIN --> SRCH
    SRCH --> RFB
    FE --> DB
    WFI --> DB
    SRCH --> DB
```

## Flow: opening a video

`videoInput` change handler in [app.js](app.js) → `uploadVideo()` → `POST /api/videos` in [server.py](server.py).

```mermaid
sequenceDiagram
    participant U as User
    participant FE as app.js
    participant BE as server.py
    participant FS as database/ (disk)

    U->>FE: picks a file ("Open video")
    FE->>BE: POST /api/videos (header X-Filename, body = file bytes)
    BE->>BE: sanitize_name(filename) -> folder name
    BE->>FS: look for a video file already in database/<name>/
    alt a video file already exists there
        BE-->>FE: { path, reused: true } (existing file kept as-is)
    else nothing there yet
        BE->>FS: mkdir database/<name>/ and write the uploaded bytes
        BE-->>FE: { path, reused: false }
    end
    FE->>FE: video.src = path, captureVideo.src = path
    FE->>BE: GET <path> with Range headers (as the player seeks)
    BE-->>FE: 206 Partial Content
```

`find_existing_video()` only considers files whose extension is a known
video type — this matters because a folder can also hold `.DS_Store`,
subtitles, or files a separate pipeline dropped in (`.h5`, `.json`, etc.);
those must never be picked as "the video".

## Flow: building a FAISS dataset (Create FAISS button)

`createFaissBtn` click handler in [app.js](app.js) → `createFaissIndex()` →
`POST /api/extract-frames` → `handle_extract_frames()` in [server.py](server.py),
which walks through three cases in order — each one short-circuits the rest:

```mermaid
sequenceDiagram
    participant U as User
    participant FE as app.js
    participant BE as server.py
    participant M as models/
    participant FS as database/<video name>/ (disk)

    U->>FE: clicks "Create FAISS"
    FE->>FE: disable button + "Load video", show animated "Building FAISS index..." toast
    FE->>BE: POST /api/extract-frames { video_name }

    alt <video name>.faiss already exists
        BE->>FS: has_faiss_index()
        BE-->>FE: { skipped: true, message: "Dataset already available for this video." }
    else frames/ already has frames
        BE->>FS: has_extracted_frames()
        BE->>M: write_faiss_index.build_index_for_frames(video_path, frames_dir, video_name)
        M->>FS: write <video name>.faiss + <video name>_id_map.json
        BE-->>FE: { output_dir, faiss_path }
    else nothing extracted yet
        BE->>FS: find_shot_boundaries_json() (*_shot_boundaries_datamodel.json)
        alt json missing
            BE-->>FE: 404 { error: "Error: no shot_boudaries_file found." }
        else json found
            BE->>M: frame_extractor.extract_frames_from_video(video_path)
            M->>FS: write frames/frame_<dimension_idx>.jpg
            BE->>M: write_faiss_index.build_index_for_frames(...)
            M->>FS: write <video name>.faiss + <video name>_id_map.json
            BE-->>FE: { output_dir, faiss_path }
        end
    end

    FE->>FE: stop the toast animation, show final message for 2-3s,\nre-enable button + "Load video"
```

Notes:

- The `.faiss` file and its `_id_map.json` are saved **next to the video and
  its shot-boundaries json** (`database/<video name>/`), not inside
  `frames/` — `frames/` only ever holds the extracted `.jpg` images.
- `build_index_for_frames()` loads CLIP (`openai/clip-vit-base-patch32` by
  default, via [models/configs.py](models/configs.py)) and encodes every
  frame; this is the slow step (tens of seconds for ~1,000 frames on CPU),
  which is why the button is disabled and the toast animates until the
  request resolves.
- **Import-order gotcha**: [models/write_faiss_index.py](models/write_faiss_index.py)
  imports `torch` before `faiss`. On macOS, the reverse order corrupts
  torch's thread-pool initialization and segfaults on the first model
  forward pass — this isn't optional style, don't reorder those imports.

## Flow: searching

`searchForm` submit handler in [app.js](app.js) → `performSearch()` →
`GET /api/search` → `handle_search()` in [server.py](server.py) →
`search.search_frames()`, which embeds the query with CLIP and does a
cosine-similarity search over the video's FAISS index:

```mermaid
sequenceDiagram
    participant U as User
    participant FE as app.js
    participant BE as server.py
    participant M as models/search.py
    participant FS as database/<video name>/ (disk)

    U->>FE: types a query, clicks Search (or presses Enter)

    alt query is empty
        FE->>FE: no-op — searchForm submit handler returns early
    else query is non-empty
        FE->>BE: GET /api/search?q=<query>&video_name=<name>&k=<kSlider value>
        alt no video / no <video name>.faiss yet
            BE-->>FE: { results: [], message: "No FAISS dataset..." }
        else dataset exists
            BE->>M: search_frames(video_path, video_name, query, k)
            M->>FS: faiss.read_index(<video name>.faiss) + load <video name>_id_map.json
            M->>M: embed_query() — CLIP text embedding, L2-normalized
            M->>FS: index.search() — top-k by cosine similarity (inner product on normalized vectors)
            M-->>BE: [{ label, path, frame_idx, time, score }, ...]
            BE-->>FE: { results: [{ label, time, score, url }, ...] }
        end
        FE->>FE: renderFrameResults() — one card per match (frame image, time, similarity score)
        FE->>FE: lockSearchBox() — box shows the query read-only,\nbutton becomes "New search"
    end

    U->>FE: clicks a result card
    FE->>FE: seeks the player to that frame's time (unless "Select frames" mode is on — see the feedback flow)

    U->>FE: drags the "Results" slider (1-50, default 5)
    alt a Feedback round is active for this query
        FE->>FE: re-run that feedback round at the new k (see the feedback flow) instead of the plain search
    else
        FE->>FE: reruns the last non-empty search with the new k
    end

    U->>FE: clicks "New search"
    FE->>FE: resetSearchResults() — box clears and re-enables, button reverts to "🔍 Search",\nresults/action row hidden, lastFeedback and explainCache cleared
```

Notes:

- The search box, its button, and the k slider are all disabled until a
  video is loaded, and the slider resets to its default (5) every time a
  new video loads (`resetSearchResults()` in [app.js](app.js)).
- Once a search succeeds, the box locks (`searchLocked = true`) to show
  exactly what was searched; the submit handler branches on that flag, so
  clicking the (now relabeled) button resets instead of searching again.
  This is also what "a completely new search never reuses feedback" means
  in practice: a fresh, unlocked search always hits plain `/api/search` and
  clears any relevance-feedback state.
- `search_frames()` reuses the exact same FAISS index and CLIP model the
  Create FAISS button built — both index and query vectors are L2-normalized,
  so `IndexFlatIP`'s inner product *is* cosine similarity.
- The loaded (model, processor) pair is cached process-wide by
  `models.configs.load_vlm_wrapper()`, keyed by
  `(model_family, model_id, device)`, and shared between `search.py` and
  `write_faiss_index.build_index_for_frames()` — so `from_pretrained()`
  (several seconds) only runs once per process, not on every search. A lock
  around the cache serializes concurrent first-time loads of the same entry
  instead of racing to build it twice; once cached, lookups are effectively
  free and don't block each other.
- A frame's playback time is derived from its filename
  (`frame_<dimension_idx>.jpg`) divided by the video's fps
  (`frame_extractor.get_video_fps()`, read straight from the video file via
  OpenCV) — not from the shot-boundaries json, so search still works even if
  that json is later removed.
- **Threading gotcha**: unlike `write_faiss_index.py` (which only ever calls
  `index.add()`), `search.py` calls `index.search()` — and doing that from a
  `ThreadingHTTPServer` request thread that already ran a torch forward pass
  in the same process segfaults, even with the torch-before-faiss import
  order and `faiss.omp_set_num_threads(1)`. [server.py](server.py) works
  around it by setting `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=TRUE`
  as environment variables *before* importing anything that pulls in
  torch/faiss — that has to happen at the very top of the file, since OpenMP
  reads them at library-load time.

## Flow: relevance feedback (Select frames + Feedback buttons)

`selectFramesBtn`/`feedbackBtn` click handlers in [app.js](app.js) →
`runFeedback()` → `performFeedbackSearch()` → `POST /api/feedback` →
`handle_feedback()` in [server.py](server.py) →
`search.feedback_search_frames()`:

```mermaid
sequenceDiagram
    participant U as User
    participant FE as app.js
    participant BE as server.py
    participant M as models/search.py + relevance_feedback.py
    participant FS as database/<video name>/ (disk)

    U->>FE: clicks "Select frames"
    FE->>FE: selectFramesMode = true — clicking a result card now toggles its selection\ninstead of seeking the player
    U->>FE: clicks one or more result cards
    FE->>FE: selectedFrames.add(card) — "Feedback" enables once at least one is selected

    U->>FE: clicks "Feedback"
    FE->>FE: split the currently displayed labels: selected -> negative_labels,\nthe rest of that same result set -> positive_labels
    FE->>BE: POST /api/feedback { video_name, query, k, positive_labels, negative_labels }
    BE->>M: feedback_search_frames(video_path, video_name, query, positive_labels, negative_labels, k)
    M->>M: embed_query() — CLIP text embedding, L2-normalized
    M->>FS: ImageEmbeddingRelevanceFeedback — open + CLIP-embed each positive/negative frame,\naverage each group (no captioning: kept fast for an interactive click)
    M->>M: RocchioUpdate — updated_query = alpha*query + beta*avg(positive) - gamma*avg(negative), re-normalized
    M->>FS: faiss index.search() with the updated query vector — same index, no rebuild
    M-->>BE: [{ label, path, frame_idx, time, score }, ...]
    BE-->>FE: { query, results: [...] }
    FE->>FE: renderFrameResults() — replaces the grid, resets selection
    FE->>FE: lastFeedback = { positiveLabels, negativeLabels } (kept for the k slider and Explain)\nexplainCache = null (any cached Explain captions are now stale)
```

Notes:

- Feedback is stateless per click: each round recomputes the update from
  the *original* (locked) query text plus whichever frames are currently
  marked — it doesn't compound across successive rounds.
- The search box itself never changes: it keeps showing the original
  locked query throughout, since feedback only changes which frames come
  back, not what's displayed as "the query".
- `/api/feedback` returns 400 if `query` or `negative_labels` is empty —
  the front end already guards this (`feedbackBtn` stays disabled until at
  least one frame is selected), so this is mostly a safety net for direct
  API use.

## Flow: explaining a feedback round (Explain button)

`explainBtn` click handler in [app.js](app.js) → `openExplainModal()` →
`POST /api/explain` → `handle_explain()` in [server.py](server.py) →
`search.explain_negative_feedback()`:

```mermaid
sequenceDiagram
    participant U as User
    participant FE as app.js
    participant BE as server.py
    participant M as models/search.py + relevance_feedback.py
    participant FS as database/<video name>/ (disk)

    U->>FE: clicks "Explain"
    alt captions already cached for this feedback round
        FE->>FE: renderExplainGallery(explainCache.negative) — reopens instantly, no request
    else first Explain click since the last Feedback round
        FE->>FE: show modal with a "Generating captions..." placeholder
        FE->>BE: POST /api/explain { video_name, query, negative_labels: lastFeedback.negativeLabels }
        BE->>M: explain_negative_feedback(video_path, query, negative_labels)
        M->>FS: CaptionVLMRelevanceFeedback — whole_image_box() per frame, captioned by a small\nLLaVA variant (llava-hf/llava-interleave-qwen-0.5b-hf), not the app's default 7B model
        M->>M: truncate each caption to 25 words (the prompt asks for it,\nbut this small model doesn't reliably honor it on its own)
        M-->>BE: [{ label, caption }, ...]
        BE-->>FE: { query, negative: [{ label, url, caption }, ...] }
        FE->>FE: explainCache = { negative } — reused by any Explain click\nuntil the next Feedback round\nrenderExplainGallery() — one card per frame: image + caption below it
    end
```

Notes:

- Purely explanatory: these captions are never fed back into the query —
  the actual update (previous section) is computed from raw CLIP image
  embeddings, not text.
- `explainCache` is invalidated at exactly the points `lastFeedback`
  changes: a new Feedback round, a fresh search, or "New search" — so it
  always reflects exactly one feedback round's negative frames, and
  reopening Explain without a new Feedback round costs nothing.
- The lightweight captioning model is a deliberate choice, not the
  default: loading and running `models/configs.py`'s default 7B LLaVA for
  this took multiple minutes end-to-end on this project's dev hardware
  (CPU/MPS, no CUDA) — the 0.5B interleave variant returns in roughly
  10-15 seconds including a cold model load, which is what makes an
  interactive "click Explain, see captions" flow viable at all.
- `get_device()` in [models/search.py](models/search.py) checks CUDA then
  MPS then falls back to CPU — the MPS check matters on Apple Silicon,
  where omitting it silently forces every VLM call (search, feedback, and
  especially captioning) onto the CPU path instead.

## In-page state (app.js)

- `pins`: `{ id, time, label, thumb }[]` — markers placed on the timeline,
  captured as JPEG thumbnails via a hidden `<video>`/`<canvas>` pair so
  capturing a thumbnail never interrupts the main player.
- `committedTime`: the actual playhead position, kept separate from the
  live hover-preview (`isHovering`) that temporarily scrubs the paused
  player while the mouse moves over the timeline.
- `lastSearchQuery` / `lastSearchK`: the query and k actually sent to the
  backend for the current result set — what the k slider replays on
  change, and what locks into the search box until "New search".
- `searchLocked`: whether the search box is currently showing a submitted
  query read-only (button reads "New search") versus editable (button
  reads "🔍 Search") — toggled by `lockSearchBox()` / `resetSearchResults()`.
- `selectFramesMode` / `selectedFrames`: whether clicking a result card
  toggles its selection (vs. seeking the player), and the set of currently
  selected cards — cleared by `resetSelection()` on every new result set.
- `lastFeedback`: `{ positiveLabels, negativeLabels }` from the last
  applied Feedback round, or `null` if none is active for the current
  query. Also what gates the Explain button and what the k slider replays
  instead of a plain search when set.
- `explainCache`: `{ negative }` — the last `/api/explain` response,
  reused so reopening Explain without a new Feedback round doesn't re-run
  the captioning model. Invalidated everywhere `lastFeedback` is
  reassigned.

## Keyboard shortcuts

| Key | Action |
|---|---|
| `Space` | Play / pause |
| `A` | Add a pin at the current time |

Both are ignored while focus is on a text input (e.g. the search box).

## Known POC boundaries

- Search results aren't tied back to pins/markers — clicking a result seeks
  the player to that frame's time, but no marker is dropped on the timeline
  for it.
- The shot-boundaries json (`*_shot_boundaries_datamodel.json`) is expected
  to already exist next to the video, produced by a separate upstream
  pipeline — `server.py`/`frame_extractor.py` only read it, they don't
  generate it.
- Relevance feedback doesn't compound or persist: each Feedback click
  recomputes the update from the original query text plus whatever's
  currently marked, and nothing survives a fresh search or page reload.
- Explain only covers the frames marked negative in the last Feedback
  round — there's no equivalent view for the positive side, and no
  standalone way to caption an image outside that flow.
- No authentication, no HTTPS — intended for local use only.
