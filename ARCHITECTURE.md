# VideoSearch GUI — Architecture

A minimal video editor / search UI: static front-end + a single-file Python
backend (`server.py`), plus a small `models/` package it calls into for
frame extraction and FAISS index building. See [requirements.txt](requirements.txt)
for the Python dependencies — the backend is no longer standard-library-only
now that it builds CLIP-based FAISS datasets.

## Files

| File | Role |
|---|---|
| [index.html](index.html) | Page structure: player, controls, timeline, search panel |
| [style.css](style.css) | Dark theme, layout (player top half, search strip below) |
| [app.js](app.js) | All front-end behavior — player, timeline/hover-preview, pins, keyboard shortcuts, search wiring, Create FAISS button |
| [server.py](server.py) | Serves the static files, stores/reuses uploaded videos under `database/`, exposes `/api/search`, `/api/videos`, `/api/extract-frames` |
| [models/frame_extractor.py](models/frame_extractor.py) | Reads a video's `*_shot_boundaries_datamodel.json` and saves one `.jpg` per shot-boundary `dimension_idx` into `frames/` next to the video |
| [models/write_faiss_index.py](models/write_faiss_index.py) | Encodes a folder of frames with a VLM (CLIP by default) and writes/loads the resulting FAISS index; also the standalone `write_faiss_index.py` CLI for building an index over an arbitrary image folder |
| [models/configs.py](models/configs.py), [models/clip.py](models/clip.py), [models/siglip.py](models/siglip.py), [models/llava.py](models/llava.py), [models/vlm_wrapper.py](models/vlm_wrapper.py), [models/utils.py](models/utils.py) | VLM model/processor wrappers (CLIP, SigLIP, LLaVA) that `write_faiss_index.py` picks between via `model_family` |
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
        SEARCH["GET /api/search\nprocess_query()"]
        UPLOAD["POST /api/videos\nfind_existing_video() / save"]
        EXTRACT["POST /api/extract-frames\nhandle_extract_frames()"]
    end

    subgraph "models/"
        FE["frame_extractor.py\nextract_frames_from_video()"]
        WFI["write_faiss_index.py\nbuild_index_for_frames()"]
    end

    DB[("database/<video name>/\nvideo, json, frames/, .faiss")]

    JS -- "GET /" --> STATIC
    JS -- "GET video src (Range)" --> RANGE
    JS -- "GET /api/search?q=" --> SEARCH
    JS -- "POST /api/videos" --> UPLOAD
    JS -- "POST /api/extract-frames\n(Create FAISS click)" --> EXTRACT
    RANGE --> DB
    UPLOAD --> DB
    EXTRACT --> FE
    EXTRACT --> WFI
    FE --> DB
    WFI --> DB
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

```mermaid
sequenceDiagram
    participant U as User
    participant FE as app.js
    participant BE as server.py

    U->>FE: types a query, clicks Search (or presses Enter)
    FE->>BE: GET /api/search?q=<query>
    BE->>BE: process_query(query) — POC placeholder
    BE-->>FE: { query, result }
    FE->>FE: renderResults() — shows the backend echo + pins whose\nlabel matches the query (local fallback)
    U->>FE: clicks a result card
    FE->>FE: seeks the player to that pin's time
```

`process_query()` in [server.py](server.py) and the local label filter in
`performSearch()` in [app.js](app.js) are both placeholders — the intended
integration point for real search logic (embeddings, a model, whatever
ends up finding frames for a query). Search results are expected to come
back as frames, each tied to a pin (`{ pinId, time, label, thumbnailUrl }`).

The per-video FAISS index built by the Create FAISS button (see above) is
the dataset this real search is meant to query — encode the text query with
the same CLIP model, `faiss.read_index()` the video's `.faiss` file, and
search it — but `process_query()` doesn't do that yet.

## In-page state (app.js)

- `pins`: `{ id, time, label, thumb }[]` — markers placed on the timeline,
  captured as JPEG thumbnails via a hidden `<video>`/`<canvas>` pair so
  capturing a thumbnail never interrupts the main player.
- `committedTime`: the actual playhead position, kept separate from the
  live hover-preview (`isHovering`) that temporarily scrubs the paused
  player while the mouse moves over the timeline.

## Keyboard shortcuts

| Key | Action |
|---|---|
| `Space` | Play / pause |
| `A` | Add a pin at the current time |

Both are ignored while focus is on a text input (e.g. the search box).

## Known POC boundaries

- `process_query()` (server.py) — swap for real search logic that queries
  the FAISS dataset the Create FAISS button already builds.
- `performSearch()` (app.js) — swap the local pin-label filter for the
  frames the real search endpoint returns.
- The shot-boundaries json (`*_shot_boundaries_datamodel.json`) is expected
  to already exist next to the video, produced by a separate upstream
  pipeline — `server.py`/`frame_extractor.py` only read it, they don't
  generate it.
- No authentication, no HTTPS — intended for local use only.
