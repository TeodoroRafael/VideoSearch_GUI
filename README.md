# VideoSearch GUI

A video search tool built around **VisualReF**, designed to speed up video
editing: instead of scrubbing through raw footage by hand, you search for a
moment and jump straight to the matching frames, each tied to a marker on
the timeline. A relevance-feedback loop lets you refine a search by pointing
out which of its results missed the mark, and an Explain view shows what
the model saw in those results.

This repository currently holds the interface — player, timeline, markers,
the search panel, and the relevance-feedback loop — with clearly marked
hooks for wiring in the real VisualReF-based search backend.

## Interface

![VideoSearch GUI overview](docs/gui-overview.svg)

*(diagram of the interface layout — not a live screenshot)*

| # | Element | What it does |
|---|---|---|
| 1 | **Load video** | Picks a local video file. It's uploaded to the Python backend, which stores a reference copy under `database/<video name>/` (or reuses it if that video was opened before) — see [ARCHITECTURE.md](ARCHITECTURE.md). |
| 2 | **Player** | Shows the loaded video. Fills the top half of the window (`object-fit: contain`, so the aspect ratio is preserved). |
| 3 | **Play / Pause** | Toggles playback. Also bound to the `Space` key. |
| 4 | **Stop** | Pauses and resets playback to `00:00`. |
| 5 | **Time display** | Current position / total duration. |
| 6 | **Mark** | Drops a pin at the current playback time. Also bound to the `A` key. |
| 7 | **Fullscreen** | Expands the player to fill the screen. |
| 8 | **Timeline / track** | Hovering scrubs a live preview directly in the player (frame at that point, without affecting playback); clicking seeks to that position. |
| 9 | **Pins (markers)** | One per marked moment. Click to jump to it, double-click to rename, right-click to remove. Persisted to `database/<video name>/<video name>_pins.json` as you go, so they're still there next time you load that video. |
| 10 | **Search box** | Query field + search button, disabled until a video is loaded. Submitting an empty query does nothing; a non-empty one runs a real CLIP-based similarity search over the video's FAISS dataset (built by Create FAISS) and returns its k closest frames — see the Results slider. Once a search runs, the box shows that query read-only and the button becomes **New search**; clicking it clears the box back to editable and drops the current results, starting fresh. |
| 11 | **Search results** | The matching frames, laid out as a horizontally scrolling strip below the player, each showing its timestamp and similarity score. Clicking a result seeks the player to that frame's time. Having results also reveals the action row (14) below. |
| 12 | **Create FAISS** | Builds a searchable FAISS dataset for the loaded video — extracting shot-boundary frames and encoding them with CLIP — and reuses whatever's already built rather than redoing it. See [ARCHITECTURE.md](ARCHITECTURE.md) for the exact steps it follows and where each file ends up. |
| 13 | **Results slider** | Sets k, how many frames a search returns — 1 to 50, default 5. Disabled until a video is loaded, and reset to 5 every time a new one is. If a search was already made, releasing the slider on a different value reruns it with the new k — replaying the last Feedback round at that k instead of the plain search, if one is active. |
| 14 | **Select frames / Feedback / Explain** | The relevance-feedback loop. **Select frames** toggles a mode where clicking result cards marks them; **Feedback** (enabled once at least one is marked) treats those as *negative* signal and the rest of that result set as *positive*, folds their average image embeddings into the query via Rocchio's algorithm, and replaces the results with a fresh search on the updated query. **Explain** (enabled once a Feedback round has run) opens a modal captioning each frame marked negative, so you can see what the model read in the images it moved away from — captioned once per Feedback round and cached, not regenerated on every open. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full request flow. |

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 server.py
```

Then open `http://localhost:8000`. The backend needs the packages in
[requirements.txt](requirements.txt) (torch, transformers, faiss-cpu,
opencv-python-headless, ...) — Create FAISS and the search box both use them
(extracting frames, encoding them with CLIP, and, for search, embedding the
query and running a similarity search over the resulting index). The first
time either runs, it downloads the CLIP model from Hugging Face unless
already cached locally. Search only returns results once Create FAISS has
been run for the loaded video. Clicking **Explain** additionally downloads a
small captioning model (`llava-hf/llava-interleave-qwen-0.5b-hf`) the first
time it's used.

## Architecture

File-by-file breakdown, request/response flow for opening a video, building
a video's FAISS dataset, searching it, running relevance feedback, and
explaining a feedback round, in-page state, and what's still left as a POC
placeholder: see [ARCHITECTURE.md](ARCHITECTURE.md).
