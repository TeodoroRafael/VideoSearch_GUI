// ============================================================================
// VideoSearch GUI — front-end / layout layer.
//
// This file handles: the player, the timeline with hover-preview, pins
// (markers), Create FAISS, and the search box.
//
// performSearch(query) calls GET /api/search?q=...&video_name=... (server.py
// -> models/search.py), which embeds the query with CLIP and searches the
// loaded video's FAISS dataset for the closest shot-boundary frames — see
// ARCHITECTURE.md. An empty query instead just lists the video's markers
// (pins), unrelated to that search.
// ============================================================================

(() => {
  const video        = document.getElementById('videoPlayer');
  const videoInput    = document.getElementById('videoFileInput');
  const videoWrapper    = document.getElementById('videoWrapper');
  const emptyState     = document.getElementById('emptyState');
  const createFaissBtn  = document.getElementById('createFaissBtn');

  const playBtn        = document.getElementById('playBtn');
  const stopBtn         = document.getElementById('stopBtn');
  const addPinBtn        = document.getElementById('addPinBtn');
  const fullscreenBtn      = document.getElementById('fullscreenBtn');
  const currentTimeEl     = document.getElementById('currentTime');
  const durationEl         = document.getElementById('duration');

  const timelineTrack       = document.getElementById('timelineTrack');
  const timelineProgress     = document.getElementById('timelineProgress');
  const timelineHover         = document.getElementById('timelineHover');
  const hoverTimeTooltip       = document.getElementById('hoverTimeTooltip');
  const pinsLayer                = document.getElementById('pinsLayer');

  const searchForm       = document.getElementById('searchForm');
  const searchInput        = document.getElementById('searchInput');
  const searchResults        = document.getElementById('searchResults');

  const captureVideo       = document.getElementById('captureVideo');
  const captureCanvas        = document.getElementById('captureCanvas');

  let pins = [];          // { id, time, label, thumb }
  let pinIdCounter = 1;
  let committedTime = 0;  // actual playhead position (outside of hover-preview)
  let isHovering = false;
  let currentVideoName = null; // database/<currentVideoName>/ for the loaded video

  const clamp = (v, min, max) => Math.min(max, Math.max(min, v));

  function formatTime(seconds) {
    if (!isFinite(seconds)) return '00:00';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  }

  function setControlsEnabled(enabled) {
    playBtn.disabled = !enabled;
    stopBtn.disabled = !enabled;
    addPinBtn.disabled = !enabled;
    fullscreenBtn.disabled = !enabled;
    createFaissBtn.disabled = !enabled;
  }

  // ---------------------------------------------------------------------
  // Load local video
  // ---------------------------------------------------------------------

  videoInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const originalText = emptyState.innerHTML;
    emptyState.textContent = 'Copying video to database...';
    emptyState.hidden = false;

    let reference;
    try {
      reference = await uploadVideo(file);
    } catch (err) {
      emptyState.innerHTML = originalText;
      alert('Could not reach the Python backend (is server.py running on http://localhost:8000 ?)');
      return;
    }

    // BACKEND: reference.path points at database/<video name>/<file>,
    // created (or reused, if it already existed) by POST /api/videos in
    // server.py — this is the on-disk copy future processing should use.
    video.src = reference.path;
    captureVideo.src = reference.path;
    currentVideoName = reference.name;

    emptyState.hidden = true;
    setControlsEnabled(true);

    pins = [];
    renderPins();
    resetSearchResults();

    committedTime = 0;
    playBtn.textContent = '▶';
  });

  async function uploadVideo(file) {
    const res = await fetch('/api/videos', {
      method: 'POST',
      headers: { 'X-Filename': encodeURIComponent(file.name) },
      body: file,
    });
    if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
    return res.json(); // { name, path, reused }
  }

  video.addEventListener('error', () => {
    setControlsEnabled(false);
    emptyState.hidden = false;
    emptyState.textContent = `Could not load "${video.src}" as a video. The reference file in database/ may be wrong or unsupported — open the video again.`;
  });

  video.addEventListener('loadedmetadata', () => {
    durationEl.textContent = formatTime(video.duration);
    updateProgressUI();
  });

  video.addEventListener('timeupdate', () => {
    if (!isHovering) {
      committedTime = video.currentTime;
    }
    updateProgressUI();
  });

  video.addEventListener('play', () => { playBtn.textContent = '⏸'; });
  video.addEventListener('pause', () => { playBtn.textContent = '▶'; });

  function updateProgressUI() {
    currentTimeEl.textContent = formatTime(video.currentTime);
    if (video.duration) {
      const ratio = video.currentTime / video.duration;
      timelineProgress.style.width = `${ratio * 100}%`;
    }
  }

  // ---------------------------------------------------------------------
  // Play / Stop
  // ---------------------------------------------------------------------

  playBtn.addEventListener('click', () => {
    if (video.paused) video.play();
    else video.pause();
  });

  stopBtn.addEventListener('click', () => {
    video.pause();
    video.currentTime = 0;
    committedTime = 0;
    updateProgressUI();
  });

  // ---------------------------------------------------------------------
  // Create FAISS — enabled once a video is loaded
  // ---------------------------------------------------------------------

  createFaissBtn.addEventListener('click', () => {
    createFaissIndex();
  });

  async function createFaissIndex() {
    if (!currentVideoName) return;

    // Block further clicks and video changes until the backend responds —
    // extraction runs synchronously server-side and can take a while.
    createFaissBtn.disabled = true;
    videoInput.disabled = true;
    const processing = showProcessingToast('Building FAISS index');

    try {
      const res = await fetch('/api/extract-frames', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_name: currentVideoName }),
      });
      const data = await res.json();

      if (!res.ok) {
        processing.finish(data.error || 'Error: no shot_boudaries_file found.', 3000);
        return;
      }

      if (data.skipped) {
        processing.finish(data.message || 'Dataset already available for this video.', 3000);
        return;
      }

      processing.finish('FAISS index created.');
    } catch (err) {
      processing.finish('Could not reach the Python backend.', 3000);
    } finally {
      createFaissBtn.disabled = false;
      videoInput.disabled = false;
    }
  }

  function showToast(message, duration = 2000) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    videoWrapper.appendChild(toast);
    if (duration != null) setTimeout(() => toast.remove(), duration);
    return toast;
  }

  // A toast with a spinner and animated "..." that stays until finish() is
  // called — used while an async backend call is in flight.
  function showProcessingToast(message) {
    const toast = showToast(message, null);
    toast.classList.add('processing');

    let dots = 0;
    const interval = setInterval(() => {
      dots = (dots + 1) % 4;
      toast.textContent = message + '.'.repeat(dots);
    }, 400);

    return {
      finish(finalMessage, duration = 2000) {
        clearInterval(interval);
        toast.classList.remove('processing');
        toast.textContent = finalMessage;
        setTimeout(() => toast.remove(), duration);
      },
    };
  }

  // ---------------------------------------------------------------------
  // Keyboard shortcuts — Space: play/pause, A: add pin at current time
  // ---------------------------------------------------------------------

  document.addEventListener('keydown', (e) => {
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || e.target.isContentEditable) return;

    if (e.code === 'Space') {
      e.preventDefault();
      playBtn.click(); // no-op while disabled (no video loaded)
    } else if (e.key === 'a' || e.key === 'A') {
      e.preventDefault();
      addPinBtn.click(); // no-op while disabled (no video loaded)
    }
  });

  // ---------------------------------------------------------------------
  // Fullscreen
  // ---------------------------------------------------------------------

  fullscreenBtn.addEventListener('click', () => {
    if (!document.fullscreenElement) {
      videoWrapper.requestFullscreen?.();
    } else {
      document.exitFullscreen?.();
    }
  });

  document.addEventListener('fullscreenchange', () => {
    const isFullscreen = document.fullscreenElement === videoWrapper;
    fullscreenBtn.textContent = isFullscreen ? '⤡' : '⛶';
    fullscreenBtn.title = isFullscreen ? 'Exit fullscreen' : 'Fullscreen';
  });

  // ---------------------------------------------------------------------
  // Timeline: hover-preview (shows the frame directly in the player) + seek
  // ---------------------------------------------------------------------

  timelineTrack.addEventListener('mousemove', (e) => {
    if (!video.duration) return;

    isHovering = true;
    const rect = timelineTrack.getBoundingClientRect();
    const ratio = clamp((e.clientX - rect.left) / rect.width, 0, 1);
    const hoverTime = ratio * video.duration;

    timelineHover.hidden = false;
    timelineHover.style.left = `${ratio * 100}%`;

    hoverTimeTooltip.hidden = false;
    hoverTimeTooltip.style.left = `${e.clientX - rect.left}px`;
    hoverTimeTooltip.textContent = formatTime(hoverTime);

    // Preview the frame directly in the player, without affecting playback.
    if (video.paused) {
      video.currentTime = hoverTime;
    }
  });

  timelineTrack.addEventListener('mouseleave', () => {
    isHovering = false;
    timelineHover.hidden = true;
    hoverTimeTooltip.hidden = true;

    if (video.paused) {
      video.currentTime = committedTime;
    }
  });

  timelineTrack.addEventListener('click', (e) => {
    if (!video.duration) return;
    const rect = timelineTrack.getBoundingClientRect();
    const ratio = clamp((e.clientX - rect.left) / rect.width, 0, 1);
    const t = ratio * video.duration;

    committedTime = t;
    video.currentTime = t;
  });

  // ---------------------------------------------------------------------
  // Pins (markers)
  // ---------------------------------------------------------------------

  addPinBtn.addEventListener('click', async () => {
    if (!video.duration) return;

    const time = committedTime;
    const thumb = await captureThumbnail(time);

    const pin = {
      id: pinIdCounter++,
      time,
      label: `Marker ${pins.length + 1}`,
      thumb,
    };

    pins.push(pin);
    pins.sort((a, b) => a.time - b.time);
    renderPins();
  });

  function renderPins() {
    pinsLayer.innerHTML = '';
    if (!video.duration) return;

    for (const pin of pins) {
      const el = document.createElement('div');
      el.className = 'pin';
      el.style.left = `${(pin.time / video.duration) * 100}%`;
      el.title = `${pin.label} — ${formatTime(pin.time)} (double-click to rename, right-click to remove)`;
      el.dataset.pinId = pin.id;

      el.addEventListener('click', (e) => {
        e.stopPropagation();
        committedTime = pin.time;
        video.currentTime = pin.time;
      });

      el.addEventListener('dblclick', (e) => {
        e.stopPropagation();
        const newLabel = prompt('Marker name:', pin.label);
        if (newLabel && newLabel.trim()) {
          pin.label = newLabel.trim();
          renderPins();
        }
      });

      el.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        e.stopPropagation();
        pins = pins.filter((p) => p.id !== pin.id);
        renderPins();
      });

      pinsLayer.appendChild(el);
    }
  }

  function highlightPin(pinId) {
    const el = pinsLayer.querySelector(`[data-pin-id="${pinId}"]`);
    if (!el) return;
    el.classList.remove('highlight');
    // force reflow so the animation restarts even if already highlighted
    void el.offsetWidth;
    el.classList.add('highlight');
  }

  // Captures a video frame at a given time using a hidden <video>/<canvas>
  // pair, so the main player's playback is never interrupted.
  function captureThumbnail(time) {
    return new Promise((resolve) => {
      const cv = captureVideo;

      const finish = () => {
        cv.removeEventListener('seeked', finish);
        clearTimeout(fallback);
        captureCanvas.width = 160;
        captureCanvas.height = 90;
        const ctx = captureCanvas.getContext('2d');
        try {
          ctx.drawImage(cv, 0, 0, 160, 90);
          resolve(captureCanvas.toDataURL('image/jpeg', 0.7));
        } catch (err) {
          resolve(null);
        }
      };

      // fallback in case 'seeked' never fires (e.g. time equals current time)
      const fallback = setTimeout(finish, 300);

      cv.addEventListener('seeked', finish);
      cv.currentTime = time;
    });
  }

  // ---------------------------------------------------------------------
  // Search (layout) — results grid linked to the pins
  // ---------------------------------------------------------------------

  searchForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const query = searchInput.value.trim();
    performSearch(query);
  });

  async function performSearch(query) {
    // Empty query: fall back to just listing the video's markers, same as
    // before this searched a real FAISS dataset.
    if (!query) {
      renderPinResults(pins.slice(), query);
      return;
    }

    if (!currentVideoName) {
      renderSearchMessage('Load a video first.');
      return;
    }

    renderSearchMessage('Searching...');

    let data;
    try {
      const res = await fetch(
        `/api/search?q=${encodeURIComponent(query)}&video_name=${encodeURIComponent(currentVideoName)}`
      );
      data = await res.json();
    } catch (err) {
      renderSearchMessage('Could not reach the Python backend.');
      return;
    }

    if (data.message) {
      renderSearchMessage(data.message);
      return;
    }

    renderFrameResults(data.results || [], query);
  }

  function renderSearchMessage(message) {
    searchResults.innerHTML = '';
    const p = document.createElement('p');
    p.className = 'results-placeholder';
    p.textContent = message;
    searchResults.appendChild(p);
  }

  function renderPinResults(list, query) {
    searchResults.innerHTML = '';

    if (!pins.length) {
      renderSearchMessage('Add markers to the video ("📌 Mark" button) so you can search them here.');
      return;
    }

    if (!list.length) {
      renderSearchMessage(`No results for "${query}".`);
      return;
    }

    for (const pin of list) {
      const card = buildResultCard({ thumb: pin.thumb || '', label: pin.label, caption: formatTime(pin.time) });
      card.addEventListener('click', () => {
        committedTime = pin.time;
        video.currentTime = pin.time;
        highlightPin(pin.id);
      });
      searchResults.appendChild(card);
    }
  }

  // results: [{ label, time, score, url }], as returned by
  // GET /api/search?q=...&video_name=... (models/search.py's FAISS
  // cosine-similarity search over the video's shot-boundary frames).
  function renderFrameResults(results, query) {
    searchResults.innerHTML = '';

    if (!results.length) {
      renderSearchMessage(`No results for "${query}".`);
      return;
    }

    for (const result of results) {
      const card = buildResultCard({
        thumb: result.url,
        label: result.time != null ? formatTime(result.time) : result.label,
        caption: `similarity ${result.score.toFixed(3)}`,
      });
      card.addEventListener('click', () => {
        if (result.time == null) return;
        committedTime = result.time;
        video.currentTime = result.time;
      });
      searchResults.appendChild(card);
    }
  }

  function buildResultCard({ thumb, label, caption }) {
    const card = document.createElement('div');
    card.className = 'result-card';

    const img = document.createElement('img');
    img.src = thumb;
    img.alt = label;

    const meta = document.createElement('div');
    meta.className = 'result-meta';

    const labelEl = document.createElement('div');
    labelEl.className = 'result-label';
    labelEl.textContent = label;

    const captionEl = document.createElement('div');
    captionEl.className = 'result-time';
    captionEl.textContent = caption;

    meta.appendChild(labelEl);
    meta.appendChild(captionEl);
    card.appendChild(img);
    card.appendChild(meta);

    return card;
  }

  function resetSearchResults() {
    searchInput.value = '';
    searchResults.innerHTML = '';
    const p = document.createElement('p');
    p.className = 'results-placeholder';
    p.textContent = "Search results will appear here as frames linked to the video's markers.";
    searchResults.appendChild(p);
  }
})();
