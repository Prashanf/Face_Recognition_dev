/**
 * Face Detection & Recognition Frontend Application
 *
 * Implements:
 * 1. Reference folder upload & ArcFace 512-d registration
 * 2. Group photo upload, detection & canvas visualization
 * 3. Face recognition with greedy 1-to-1 matching and results breakdown
 * 4. Real-time threshold re-matching via slider
 * 5. Click-to-select interaction between canvas boxes and result cards
 *    (no hover effects, so the photo can be inspected freely)
 * 6. High-resolution face thumbnail generation
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements - Step 1 Reference Folder
  const folderInput = document.getElementById("folderInput");
  const folderBtn = document.getElementById("folderBtn");
  const folderProgressBox = document.getElementById("folderProgressBox");
  const folderProgressBar = document.getElementById("folderProgressBar");
  const folderProgressText = document.getElementById("folderProgressText");
  const reuploadFolderLink = document.getElementById("reuploadFolderLink");
  const warningsWrapper = document.getElementById("warningsWrapper");
  const warningsToggleBtn = document.getElementById("warningsToggleBtn");
  const warningsCount = document.getElementById("warningsCount");
  const warningsList = document.getElementById("warningsList");

  // DOM Elements - Step 2 Group Photo
  const groupPhotoInput = document.getElementById("groupPhotoInput");
  const groupPhotoBtn = document.getElementById("groupPhotoBtn");
  const groupDropTarget = document.getElementById("groupDropTarget");
  const changePhotoLink = document.getElementById("changePhotoLink");
  const viewerCard = document.getElementById("viewerCard");
  const emptyState = document.getElementById("emptyState");
  const photoStage = document.getElementById("photoStage");
  const displayImage = document.getElementById("displayImage");
  const fullResImage = document.getElementById("fullResImage");
  const overlayCanvas = document.getElementById("overlayCanvas");
  const photoMeta = document.getElementById("photoMeta");
  const legend = document.getElementById("legend");

  // DOM Elements - Selection bar
  const selectionEmpty = document.getElementById("selectionEmpty");
  const selectionInfo = document.getElementById("selectionInfo");
  const selectionThumb = document.getElementById("selectionThumb");
  const selectionName = document.getElementById("selectionName");
  const selectionSub = document.getElementById("selectionSub");
  const selectionClear = document.getElementById("selectionClear");

  // DOM Elements - Step 3 Recognition & Results
  const recognizeBtn = document.getElementById("recognizeBtn");
  const thresholdSlider = document.getElementById("thresholdSlider");
  const thresholdValue = document.getElementById("thresholdValue");
  const resultsPanel = document.getElementById("resultsPanel");
  const summaryTitle = document.getElementById("summaryTitle");
  const statMatched = document.getElementById("statMatched");
  const statUnknown = document.getElementById("statUnknown");
  const statTotal = document.getElementById("statTotal");
  const proportionMatched = document.getElementById("proportionMatched");
  const proportionUnknown = document.getElementById("proportionUnknown");
  const resultsList = document.getElementById("resultsList");

  // Inline icons for button labels
  const SVG_ATTRS = 'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"';
  const ICONS = {
    folder: `<svg class="btn-icon" ${SVG_ATTRS}><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path></svg>`,
    camera: `<svg class="btn-icon" ${SVG_ATTRS}><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>`,
    check: `<svg class="btn-icon" ${SVG_ATTRS}><path d="M20 6L9 17l-5-5"></path></svg>`,
    search: `<svg class="btn-icon" ${SVG_ATTRS}><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>`,
    refresh: `<svg class="btn-icon" ${SVG_ATTRS}><polyline points="23 4 23 10 17 10"></polyline><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path></svg>`,
  };
  const SPINNER = '<span class="spinner"></span>';

  // Face colors (shared by canvas, cards and selection bar)
  const COLOR_DETECTED = "#3B82F6";
  const COLOR_MATCHED = "#10B981";
  const COLOR_UNKNOWN = "#F59E0B";
  const MAX_UPSCALE = 3; // low-res photos are enlarged at most this much

  // Application State
  let referenceFolderReady = false;
  let groupPhotoReady = false;
  let isRecognized = false;
  let currentGroupFile = null;
  let currentFacesData = null; // Holds the most recent response from /api/detect or /api/recognize
  let selectedFaceId = null;   // Set by clicking a box on the photo or a card in the list

  // =========================================================================
  // 1. Step 1: Reference Folder Upload & Registration
  // =========================================================================

  folderBtn.addEventListener("click", () => folderInput.click());
  reuploadFolderLink.addEventListener("click", () => folderInput.click());

  folderInput.addEventListener("change", async (e) => {
    const rawFiles = Array.from(e.target.files || []);
    if (!rawFiles.length) return;

    // Filter to valid image files
    const validExtensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp"];
    const imageFiles = rawFiles.filter((f) => {
      const ext = f.name.substring(f.name.lastIndexOf(".")).toLowerCase();
      return validExtensions.includes(ext);
    });

    if (!imageFiles.length) {
      alert("No supported images (JPG, PNG, WebP) found in selected folder.");
      return;
    }

    // Update UI to Processing State
    folderBtn.disabled = true;
    folderBtn.innerHTML = `${SPINNER} Processing reference folder...`;
    folderProgressBox.style.display = "block";
    folderProgressBar.style.width = "40%";
    folderProgressText.textContent = `Uploading & processing ${imageFiles.length} photos...`;
    warningsWrapper.style.display = "none";
    warningsList.style.display = "none";

    const formData = new FormData();
    imageFiles.forEach((file) => {
      formData.append("files", file);
    });

    try {
      const response = await fetch("/api/register-faces", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const data = await response.json();
      folderProgressBar.style.width = "100%";
      folderProgressText.textContent = `Completed registration: ${data.registered} faces registered`;

      setTimeout(() => {
        folderProgressBox.style.display = "none";
      }, 1000);

      // Success State UI
      folderBtn.disabled = false;
      folderBtn.className = "btn btn-state-success";
      folderBtn.innerHTML = `${ICONS.check} ${data.registered}/${data.total_files} faces registered`;
      reuploadFolderLink.style.display = "inline-block";
      referenceFolderReady = true;

      // Handle Warnings
      if (data.warnings && data.warnings.length > 0) {
        warningsCount.textContent = data.warnings.length;
        warningsList.innerHTML = "";
        data.warnings.forEach((w) => {
          const li = document.createElement("li");
          li.textContent = `${w.file}: ${w.issue}`;
          warningsList.appendChild(li);
        });
        warningsWrapper.style.display = "block";
      }

      checkReadyToRecognize();

      // If group photo was already recognized, re-run recognition with new reference faces
      if (isRecognized && groupPhotoReady) {
        triggerRecognition();
      }
    } catch (err) {
      console.error("Folder registration error:", err);
      folderBtn.disabled = false;
      folderBtn.className = "btn btn-primary";
      folderBtn.innerHTML = `${ICONS.folder} Select folder`;
      folderProgressBox.style.display = "none";
      alert(`Registration failed: ${err.message}`);
    }
  });

  // Toggle warnings list dropdown
  warningsToggleBtn.addEventListener("click", () => {
    const isHidden = warningsList.style.display === "none";
    warningsList.style.display = isHidden ? "block" : "none";
    warningsToggleBtn.querySelector(".arrow-icon").textContent = isHidden ? "▴" : "▾";
  });

  // =========================================================================
  // 2. Step 2: Group Photo Upload & Initial Detection
  // =========================================================================

  groupPhotoBtn.addEventListener("click", () => groupPhotoInput.click());
  changePhotoLink.addEventListener("click", () => groupPhotoInput.click());
  emptyState.addEventListener("click", () => groupPhotoInput.click());

  // Drag and Drop Handling (button area + whole viewer)
  const dropZones = [
    { el: groupDropTarget },
    { el: viewerCard },
  ];

  dropZones.forEach(({ el }) => {
    ["dragenter", "dragover"].forEach((eventName) => {
      el.addEventListener(eventName, (e) => {
        e.preventDefault();
        el.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      el.addEventListener(eventName, (e) => {
        e.preventDefault();
        el.classList.remove("dragover");
      });
    });

    el.addEventListener("drop", (e) => {
      const files = e.dataTransfer.files;
      if (files && files[0]) {
        handleGroupPhotoSelection(files[0]);
      }
    });
  });

  groupPhotoInput.addEventListener("change", (e) => {
    const files = e.target.files;
    if (files && files[0]) {
      handleGroupPhotoSelection(files[0]);
    }
  });

  function handleGroupPhotoSelection(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please select a valid image file (JPG, PNG, WebP).");
      return;
    }

    currentGroupFile = file;
    isRecognized = false;
    currentFacesData = null;
    selectedFaceId = null;
    resultsPanel.style.display = "none";
    legend.style.display = "none";
    photoMeta.textContent = file.name;
    updateSelectionBar();

    const reader = new FileReader();
    reader.onload = (loadEvent) => {
      const dataUrl = loadEvent.target.result;
      displayImage.src = dataUrl;
      fullResImage.src = dataUrl;
    };
    reader.readAsDataURL(file);
  }

  // Display Image Loaded Callback
  displayImage.addEventListener("load", async () => {
    emptyState.style.display = "none";
    photoStage.style.display = "inline-block";
    fitImageToStage();
    syncCanvasSize();

    // Run initial face detection
    groupPhotoBtn.disabled = true;
    groupPhotoBtn.className = "btn btn-indigo";
    groupPhotoBtn.innerHTML = `${SPINNER} Detecting faces...`;

    const formData = new FormData();
    formData.append("file", currentGroupFile);

    try {
      const response = await fetch("/api/detect", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const data = await response.json();
      currentFacesData = data;
      groupPhotoReady = true;

      // Update Group Photo Button UI
      groupPhotoBtn.disabled = false;
      groupPhotoBtn.className = "btn btn-state-success";
      groupPhotoBtn.innerHTML = `${ICONS.check} Photo uploaded — ${data.total} faces detected`;
      changePhotoLink.style.display = "inline-block";
      photoMeta.textContent = `${currentGroupFile.name} · ${data.total} faces`;
      updateLegend();

      // Draw detection bounding boxes
      drawCanvas();

      checkReadyToRecognize();

      // If reference folder is already registered, automatically perform recognition for convenience!
      if (referenceFolderReady) {
        triggerRecognition();
      }
    } catch (err) {
      console.error("Initial detection error:", err);
      groupPhotoBtn.disabled = false;
      groupPhotoBtn.className = "btn btn-indigo";
      groupPhotoBtn.innerHTML = `${ICONS.camera} Upload photo`;
      alert(`Detection failed: ${err.message}`);
    }
  });

  function checkReadyToRecognize() {
    if (referenceFolderReady && groupPhotoReady) {
      recognizeBtn.disabled = false;
    } else {
      recognizeBtn.disabled = true;
    }
  }

  // =========================================================================
  // 3. Step 3: Face Recognition & Greedy Matching
  // =========================================================================

  recognizeBtn.addEventListener("click", () => {
    triggerRecognition();
  });

  async function triggerRecognition() {
    if (!referenceFolderReady || !groupPhotoReady) return;

    recognizeBtn.disabled = true;
    recognizeBtn.innerHTML = `${SPINNER} Recognizing faces...`;

    const currentThreshold = parseFloat(thresholdSlider.value);
    const formData = new FormData();
    if (currentGroupFile) {
      formData.append("file", currentGroupFile);
    }
    formData.append("threshold", currentThreshold);

    try {
      const response = await fetch("/api/recognize", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const data = await response.json();
      currentFacesData = data;
      isRecognized = true;

      // Update Recognize Button UI
      recognizeBtn.disabled = false;
      recognizeBtn.innerHTML = `${ICONS.refresh} Re-run recognition`;

      applyResults(data);
    } catch (err) {
      console.error("Recognition error:", err);
      recognizeBtn.disabled = false;
      recognizeBtn.innerHTML = `${ICONS.search} Recognize faces`;
      alert(`Recognition failed: ${err.message}`);
    }
  }

  /** Redraws everything that depends on a fresh recognition / re-match result. */
  function applyResults(data) {
    // Drop the selection if that face no longer exists
    if (selectedFaceId !== null && !data.faces.some((f) => f.id === selectedFaceId)) {
      selectedFaceId = null;
    }
    updateLegend();
    drawCanvas();
    renderResultsPanel(data);
    syncCardSelection(false);
    updateSelectionBar();
  }

  // =========================================================================
  // 4. Threshold Slider & Dynamic Re-matching
  // =========================================================================

  thresholdSlider.addEventListener("input", debounce(async (e) => {
    const threshold = parseFloat(e.target.value);
    thresholdValue.textContent = threshold.toFixed(2);

    if (!isRecognized || !groupPhotoReady) return;

    try {
      const response = await fetch("/api/re-match", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ threshold }),
      });

      if (!response.ok) return;

      const data = await response.json();
      currentFacesData = data;

      applyResults(data);
    } catch (err) {
      console.error("Re-match error:", err);
    }
  }, 250));

  // Keep the label in sync while dragging, before the debounced request fires
  thresholdSlider.addEventListener("input", (e) => {
    thresholdValue.textContent = parseFloat(e.target.value).toFixed(2);
  });

  // =========================================================================
  // 5. Canvas Drawing (Detection vs Recognition + Selection spotlight)
  // =========================================================================

  /**
   * Scales the photo to fill the stage (small photos are enlarged, big ones
   * shrunk) while keeping its aspect ratio, so the canvas overlay matches it.
   */
  function fitImageToStage() {
    if (!displayImage.naturalWidth) return;
    const pad = 32; // viewer card padding on both sides
    const stacked = window.matchMedia("(max-width: 1000px)").matches;
    const availW = viewerCard.clientWidth - pad;
    const availH = stacked ? window.innerHeight * 0.7 : viewerCard.clientHeight - pad;
    if (availW <= 0 || availH <= 0) return;

    const scale = Math.min(availW / displayImage.naturalWidth, availH / displayImage.naturalHeight, MAX_UPSCALE);
    const w = Math.round(displayImage.naturalWidth * scale);
    const h = Math.round(displayImage.naturalHeight * scale);
    if (displayImage.style.width !== `${w}px` || displayImage.style.height !== `${h}px`) {
      displayImage.style.width = `${w}px`;
      displayImage.style.height = `${h}px`;
    }
  }

  function syncCanvasSize() {
    const w = displayImage.clientWidth;
    const h = displayImage.clientHeight;
    if (!w || !h) return false;

    // Back the canvas with device pixels so boxes and labels stay crisp
    const dpr = window.devicePixelRatio || 1;
    const pw = Math.round(w * dpr);
    const ph = Math.round(h * dpr);
    if (overlayCanvas.width !== pw || overlayCanvas.height !== ph) {
      overlayCanvas.width = pw;
      overlayCanvas.height = ph;
    }
    return true;
  }

  function faceColor(face) {
    if (!isRecognized) return COLOR_DETECTED;
    return face.is_unknown ? COLOR_UNKNOWN : COLOR_MATCHED;
  }

  function faceLabel(face) {
    if (!isRecognized) return `Face ${face.id}`;
    return face.is_unknown ? "Unknown" : face.name;
  }

  /** Face box in canvas CSS pixels. */
  function faceRect(face, scaleX, scaleY) {
    const x1 = face.bbox[0] * scaleX;
    const y1 = face.bbox[1] * scaleY;
    const x2 = face.bbox[2] * scaleX;
    const y2 = face.bbox[3] * scaleY;
    return { x1, y1, w: x2 - x1, h: y2 - y1 };
  }

  function fitLabel(ctx, label, maxWidth) {
    if (ctx.measureText(label).width <= maxWidth) return label;
    let t = label;
    while (t.length > 1 && ctx.measureText(`${t}\u2026`).width > maxWidth) t = t.slice(0, -1);
    return `${t}\u2026`;
  }

  function pathRoundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(x, y, w, h, r);
    } else {
      ctx.rect(x, y, w, h);
    }
  }

  function drawCanvas() {
    if (!currentFacesData || !currentFacesData.faces) return;
    if (!syncCanvasSize()) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = overlayCanvas.width / dpr;
    const cssH = overlayCanvas.height / dpr;

    const ctx = overlayCanvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    const scaleX = cssW / currentFacesData.image_width;
    const scaleY = cssH / currentFacesData.image_height;
    const faces = currentFacesData.faces;
    const selected = faces.find((f) => f.id === selectedFaceId) || null;

    // Spotlight: dim the whole photo except the selected face
    if (selected) {
      const r = faceRect(selected, scaleX, scaleY);
      ctx.save();
      ctx.fillStyle = "rgba(8, 12, 24, 0.6)";
      ctx.beginPath();
      ctx.rect(0, 0, cssW, cssH);
      if (ctx.roundRect) {
        ctx.roundRect(r.x1, r.y1, r.w, r.h, 6);
      } else {
        ctx.rect(r.x1, r.y1, r.w, r.h);
      }
      ctx.fill("evenodd");
      ctx.restore();
    }

    // Draw unselected faces first so the selected one always sits on top
    const ordered = selected ? [...faces.filter((f) => f !== selected), selected] : faces;

    ordered.forEach((face) => {
      const isSelected = face === selected;
      const color = faceColor(face);
      const label = faceLabel(face);
      const r = faceRect(face, scaleX, scaleY);

      ctx.save();
      ctx.globalAlpha = selected && !isSelected ? 0.7 : 1;

      // Bounding box (selected gets a glow)
      if (isSelected) {
        ctx.shadowColor = color;
        ctx.shadowBlur = 14;
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = isSelected ? 3.5 : 2.5;
      pathRoundRect(ctx, r.x1, r.y1, r.w, r.h, 6);
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Label badge
      ctx.font = `600 ${isSelected ? 13 : 12}px system-ui, -apple-system, sans-serif`;
      const padX = 8;
      const badgeH = 22;
      // Unselected labels are trimmed to roughly the box width so crowded
      // photos stay readable; the selected face always shows its full name.
      const text = isSelected ? label : fitLabel(ctx, label, Math.max(r.w + 24, 84) - padX * 2);
      const badgeW = ctx.measureText(text).width + padX * 2;

      const badgeX = Math.max(0, Math.min(r.x1, cssW - badgeW));
      const badgeY = r.y1 - badgeH - 3 >= 0 ? r.y1 - badgeH - 3 : r.y1 + 3;

      ctx.fillStyle = color;
      pathRoundRect(ctx, badgeX, badgeY, badgeW, badgeH, 6);
      ctx.fill();

      ctx.fillStyle = "#FFFFFF";
      ctx.textBaseline = "middle";
      ctx.fillText(text, badgeX + padX, badgeY + badgeH / 2 + 0.5);
      ctx.restore();
    });
  }

  // =========================================================================
  // 6. Click-to-select (photo <-> list). No hover effects on the photo.
  // =========================================================================

  /** Smallest face box containing the point, so overlapping faces stay clickable. */
  function faceAtPoint(clientX, clientY) {
    if (!currentFacesData || !currentFacesData.faces) return null;

    const rect = overlayCanvas.getBoundingClientRect();
    const px = clientX - rect.left;
    const py = clientY - rect.top;
    const scaleX = rect.width / currentFacesData.image_width;
    const scaleY = rect.height / currentFacesData.image_height;

    let best = null;
    let bestArea = Infinity;
    for (const face of currentFacesData.faces) {
      const r = faceRect(face, scaleX, scaleY);
      if (px >= r.x1 && px <= r.x1 + r.w && py >= r.y1 && py <= r.y1 + r.h) {
        const area = r.w * r.h;
        if (area < bestArea) {
          best = face;
          bestArea = area;
        }
      }
    }
    return best;
  }

  overlayCanvas.addEventListener("click", (e) => {
    const face = faceAtPoint(e.clientX, e.clientY);
    // Clicking the same face again, or empty space, clears the selection
    selectFace(face && face.id !== selectedFaceId ? face.id : null);
  });

  // Cursor only: tells the user a face is clickable without redrawing anything
  overlayCanvas.addEventListener("mousemove", (e) => {
    overlayCanvas.style.cursor = faceAtPoint(e.clientX, e.clientY) ? "pointer" : "default";
  });

  selectionClear.addEventListener("click", () => selectFace(null));

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && selectedFaceId !== null) selectFace(null);
  });

  function selectFace(faceId) {
    selectedFaceId = faceId;
    drawCanvas();
    syncCardSelection(true);
    updateSelectionBar();
  }

  /** Marks the matching card as selected and scrolls the list to it. */
  function syncCardSelection(scrollToCard) {
    let selectedCard = null;
    resultsList.querySelectorAll(".result-card").forEach((card) => {
      const isSel = selectedFaceId !== null && parseInt(card.dataset.faceId, 10) === selectedFaceId;
      card.classList.toggle("selected", isSel);
      card.setAttribute("aria-pressed", isSel ? "true" : "false");
      if (isSel) selectedCard = card;
    });

    if (selectedCard && scrollToCard) {
      const top = selectedCard.offsetTop - (resultsList.clientHeight - selectedCard.offsetHeight) / 2;
      resultsList.scrollTo({ top: Math.max(0, top), behavior: "smooth" });

      // Restart the pulse so repeated clicks always flash
      selectedCard.classList.remove("pulse");
      void selectedCard.offsetWidth;
      selectedCard.classList.add("pulse");
    }
  }

  function updateLegend() {
    legend.style.display = isRecognized ? "flex" : "none";
  }

  function updateSelectionBar() {
    const face = currentFacesData && currentFacesData.faces
      ? currentFacesData.faces.find((f) => f.id === selectedFaceId)
      : null;

    if (!face) {
      selectionInfo.style.display = "none";
      selectionEmpty.style.display = "flex";
      return;
    }

    const color = faceColor(face);
    setSelectionColors(color);

    selectionThumb.src = cropFace(fullResImage, face.bbox, 96);
    if (!isRecognized) {
      selectionName.textContent = `Face ${face.id}`;
      selectionSub.textContent = "Run recognition to identify this person";
    } else if (face.is_unknown) {
      selectionName.textContent = `Unknown #${face.id}`;
      selectionSub.textContent = "No reference match above the threshold";
    } else {
      selectionName.textContent = face.name;
      selectionSub.textContent = `Identified · ${Math.round((face.confidence || 0) * 100)}% similarity`;
    }

    selectionEmpty.style.display = "none";
    selectionInfo.style.display = "flex";
  }

  function setSelectionColors(color) {
    const ink = color === COLOR_MATCHED ? "#047857" : color === COLOR_UNKNOWN ? "#b45309" : "#1d4ed8";
    selectionInfo.style.setProperty("--sel-color", color);
    selectionInfo.style.setProperty("--sel-ink", ink);
  }

  // =========================================================================
  // 7. Results Panel Rendering & Thumbnail Cropping
  // =========================================================================

  function renderResultsPanel(data) {
    resultsPanel.style.display = "flex";

    // Summary
    summaryTitle.textContent = "Recognition complete";
    statMatched.textContent = data.matched;
    statUnknown.textContent = data.unknown;
    statTotal.textContent = data.total_detected;

    // Horizontal stacked proportion bar
    const total = data.total_detected || 1;
    proportionMatched.style.width = `${(data.matched / total) * 100}%`;
    proportionUnknown.style.width = `${(data.unknown / total) * 100}%`;

    // Clear list
    resultsList.innerHTML = "";

    const matchedFaces = data.faces.filter((f) => !f.is_unknown);
    const unknownFaces = data.faces.filter((f) => f.is_unknown);

    appendGroup("Identified", "matched", matchedFaces, false);
    appendGroup("Unknown", "unknown", unknownFaces, true);
  }

  function appendGroup(title, kind, faces, isUnknown) {
    if (!faces.length) return;

    const header = document.createElement("div");
    header.className = `group-separator ${kind}`;
    const label = document.createElement("span");
    label.textContent = title;
    const count = document.createElement("span");
    count.className = "group-count";
    count.textContent = faces.length;
    header.append(label, count);
    resultsList.appendChild(header);

    faces.forEach((face) => resultsList.appendChild(createResultCard(face, isUnknown)));
  }

  function createResultCard(face, isUnknown) {
    const card = document.createElement("div");
    card.className = `result-card ${isUnknown ? "is-unknown" : "is-matched"}`;
    card.dataset.faceId = face.id;
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.setAttribute("aria-pressed", "false");

    const confPct = Math.round((face.confidence || 0) * 100);
    const displayName = isUnknown ? `Unknown #${face.id}` : face.name;

    const thumb = document.createElement("img");
    thumb.className = "face-thumb";
    thumb.src = cropFace(fullResImage, face.bbox, 96); // 2x for crisp 48px display
    thumb.alt = displayName;

    const details = document.createElement("div");
    details.className = "card-details";
    details.innerHTML = `
      <div class="card-name-row">
        <span class="person-name ${isUnknown ? "unknown" : "matched"}"></span>
        <span class="confidence-val">${isUnknown ? "" : `${confPct}%`}</span>
      </div>
      <div class="card-confidence-bar">
        <div class="card-confidence-fill ${isUnknown ? "unknown" : "matched"}" style="width: ${isUnknown ? 100 : confPct}%;"></div>
      </div>
    `;
    details.querySelector(".person-name").textContent = displayName;

    card.append(thumb, details);

    // Click a card -> select that face and spotlight it on the photo
    const toggle = () => selectFace(selectedFaceId === face.id ? null : face.id);
    card.addEventListener("click", toggle);
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        toggle();
      }
    });

    return card;
  }

  /**
   * Crop a square face thumbnail from the full-resolution image with 20% padding.
   */
  function cropFace(imgElement, bbox, size = 48) {
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");

    if (!imgElement || !imgElement.naturalWidth) {
      return "";
    }

    const [x1, y1, x2, y2] = bbox;
    const faceW = x2 - x1;
    const faceH = y2 - y1;

    // 20% padding around face
    const pad = 0.2;
    const px = faceW * pad;
    const py = faceH * pad;

    const srcX = Math.max(0, x1 - px);
    const srcY = Math.max(0, y1 - py);
    const srcW = Math.min(imgElement.naturalWidth - srcX, faceW + px * 2);
    const srcH = Math.min(imgElement.naturalHeight - srcY, faceH + py * 2);

    ctx.drawImage(imgElement, srcX, srcY, srcW, srcH, 0, 0, size, size);

    return canvas.toDataURL("image/jpeg", 0.88);
  }

  // =========================================================================
  // 8. Responsive Window Resizing
  // =========================================================================

  // Re-fit the photo when the stage resizes; the image observer redraws the canvas.
  new ResizeObserver(() => {
    if (photoStage.style.display !== "none") fitImageToStage();
  }).observe(viewerCard);

  new ResizeObserver(() => {
    if (photoStage.style.display !== "none" && displayImage.clientWidth > 0) {
      drawCanvas();
    }
  }).observe(displayImage);

  // Helper debounce
  function debounce(func, wait) {
    let timeout;
    return function (...args) {
      clearTimeout(timeout);
      timeout = setTimeout(() => func.apply(this, args), wait);
    };
  }
});
