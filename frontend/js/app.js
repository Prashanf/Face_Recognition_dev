/**
 * Face Detection & Recognition Frontend Application (Step 2)
 *
 * Implements:
 * 1. Reference folder upload & ArcFace 512-d registration
 * 2. Group photo upload, detection & canvas visualization
 * 3. Face recognition with greedy 1-to-1 matching and results breakdown
 * 4. Real-time threshold re-matching via slider
 * 5. Bidirectional hover interactions between canvas boxes and result cards
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
  const emptyState = document.getElementById("emptyState");
  const photoStage = document.getElementById("photoStage");
  const displayImage = document.getElementById("displayImage");
  const fullResImage = document.getElementById("fullResImage");
  const overlayCanvas = document.getElementById("overlayCanvas");

  // DOM Elements - Step 3 Recognition & Results
  const recognizeBtn = document.getElementById("recognizeBtn");
  const thresholdSlider = document.getElementById("thresholdSlider");
  const thresholdValue = document.getElementById("thresholdValue");
  const resultsPanel = document.getElementById("resultsPanel");
  const summaryTitle = document.getElementById("summaryTitle");
  const summarySubtitle = document.getElementById("summarySubtitle");
  const proportionMatched = document.getElementById("proportionMatched");
  const proportionUnknown = document.getElementById("proportionUnknown");
  const resultsList = document.getElementById("resultsList");

  // Application State
  let referenceFolderReady = false;
  let groupPhotoReady = false;
  let isRecognized = false;
  let currentGroupFile = null;
  let currentFacesData = null; // Holds the most recent response from /api/detect or /api/recognize
  let highlightedFaceId = null;

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
    folderBtn.innerHTML = '<span class="spinner"></span> Processing reference folder...';
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
      folderBtn.textContent = `✅ ${data.registered}/${data.total_files} faces registered`;
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
      folderBtn.textContent = "📁 Select Folder";
      folderProgressBox.style.display = "none";
      alert(`Registration failed: ${err.message}`);
    }
  });

  // Toggle warnings list dropdown
  warningsToggleBtn.addEventListener("click", () => {
    const isHidden = warningsList.style.display === "none";
    warningsList.style.display = isHidden ? "block" : "none";
    warningsToggleBtn.querySelector(".arrow-icon").textContent = isHidden ? "▲" : "▼";
  });

  // =========================================================================
  // 2. Step 2: Group Photo Upload & Initial Detection
  // =========================================================================

  groupPhotoBtn.addEventListener("click", () => groupPhotoInput.click());
  changePhotoLink.addEventListener("click", () => groupPhotoInput.click());
  emptyState.addEventListener("click", () => groupPhotoInput.click());

  // Drag and Drop Handling
  ["dragenter", "dragover"].forEach((eventName) => {
    groupDropTarget.addEventListener(eventName, (e) => {
      e.preventDefault();
      groupDropTarget.classList.add("dragover");
    });
    emptyState.addEventListener(eventName, (e) => {
      e.preventDefault();
      emptyState.style.borderColor = "var(--secondary)";
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    groupDropTarget.addEventListener(eventName, (e) => {
      e.preventDefault();
      groupDropTarget.classList.remove("dragover");
    });
    emptyState.addEventListener(eventName, (e) => {
      e.preventDefault();
      emptyState.style.borderColor = "";
    });
  });

  groupDropTarget.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files && files[0]) {
      handleGroupPhotoSelection(files[0]);
    }
  });

  emptyState.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files && files[0]) {
      handleGroupPhotoSelection(files[0]);
    }
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
    resultsPanel.style.display = "none";

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
    syncCanvasSize();

    // Run initial face detection (Feature 1 flow)
    groupPhotoBtn.disabled = true;
    groupPhotoBtn.innerHTML = '<span class="spinner"></span> Detecting faces...';

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
      groupPhotoBtn.textContent = `✅ Photo uploaded — ${data.total} faces detected`;
      changePhotoLink.style.display = "inline-block";

      // Draw detection bounding boxes (Blue, Feature 1 style)
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
      groupPhotoBtn.textContent = "📷 Upload Photo";
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
    recognizeBtn.innerHTML = '<span class="spinner"></span> Recognizing faces...';

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
      recognizeBtn.innerHTML = "🔄 Re-run Recognition";

      // Draw Recognition Bounding Boxes (Green for Matched, Amber for Unknown)
      drawCanvas();

      // Render Results Panel
      renderResultsPanel(data);
    } catch (err) {
      console.error("Recognition error:", err);
      recognizeBtn.disabled = false;
      recognizeBtn.innerHTML = "🔍 Recognize Faces";
      alert(`Recognition failed: ${err.message}`);
    }
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

      // Redraw Canvas & Update Results Panel in real-time
      drawCanvas();
      renderResultsPanel(data);
    } catch (err) {
      console.error("Re-match error:", err);
    }
  }, 250));

  // =========================================================================
  // 5. Canvas Drawing (Detection vs Recognition + Glow/Highlight)
  // =========================================================================

  function syncCanvasSize() {
    if (!displayImage.clientWidth || !displayImage.clientHeight) return;
    overlayCanvas.width = displayImage.clientWidth;
    overlayCanvas.height = displayImage.clientHeight;
  }

  function drawCanvas() {
    if (!currentFacesData || !currentFacesData.faces) return;
    syncCanvasSize();

    const ctx = overlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    const scaleX = overlayCanvas.width / currentFacesData.image_width;
    const scaleY = overlayCanvas.height / currentFacesData.image_height;

    currentFacesData.faces.forEach((face) => {
      const isHighlighted = (highlightedFaceId === face.id);
      const x1 = face.bbox[0] * scaleX;
      const y1 = face.bbox[1] * scaleY;
      const x2 = face.bbox[2] * scaleX;
      const y2 = face.bbox[3] * scaleY;
      const boxWidth = x2 - x1;
      const boxHeight = y2 - y1;

      // Color selection
      let color = "#3B82F6"; // Feature 1 default Blue
      let label = `Face ${face.id}`;

      if (isRecognized) {
        if (face.is_unknown) {
          color = "#F59E0B"; // Amber for Unknown
          label = "Unknown";
        } else {
          color = "#10B981"; // Green for Matched
          label = face.name;
        }
      }

      // If highlighted, draw outer glow and background tint
      if (isHighlighted) {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 8;
        ctx.globalAlpha = 0.35;
        ctx.strokeRect(x1, y1, boxWidth, boxHeight);

        ctx.fillStyle = color;
        ctx.globalAlpha = 0.12;
        ctx.fillRect(x1, y1, boxWidth, boxHeight);
        ctx.restore();
      }

      // Draw Main Bounding Box
      ctx.strokeStyle = color;
      ctx.lineWidth = isHighlighted ? 4 : 2.5;
      ctx.strokeRect(x1, y1, boxWidth, boxHeight);

      // Draw Label Badge
      ctx.font = "bold 13px system-ui, -apple-system, sans-serif";
      const metrics = ctx.measureText(label);
      const padX = 8;
      const padY = 4;
      const badgeW = metrics.width + padX * 2;
      const badgeH = 14 + padY * 2;

      // Place badge at top of the bounding box (sitting on top edge)
      const badgeY = (y1 - badgeH >= 0) ? (y1 - badgeH) : y1;

      ctx.fillStyle = color;
      ctx.fillRect(x1, badgeY, badgeW, badgeH);

      ctx.fillStyle = "#FFFFFF";
      ctx.textBaseline = "top";
      ctx.fillText(label, x1 + padX, badgeY + padY);
    });
  }

  // =========================================================================
  // 6. Interactive Canvas Hover
  // =========================================================================

  overlayCanvas.addEventListener("mousemove", (e) => {
    if (!currentFacesData || !currentFacesData.faces) return;

    const rect = overlayCanvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const scaleX = overlayCanvas.width / currentFacesData.image_width;
    const scaleY = overlayCanvas.height / currentFacesData.image_height;

    let hoveredFace = null;

    for (const face of currentFacesData.faces) {
      const x1 = face.bbox[0] * scaleX;
      const y1 = face.bbox[1] * scaleY;
      const x2 = face.bbox[2] * scaleX;
      const y2 = face.bbox[3] * scaleY;

      if (mouseX >= x1 && mouseX <= x2 && mouseY >= y1 && mouseY <= y2) {
        hoveredFace = face;
        break;
      }
    }

    if (hoveredFace && highlightedFaceId !== hoveredFace.id) {
      highlightedFaceId = hoveredFace.id;
      drawCanvas();
      highlightResultCard(hoveredFace.id);
    } else if (!hoveredFace && highlightedFaceId !== null) {
      highlightedFaceId = null;
      drawCanvas();
      highlightResultCard(null);
    }
  });

  overlayCanvas.addEventListener("mouseleave", () => {
    if (highlightedFaceId !== null) {
      highlightedFaceId = null;
      drawCanvas();
      highlightResultCard(null);
    }
  });

  function highlightResultCard(faceId) {
    const cards = resultsList.querySelectorAll(".result-card");
    cards.forEach((card) => {
      if (faceId !== null && parseInt(card.dataset.faceId, 10) === faceId) {
        card.classList.add("highlighted");
        card.scrollIntoView({ behavior: "smooth", block: "nearest" });
      } else {
        card.classList.remove("highlighted");
      }
    });
  }

  // =========================================================================
  // 7. Results Panel Rendering & Thumbnail Cropping
  // =========================================================================

  function renderResultsPanel(data) {
    resultsPanel.style.display = "flex";

    // Summary Header
    summaryTitle.textContent = "Recognition Complete";
    summarySubtitle.textContent = `${data.matched} matched, ${data.unknown} unknown out of ${data.total_detected} detected`;

    // Horizontal Stacked Proportion Bar
    const total = data.total_detected || 1;
    const matchedPct = (data.matched / total) * 100;
    const unknownPct = (data.unknown / total) * 100;
    proportionMatched.style.width = `${matchedPct}%`;
    proportionUnknown.style.width = `${unknownPct}%`;

    // Clear list
    resultsList.innerHTML = "";

    const matchedFaces = data.faces.filter((f) => !f.is_unknown);
    const unknownFaces = data.faces.filter((f) => f.is_unknown);

    // Group 1: Identified Faces
    if (matchedFaces.length > 0) {
      const groupHeader = document.createElement("div");
      groupHeader.className = "group-separator";
      groupHeader.textContent = `── Identified (${matchedFaces.length}) ──`;
      resultsList.appendChild(groupHeader);

      matchedFaces.forEach((face) => {
        const card = createResultCard(face, false);
        resultsList.appendChild(card);
      });
    }

    // Group 2: Unknown Faces
    if (unknownFaces.length > 0) {
      const groupHeader = document.createElement("div");
      groupHeader.className = "group-separator";
      groupHeader.textContent = `── Unknown (${unknownFaces.length}) ──`;
      resultsList.appendChild(groupHeader);

      unknownFaces.forEach((face) => {
        const card = createResultCard(face, true);
        resultsList.appendChild(card);
      });
    }
  }

  function createResultCard(face, isUnknown) {
    const card = document.createElement("div");
    card.className = "result-card";
    card.dataset.faceId = face.id;

    // Generate high-resolution 48x48 thumbnail
    const thumbDataUrl = cropFace(fullResImage, face.bbox, 48);

    const confPct = Math.round((face.confidence || 0) * 100);

    card.innerHTML = `
      <img class="face-thumb" src="${thumbDataUrl}" alt="${isUnknown ? 'Unknown' : face.name}">
      <div class="card-details">
        <div class="card-name-row">
          <span class="person-name ${isUnknown ? 'unknown' : 'matched'}">
            ${isUnknown ? `Unknown #${face.id}` : face.name}
          </span>
          <span class="confidence-val">${isUnknown ? '' : `${confPct}%`}</span>
        </div>
        <div class="card-confidence-bar">
          <div class="card-confidence-fill ${isUnknown ? 'unknown' : 'matched'}" style="width: ${isUnknown ? 100 : confPct}%;"></div>
        </div>
      </div>
    `;

    // Card Hover Interaction -> Highlights bounding box on canvas
    card.addEventListener("mouseenter", () => {
      highlightedFaceId = face.id;
      drawCanvas();
      card.classList.add("highlighted");
    });

    card.addEventListener("mouseleave", () => {
      highlightedFaceId = null;
      drawCanvas();
      card.classList.remove("highlighted");
    });

    // Card Click -> centers photo view
    card.addEventListener("click", () => {
      displayImage.scrollIntoView({ behavior: "smooth", block: "center" });
    });

    return card;
  }

  /**
   * Crop pristine 48x48 face thumbnail from full-resolution image with 20% padding.
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

    return canvas.toDataURL("image/jpeg", 0.85);
  }

  // =========================================================================
  // 8. Responsive Window Resizing
  // =========================================================================

  const resizeObserver = new ResizeObserver(() => {
    if (photoStage.style.display !== "none" && displayImage.clientWidth > 0) {
      syncCanvasSize();
      drawCanvas();
    }
  });
  resizeObserver.observe(displayImage);

  // Helper debounce
  function debounce(func, wait) {
    let timeout;
    return function (...args) {
      clearTimeout(timeout);
      timeout = setTimeout(() => func.apply(this, args), wait);
    };
  }
});
