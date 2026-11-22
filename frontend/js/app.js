/**
 * Face Detection Frontend Application (Step 1)
 *
 * Handles file selection, image rendering, overlay canvas drawing,
 * and calls the /api/detect endpoint to highlight detected faces.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const fileInput = document.getElementById("fileInput");
  const uploadBtn = document.getElementById("uploadBtn");
  const detectBtn = document.getElementById("detectBtn");
  const detectIcon = document.getElementById("detectIcon");
  const detectText = document.getElementById("detectText");
  const emptyState = document.getElementById("emptyState");
  const photoStage = document.getElementById("photoStage");
  const displayImage = document.getElementById("displayImage");
  const overlayCanvas = document.getElementById("overlayCanvas");
  const statusText = document.getElementById("statusText");

  let currentFile = null;
  let lastDetectionData = null;

  // 1. File Upload Triggers
  uploadBtn.addEventListener("click", () => fileInput.click());
  emptyState.addEventListener("click", () => fileInput.click());

  // 2. Handle File Selection
  fileInput.addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setStatus("Please select a valid image file (JPG, PNG, WEBP).", "error");
      return;
    }

    currentFile = file;
    lastDetectionData = null;

    const reader = new FileReader();
    reader.onload = (loadEvent) => {
      displayImage.src = loadEvent.target.result;
    };
    reader.readAsDataURL(file);
  });

  // 3. Image Loaded Callback
  displayImage.addEventListener("load", () => {
    // Show photo stage, hide empty state
    emptyState.style.display = "none";
    photoStage.style.display = "inline-block";

    // Synchronize canvas dimensions with displayed image dimensions
    syncCanvasSize();

    // Enable detect button
    detectBtn.disabled = false;
    setStatus("Photo loaded. Click 'Detect Faces' to locate faces.", "");
  });

  // 4. Synchronize Canvas Size with Displayed Image
  function syncCanvasSize() {
    if (!displayImage.clientWidth || !displayImage.clientHeight) return;

    overlayCanvas.width = displayImage.clientWidth;
    overlayCanvas.height = displayImage.clientHeight;
  }

  // 5. Detect Faces Action
  detectBtn.addEventListener("click", async () => {
    if (!currentFile) return;

    setLoading(true);
    setStatus("Detecting faces with InsightFace (SCRFD)...", "");

    const formData = new FormData();
    formData.append("file", currentFile);

    try {
      const response = await fetch("/api/detect", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      lastDetectionData = data;
      syncCanvasSize();
      drawBoundingBoxes(data);

      const total = data.total;
      setStatus(
        total === 1 ? "Detected 1 face" : `Detected ${total} faces`,
        "success"
      );
    } catch (err) {
      console.error("Detection error:", err);
      setStatus(`Detection failed: ${err.message}`, "error");
    } finally {
      setLoading(false);
    }
  });

  // 6. Draw Bounding Boxes and Labels on Canvas
  function drawBoundingBoxes(data) {
    const ctx = overlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    if (!data || !data.faces || data.faces.length === 0) return;

    // Coordinate scaling factors
    const scaleX = overlayCanvas.width / data.image_width;
    const scaleY = overlayCanvas.height / data.image_height;

    data.faces.forEach((face) => {
      const x1 = face.bbox[0] * scaleX;
      const y1 = face.bbox[1] * scaleY;
      const x2 = face.bbox[2] * scaleX;
      const y2 = face.bbox[3] * scaleY;
      const boxWidth = x2 - x1;
      const boxHeight = y2 - y1;

      // Draw bounding box
      ctx.strokeStyle = "#3B82F6";
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x1, y1, boxWidth, boxHeight);

      // Label background & text
      const labelText = `Face ${face.id}`;
      ctx.font = "bold 12px system-ui, -apple-system, sans-serif";
      const textMetrics = ctx.measureText(labelText);
      const paddingX = 6;
      const paddingY = 4;
      const badgeWidth = textMetrics.width + paddingX * 2;
      const badgeHeight = 12 + paddingY * 2;

      // Position label above the box; if at top edge, place inside the box
      const badgeY = y1 - badgeHeight >= 0 ? y1 - badgeHeight : y1;

      // Draw label background
      ctx.fillStyle = "#3B82F6";
      ctx.fillRect(x1, badgeY, badgeWidth, badgeHeight);

      // Draw label text
      ctx.fillStyle = "#FFFFFF";
      ctx.textBaseline = "top";
      ctx.fillText(labelText, x1 + paddingX, badgeY + paddingY);
    });
  }

  // 7. Responsive Resizing
  const resizeObserver = new ResizeObserver(() => {
    if (photoStage.style.display !== "none" && displayImage.clientWidth > 0) {
      syncCanvasSize();
      if (lastDetectionData) {
        drawBoundingBoxes(lastDetectionData);
      }
    }
  });
  resizeObserver.observe(displayImage);

  // 8. Helper Functions
  function setStatus(message, type) {
    statusText.textContent = message;
    statusText.className = "status-text";
    if (type) statusText.classList.add(type);
  }

  function setLoading(isLoading) {
    detectBtn.disabled = isLoading;
    if (isLoading) {
      detectIcon.className = "spinner";
      detectIcon.textContent = "";
      detectText.textContent = "Detecting...";
    } else {
      detectIcon.className = "btn-icon";
      detectIcon.textContent = "🔍";
      detectText.textContent = "Detect Faces";
    }
  }
});
