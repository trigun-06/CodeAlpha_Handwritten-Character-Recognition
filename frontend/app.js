document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const backendStatus = document.getElementById("backendStatus");
  const tabNoteBtn = document.getElementById("tabNoteBtn");
  const tabCanvasBtn = document.getElementById("tabCanvasBtn");
  const tabOwnerBtn = document.getElementById("tabOwnerBtn");
  const noteMode = document.getElementById("noteMode");
  const canvasMode = document.getElementById("canvasMode");
  const ownerSection = document.getElementById("ownerSection");

  // Owner Modal Elements
  const openOwnerModalBtn = document.getElementById("openOwnerModalBtn");
  const closeOwnerModalBtn = document.getElementById("closeOwnerModalBtn");
  const ownerModal = document.getElementById("ownerModal");
  const modalSwitchToOwnerTabBtn = document.getElementById("modalSwitchToOwnerTabBtn");
  const toggleWatermark = document.getElementById("toggleWatermark");

  // Note Mode Elements
  const dropzone = document.getElementById("dropzone");
  const noteFileInput = document.getElementById("noteFileInput");
  const notePreviewContainer = document.getElementById("notePreviewContainer");
  const notePreviewImg = document.getElementById("notePreviewImg");
  const clearNoteBtn = document.getElementById("clearNoteBtn");
  const processNoteBtn = document.getElementById("processNoteBtn");
  const samplePills = document.querySelectorAll(".sample-pill-btn");

  const sequenceEmptyState = document.getElementById("sequenceEmptyState");
  const sequenceLoading = document.getElementById("sequenceLoading");
  const sequenceResults = document.getElementById("sequenceResults");
  const recognizedOutputText = document.getElementById("recognizedOutputText");
  const statTotalChars = document.getElementById("statTotalChars");
  const statAvgConfidence = document.getElementById("statAvgConfidence");
  const charPillsGrid = document.getElementById("charPillsGrid");
  const annotatedImage = document.getElementById("annotatedImage");
  const copyTextBtn = document.getElementById("copyTextBtn");
  const downloadTextBtn = document.getElementById("downloadTextBtn");
  const resultStatusBadge = document.getElementById("resultStatusBadge");

  // Mode Filter Elements & State
  let selectedMode = "auto";
  const modePillBtns = document.querySelectorAll(".mode-pill-btn");
  const modeHintText = document.getElementById("modeHintText");
  const activeModeLabel = document.getElementById("activeModeLabel");

  const modeHints = {
    auto: "Smart Context: Auto-detects whether your text is numeric or words.",
    digits: "Digits Only: Restricts recognition strictly to numbers (0–9).",
    letters: "Letters Only: Restricts recognition strictly to English alphabet (A–Z).",
    alphanumeric: "Mixed Alphanumeric: Disables filtering and allows any combination of digits and letters."
  };

  const modeLabels = {
    auto: "Auto Context Active",
    digits: "Digits Only (0-9)",
    letters: "Letters Only (A-Z)",
    alphanumeric: "Mixed Alphanumeric"
  };

  // Canvas Mode Elements
  const canvas = document.getElementById("drawingCanvas");
  const ctx = canvas.getContext("2d");
  const penSizeSlider = document.getElementById("penSizeSlider");
  const penSizeDisplay = document.getElementById("penSizeDisplay");
  const clearCanvasBtn = document.getElementById("clearCanvasBtn");
  const predictCanvasBtn = document.getElementById("predictCanvasBtn");
  const canvasEmptyState = document.getElementById("canvasEmptyState");
  const canvasResults = document.getElementById("canvasResults");
  const heroCharacter = document.getElementById("heroCharacter");
  const heroConfidence = document.getElementById("heroConfidence");
  const candidatesList = document.getElementById("candidatesList");
  const canvasCharBreakdown = document.getElementById("canvasCharBreakdown");
  const canvasCharPills = document.getElementById("canvasCharPills");
  const canvasCandidatesContainer = document.getElementById("canvasCandidatesContainer");
  const canvasAnnotatedWrapper = document.getElementById("canvasAnnotatedWrapper");
  const canvasAnnotatedImage = document.getElementById("canvasAnnotatedImage");

  let currentNoteFile = null;
  let isDrawing = false;
  let lastX = 0;
  let lastY = 0;

  // Initialize Canvas
  function initCanvas() {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = parseInt(penSizeSlider.value, 10);
  }
  initCanvas();

  // 1. Health Check & Ownership Status
  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      if (res.ok) {
        const data = await res.json();
        backendStatus.className = "status-pill status-ready";
        backendStatus.querySelector(".status-text").textContent = `Scrawl Ready (${data.num_classes} Classes)`;
      } else {
        throw new Error("Health check returned status " + res.status);
      }
    } catch (err) {
      backendStatus.className = "status-pill status-error";
      backendStatus.querySelector(".status-text").textContent = "Model Offline";
    }
  }
  checkHealth();

  // 2. Tab Navigation
  function switchTab(targetId) {
    [tabNoteBtn, tabCanvasBtn, tabOwnerBtn].forEach((btn) => {
      if (btn.dataset.target === targetId) {
        btn.classList.add("active");
        btn.setAttribute("aria-selected", "true");
      } else {
        btn.classList.remove("active");
        btn.setAttribute("aria-selected", "false");
      }
    });

    [noteMode, canvasMode, ownerSection].forEach((sec) => {
      if (sec.id === targetId) {
        sec.classList.remove("hidden");
      } else {
        sec.classList.add("hidden");
      }
    });
  }

  tabNoteBtn.addEventListener("click", () => switchTab("noteMode"));
  tabCanvasBtn.addEventListener("click", () => switchTab("canvasMode"));
  tabOwnerBtn.addEventListener("click", () => switchTab("ownerSection"));

  // 3. Ownership Modal Handlers
  openOwnerModalBtn.addEventListener("click", () => {
    ownerModal.classList.remove("hidden");
  });

  closeOwnerModalBtn.addEventListener("click", () => {
    ownerModal.classList.add("hidden");
  });

  ownerModal.addEventListener("click", (e) => {
    if (e.target === ownerModal) {
      ownerModal.classList.add("hidden");
    }
  });

  modalSwitchToOwnerTabBtn.addEventListener("click", () => {
    ownerModal.classList.add("hidden");
    switchTab("ownerSection");
  });

  // 4. Recognition Mode Pills
  function setRecognitionMode(mode) {
    selectedMode = mode;
    modePillBtns.forEach((btn) => {
      if (btn.dataset.mode === mode) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    });
    if (modeHintText && modeHints[mode]) {
      modeHintText.textContent = modeHints[mode];
    }
    if (activeModeLabel && modeLabels[mode]) {
      activeModeLabel.textContent = modeLabels[mode];
    }
  }

  modePillBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      setRecognitionMode(btn.dataset.mode);
    });
  });

  // 5. Dropzone & File Handling
  function handleFileSelected(file) {
    if (!file || !file.type.startsWith("image/")) {
      alert("Please upload an image file (PNG, JPG, JPEG, WebP).");
      return;
    }
    currentNoteFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      notePreviewImg.src = e.target.result;
      dropzone.classList.add("hidden");
      notePreviewContainer.classList.remove("hidden");
      processNoteBtn.disabled = false;
      if (resultStatusBadge) {
        resultStatusBadge.textContent = "Image Loaded";
        resultStatusBadge.className = "pill-badge pill-blue";
      }
    };
    reader.readAsDataURL(file);
  }

  noteFileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-active");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("drag-active");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-active");
    if (e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  clearNoteBtn.addEventListener("click", () => {
    currentNoteFile = null;
    noteFileInput.value = "";
    notePreviewImg.src = "";
    notePreviewContainer.classList.add("hidden");
    dropzone.classList.remove("hidden");
    processNoteBtn.disabled = true;
    if (resultStatusBadge) {
      resultStatusBadge.textContent = "Engine Idle";
      resultStatusBadge.className = "pill-badge pill-green";
    }
  });

  // 6. Quick Samples
  samplePills.forEach((btn) => {
    btn.addEventListener("click", () => {
      const sampleType = btn.dataset.sample;
      if (sampleType === "995461") {
        setRecognitionMode("digits");
      } else {
        setRecognitionMode("letters");
      }
      generateSampleImage(sampleType);
    });
  });

  function generateSampleImage(text) {
    const offscreen = document.createElement("canvas");
    offscreen.width = 460;
    offscreen.height = 140;
    const offCtx = offscreen.getContext("2d");

    // Clean white background
    offCtx.fillStyle = "#ffffff";
    offCtx.fillRect(0, 0, offscreen.width, offscreen.height);

    // Realistic ink strokes
    offCtx.fillStyle = "#0f172a";
    offCtx.font = "bold 66px 'JetBrains Mono', monospace";
    offCtx.textAlign = "center";
    offCtx.textBaseline = "middle";
    offCtx.fillText(text, offscreen.width / 2, offscreen.height / 2);

    offscreen.toBlob((blob) => {
      const file = new File([blob], `scrawl_sample_${text}.png`, { type: "image/png" });
      handleFileSelected(file);
    });
  }

  // 7. Counter Animation Helper
  function animateValue(element, start, end, duration, isPercentage = false) {
    let startTimestamp = null;
    const step = (timestamp) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      const val = Math.floor(progress * (end - start) + start);
      element.textContent = isPercentage ? `${val}%` : `${val}`;
      if (progress < 1) {
        window.requestAnimationFrame(step);
      } else {
        element.textContent = isPercentage ? `${end.toFixed(1)}%` : `${end}`;
      }
    };
    window.requestAnimationFrame(step);
  }

  // 8. Recognize Note Sequence API Call
  processNoteBtn.addEventListener("click", async () => {
    if (!currentNoteFile) return;

    sequenceEmptyState.classList.add("hidden");
    sequenceResults.classList.add("hidden");
    sequenceLoading.classList.remove("hidden");
    processNoteBtn.disabled = true;

    if (resultStatusBadge) {
      resultStatusBadge.textContent = "Processing...";
      resultStatusBadge.className = "pill-badge pill-purple";
    }

    try {
      const formData = new FormData();
      formData.append("file", currentNoteFile);
      formData.append("mode", selectedMode);

      const res = await fetch("/api/predict/sequence", {
        method: "POST",
        body: formData
      });

      if (!res.ok) {
        throw new Error(`Server returned HTTP ${res.status}`);
      }

      const data = await res.json();
      displaySequenceResults(data);

      if (resultStatusBadge) {
        resultStatusBadge.textContent = "Transcription Ready";
        resultStatusBadge.className = "pill-badge pill-green";
      }
    } catch (err) {
      alert("Error recognizing note: " + err.message);
      sequenceEmptyState.classList.remove("hidden");
      if (resultStatusBadge) {
        resultStatusBadge.textContent = "Error";
        resultStatusBadge.className = "pill-badge";
      }
    } finally {
      sequenceLoading.classList.add("hidden");
      processNoteBtn.disabled = false;
    }
  });

  function displaySequenceResults(data) {
    recognizedOutputText.value = data.recognized_text || "(No characters recognized)";

    // Smooth counter animations
    animateValue(statTotalChars, 0, data.total_characters, 500, false);

    if (data.characters.length > 0) {
      const avg = data.characters.reduce((acc, c) => acc + c.confidence, 0) / data.characters.length;
      animateValue(statAvgConfidence, 0, avg, 600, true);
    } else {
      statAvgConfidence.textContent = "0%";
    }

    // Render character confidence breakdown badges
    charPillsGrid.innerHTML = "";
    data.characters.forEach((c, idx) => {
      const badge = document.createElement("div");
      badge.className = "char-badge";
      badge.style.animationDelay = `${Math.min(idx * 0.03, 0.45)}s`;
      badge.innerHTML = `
        <span class="char-letter">${c.character}</span>
        <span class="char-conf">${c.confidence.toFixed(1)}%</span>
      `;
      charPillsGrid.appendChild(badge);
    });

    // Render annotated document preview
    if (data.annotated_image) {
      annotatedImage.src = data.annotated_image;
    }

    sequenceResults.classList.remove("hidden");
  }

  // 9. Watermark & Export Helpers
  function getAttributedText() {
    let text = recognizedOutputText.value;
    if (toggleWatermark && toggleWatermark.checked && text.trim().length > 0) {
      text += "\n\n— Transcribed with Scrawl by Trigun Parvathapuram";
    }
    return text;
  }

  // Copy & Download
  copyTextBtn.addEventListener("click", () => {
    const textToCopy = getAttributedText();
    navigator.clipboard.writeText(textToCopy);

    const origSvg = copyTextBtn.innerHTML;
    copyTextBtn.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#059669" stroke-width="2.5">
        <polyline points="20 6 9 17 4 12"></polyline>
      </svg>
    `;
    setTimeout(() => {
      copyTextBtn.innerHTML = origSvg;
    }, 1600);
  });

  downloadTextBtn.addEventListener("click", () => {
    const textToSave = getAttributedText();
    const blob = new Blob([textToSave], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "scrawl_transcription.txt";
    a.click();
    URL.revokeObjectURL(url);
  });

  // 10. Interactive Drawing Canvas
  penSizeSlider.addEventListener("input", (e) => {
    const val = parseInt(e.target.value, 10);
    ctx.lineWidth = val;
    penSizeDisplay.textContent = `${val}px`;
  });

  clearCanvasBtn.addEventListener("click", () => {
    initCanvas();
    canvasResults.classList.add("hidden");
    canvasEmptyState.classList.remove("hidden");
  });

  function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    if (e.touches && e.touches.length > 0) {
      return {
        x: (e.touches[0].clientX - rect.left) * scaleX,
        y: (e.touches[0].clientY - rect.top) * scaleY
      };
    }
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  }

  function startDrawing(e) {
    e.preventDefault();
    isDrawing = true;
    const coords = getCanvasCoords(e);
    lastX = coords.x;
    lastY = coords.y;
  }

  function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    const coords = getCanvasCoords(e);

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(coords.x, coords.y);
    ctx.stroke();

    lastX = coords.x;
    lastY = coords.y;
  }

  function stopDrawing(e) {
    if (!isDrawing) return;
    e.preventDefault();
    isDrawing = false;
  }

  // Mouse & Touch Listeners
  canvas.addEventListener("mousedown", startDrawing);
  canvas.addEventListener("mousemove", draw);
  canvas.addEventListener("mouseup", stopDrawing);
  canvas.addEventListener("mouseleave", stopDrawing);

  canvas.addEventListener("touchstart", startDrawing, { passive: false });
  canvas.addEventListener("touchmove", draw, { passive: false });
  canvas.addEventListener("touchend", stopDrawing, { passive: false });

  // Predict from Canvas
  predictCanvasBtn.addEventListener("click", async () => {
    const dataUrl = canvas.toDataURL("image/png");

    predictCanvasBtn.disabled = true;
    try {
      const res = await fetch("/api/predict/character/base64", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataUrl, top_k: 3, mode: selectedMode })
      });

      if (!res.ok) {
        throw new Error(`Server returned HTTP ${res.status}`);
      }

      const data = await res.json();
      displayCanvasResults(data);
    } catch (err) {
      alert("Prediction error: " + err.message);
    } finally {
      predictCanvasBtn.disabled = false;
    }
  });

  function displayCanvasResults(data) {
    canvasEmptyState.classList.add("hidden");
    canvasResults.classList.remove("hidden");

    heroCharacter.textContent = data.character || "—";
    heroCharacter.classList.remove("scale-pop");
    void heroCharacter.offsetWidth; // Trigger reflow for restart animation
    heroCharacter.classList.add("scale-pop");

    // Dynamic sizing for multi-character vs single character
    if (data.character && data.character.length > 2) {
      heroCharacter.style.fontSize = "2.8rem";
      heroCharacter.style.letterSpacing = "0.06em";
    } else {
      heroCharacter.style.fontSize = "4.4rem";
      heroCharacter.style.letterSpacing = "normal";
    }

    heroConfidence.textContent = `${data.confidence.toFixed(1)}%`;

    const isMultiChar = data.characters && data.characters.length > 1;

    // Multi-character breakdown
    if (isMultiChar) {
      if (canvasCharBreakdown && canvasCharPills) {
        canvasCharBreakdown.classList.remove("hidden");
        canvasCharPills.innerHTML = "";
        data.characters.forEach((c, idx) => {
          const badge = document.createElement("div");
          badge.className = "char-badge";
          badge.style.animationDelay = `${idx * 0.05}s`;
          badge.innerHTML = `
            <span class="char-letter">${c.character}</span>
            <span class="char-conf">${c.confidence.toFixed(1)}%</span>
          `;
          canvasCharPills.appendChild(badge);
        });
      }
      if (canvasCandidatesContainer) {
        canvasCandidatesContainer.classList.add("hidden");
      }
    } else {
      if (canvasCharBreakdown) {
        canvasCharBreakdown.classList.add("hidden");
      }
      if (canvasCandidatesContainer && candidatesList) {
        canvasCandidatesContainer.classList.remove("hidden");
        candidatesList.innerHTML = "";
        (data.top_predictions || []).forEach((item) => {
          const row = document.createElement("div");
          row.className = "candidate-row";
          row.innerHTML = `
            <span class="candidate-char">${item.character}</span>
            <div class="candidate-bar-bg">
              <div class="candidate-bar-fill" style="width: ${Math.max(4, item.confidence)}%"></div>
            </div>
            <span class="candidate-score">${item.confidence.toFixed(1)}%</span>
          `;
          candidatesList.appendChild(row);
        });
      }
    }

    // Annotated bounding box overlay
    if (data.annotated_image && canvasAnnotatedWrapper && canvasAnnotatedImage) {
      canvasAnnotatedImage.src = data.annotated_image;
      canvasAnnotatedWrapper.classList.remove("hidden");
    } else if (canvasAnnotatedWrapper) {
      canvasAnnotatedWrapper.classList.add("hidden");
    }
  }

  // Quick drawing sample presets
  function drawSampleOnCanvas(sampleText) {
    initCanvas();
    ctx.save();
    ctx.fillStyle = "#000000";
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = 18;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    if (sampleText === "Hi") {
      // Left vertical stem of H
      ctx.beginPath(); ctx.moveTo(60, 80); ctx.lineTo(65, 195); ctx.stroke();
      // Right vertical stem of H
      ctx.beginPath(); ctx.moveTo(130, 80); ctx.lineTo(135, 195); ctx.stroke();
      // Crossbar of H
      ctx.beginPath(); ctx.moveTo(62, 138); ctx.lineTo(132, 138); ctx.stroke();
      // Dot of i
      ctx.beginPath(); ctx.arc(195, 85, 10, 0, Math.PI * 2); ctx.fill();
      // Stem of i
      ctx.beginPath(); ctx.moveTo(195, 120); ctx.lineTo(198, 195); ctx.stroke();
    } else if (sampleText === "AI") {
      // Slanted left of A
      ctx.beginPath(); ctx.moveTo(85, 80); ctx.lineTo(45, 195); ctx.stroke();
      // Slanted right of A
      ctx.beginPath(); ctx.moveTo(85, 80); ctx.lineTo(125, 195); ctx.stroke();
      // Crossbar of A
      ctx.beginPath(); ctx.moveTo(62, 145); ctx.lineTo(108, 145); ctx.stroke();
      // Top bar of I
      ctx.beginPath(); ctx.moveTo(165, 85); ctx.lineTo(225, 85); ctx.stroke();
      // Vertical stem of I
      ctx.beginPath(); ctx.moveTo(195, 85); ctx.lineTo(195, 195); ctx.stroke();
      // Bottom bar of I
      ctx.beginPath(); ctx.moveTo(165, 195); ctx.lineTo(225, 195); ctx.stroke();
    } else if (sampleText === "A") {
      ctx.beginPath(); ctx.moveTo(140, 60); ctx.lineTo(70, 215); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(140, 60); ctx.lineTo(210, 215); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(95, 150); ctx.lineTo(185, 150); ctx.stroke();
    } else if (sampleText === "42") {
      // Number 4
      ctx.beginPath(); ctx.moveTo(100, 75); ctx.lineTo(55, 145); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(50, 145); ctx.lineTo(125, 145); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(100, 75); ctx.lineTo(100, 195); ctx.stroke();
      // Number 2
      ctx.beginPath();
      ctx.arc(180, 105, 30, Math.PI, 0, false);
      ctx.lineTo(150, 195);
      ctx.lineTo(215, 195);
      ctx.stroke();
    }
    ctx.restore();
  }

  document.querySelectorAll(".canvas-quick-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      drawSampleOnCanvas(btn.dataset.text);
    });
  });
});