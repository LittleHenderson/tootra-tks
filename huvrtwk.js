/**
 * HuvrTwk -- Text-to-Speech overlay for any webpage.
 * Desktop: hover over text 1.5s to hear it.
 * Mobile: touch-hold + draw a circle around text.
 *
 * Usage: inject via bookmarklet or <script src="huvrtwk.js"></script>
 * Zero dependencies -- uses the browser's built-in Web Speech API.
 */
(function HuvrTwk() {
  "use strict";

  /* Prevent double-injection */
  if (window.__huvrtwk_loaded) return;
  window.__huvrtwk_loaded = true;

  var synth = window.speechSynthesis;
  if (!synth) { console.warn("HuvrTwk: Web Speech API not supported"); return; }

  /* ========== State ========== */
  var PREFS_KEY = "huvrtwk_prefs";
  var prefs = loadPrefs();
  var enabled    = true;
  var panelOpen  = false;
  var speaking   = false;
  var hoverTimer = null;
  var currentEl  = null;
  var isTouch    = ("ontouchstart" in window) || (navigator.maxTouchPoints > 0);

  /* Touch circle state */
  var drawing       = false;
  var holdTimer     = null;
  var touchStart    = null;
  var touchPoints   = [];

  /* ========== Inject styles ========== */
  var css = document.createElement("style");
  css.textContent = "\
    #huvrtwk-pill {\
      position: fixed;\
      z-index: 2147483646;\
      width: 44px;\
      height: 44px;\
      border-radius: 50%;\
      background: rgba(20, 20, 40, 0.75);\
      backdrop-filter: blur(12px);\
      -webkit-backdrop-filter: blur(12px);\
      border: 1px solid rgba(255,255,255,0.12);\
      cursor: pointer;\
      display: flex;\
      align-items: center;\
      justify-content: center;\
      box-shadow: 0 4px 20px rgba(0,0,0,0.35);\
      transition: transform 0.2s, background 0.3s, width 0.3s, border-radius 0.3s;\
      touch-action: none;\
      user-select: none;\
      -webkit-user-select: none;\
    }\
    #huvrtwk-pill:hover { transform: scale(1.08); }\
    #huvrtwk-pill.speaking { background: rgba(255, 145, 0, 0.8); }\
    #huvrtwk-pill.off { opacity: 0.45; }\
    #huvrtwk-pill svg {\
      width: 22px; height: 22px;\
      fill: none; stroke: #fff; stroke-width: 1.8;\
      stroke-linecap: round; stroke-linejoin: round;\
    }\
    #huvrtwk-panel {\
      position: fixed;\
      z-index: 2147483645;\
      width: 240px;\
      background: rgba(16, 16, 36, 0.88);\
      backdrop-filter: blur(20px);\
      -webkit-backdrop-filter: blur(20px);\
      border: 1px solid rgba(255,255,255,0.1);\
      border-radius: 16px;\
      padding: 0;\
      max-height: 0;\
      overflow: hidden;\
      opacity: 0;\
      transition: max-height 0.35s ease, opacity 0.25s ease, padding 0.35s ease;\
      box-shadow: 0 8px 40px rgba(0,0,0,0.5);\
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;\
      color: #e0e0e0;\
    }\
    #huvrtwk-panel.open {\
      max-height: 500px;\
      opacity: 1;\
      padding: 16px;\
    }\
    #huvrtwk-panel .hw-title {\
      font-size: 13px;\
      font-weight: 700;\
      letter-spacing: 1.5px;\
      text-transform: uppercase;\
      color: rgba(255,255,255,0.4);\
      margin-bottom: 14px;\
      text-align: center;\
    }\
    #huvrtwk-panel .hw-row {\
      display: flex;\
      align-items: center;\
      justify-content: space-between;\
      margin-bottom: 12px;\
    }\
    #huvrtwk-panel .hw-label {\
      font-size: 12px;\
      color: rgba(255,255,255,0.55);\
      min-width: 48px;\
    }\
    #huvrtwk-panel select {\
      flex: 1;\
      background: rgba(255,255,255,0.08);\
      color: #e0e0e0;\
      border: 1px solid rgba(255,255,255,0.1);\
      border-radius: 6px;\
      padding: 5px 6px;\
      font-size: 12px;\
      max-width: 170px;\
      outline: none;\
    }\
    #huvrtwk-panel input[type='range'] {\
      flex: 1;\
      -webkit-appearance: none;\
      appearance: none;\
      height: 4px;\
      background: rgba(255,255,255,0.15);\
      border-radius: 2px;\
      outline: none;\
      margin: 0 6px;\
    }\
    #huvrtwk-panel input[type='range']::-webkit-slider-thumb {\
      -webkit-appearance: none;\
      width: 14px; height: 14px;\
      border-radius: 50%;\
      background: #e94560;\
      cursor: pointer;\
    }\
    #huvrtwk-panel .hw-val {\
      font-size: 11px;\
      color: rgba(255,255,255,0.5);\
      width: 28px;\
      text-align: right;\
    }\
    #huvrtwk-panel .hw-toggle {\
      display: flex;\
      align-items: center;\
      justify-content: space-between;\
      padding: 8px 0;\
      border-top: 1px solid rgba(255,255,255,0.06);\
      margin-top: 4px;\
    }\
    .hw-switch {\
      position: relative;\
      width: 40px; height: 22px;\
      background: rgba(255,255,255,0.12);\
      border-radius: 11px;\
      cursor: pointer;\
      transition: background 0.25s;\
    }\
    .hw-switch.on { background: #00c853; }\
    .hw-switch::after {\
      content: '';\
      position: absolute;\
      top: 2px; left: 2px;\
      width: 18px; height: 18px;\
      border-radius: 50%;\
      background: #fff;\
      transition: transform 0.25s;\
    }\
    .hw-switch.on::after { transform: translateX(18px); }\
    #huvrtwk-stop {\
      width: 100%;\
      padding: 7px 0;\
      background: rgba(233, 69, 96, 0.2);\
      border: 1px solid rgba(233, 69, 96, 0.3);\
      border-radius: 8px;\
      color: #e94560;\
      font-size: 12px;\
      font-weight: 600;\
      cursor: pointer;\
      text-align: center;\
      margin-top: 8px;\
      transition: background 0.2s;\
      display: none;\
    }\
    #huvrtwk-stop:hover { background: rgba(233, 69, 96, 0.35); }\
    #huvrtwk-stop.visible { display: block; }\
    .huvrtwk-highlight {\
      background: rgba(233, 69, 96, 0.18) !important;\
      outline: 1px solid rgba(233, 69, 96, 0.35) !important;\
      border-radius: 2px !important;\
      transition: background 0.15s !important;\
    }\
    .huvrtwk-circle-sel {\
      background: rgba(68, 138, 255, 0.15) !important;\
      outline: 2px solid rgba(68, 138, 255, 0.5) !important;\
      border-radius: 2px !important;\
    }\
    .huvrtwk-word-active {\
      background: rgba(255, 200, 0, 0.3) !important;\
      border-radius: 2px !important;\
    }\
    #huvrtwk-ring {\
      position: fixed;\
      pointer-events: none;\
      width: 26px; height: 26px;\
      z-index: 2147483647;\
      display: none;\
    }\
    #huvrtwk-ring circle {\
      fill: none;\
      stroke: #e94560;\
      stroke-width: 2.5;\
      stroke-dasharray: 69.12;\
      stroke-dashoffset: 69.12;\
      transform: rotate(-90deg);\
      transform-origin: center;\
    }\
    #huvrtwk-canvas {\
      position: fixed;\
      top: 0; left: 0;\
      width: 100%; height: 100%;\
      pointer-events: none;\
      z-index: 2147483644;\
    }\
    #huvrtwk-canvas.active { pointer-events: auto; }\
    @media (max-width: 500px) {\
      #huvrtwk-panel { width: 210px; }\
      #huvrtwk-panel select { max-width: 140px; font-size: 11px; }\
    }\
  ";
  document.head.appendChild(css);

  /* ========== Build DOM ========== */

  /* -- Pill button -- */
  var pill = document.createElement("div");
  pill.id = "huvrtwk-pill";
  pill.innerHTML = '<svg viewBox="0 0 24 24">' +
    '<path d="M12 1C9.24 1 7 3.24 7 6v6c0 2.76 2.24 5 5 5s5-2.24 5-5V6c0-2.76-2.24-5-5-5z"/>' +
    '<path d="M19 10v2c0 3.87-3.13 7-7 7s-7-3.13-7-7v-2"/>' +
    '<line x1="12" y1="19" x2="12" y2="23"/>' +
    '<line x1="8" y1="23" x2="16" y2="23"/>' +
    '</svg>';
  document.body.appendChild(pill);

  /* -- Panel -- */
  var panel = document.createElement("div");
  panel.id = "huvrtwk-panel";
  panel.innerHTML =
    '<div class="hw-title">HuvrTwk</div>' +
    '<div class="hw-row"><span class="hw-label">Voice</span><select id="hw-voice"></select></div>' +
    '<div class="hw-row"><span class="hw-label">Speed</span>' +
      '<input type="range" id="hw-speed" min="0.4" max="2.5" step="0.1" value="' + prefs.rate + '">' +
      '<span class="hw-val" id="hw-speed-v">' + prefs.rate + '</span></div>' +
    '<div class="hw-row"><span class="hw-label">Pitch</span>' +
      '<input type="range" id="hw-pitch" min="0" max="2" step="0.1" value="' + prefs.pitch + '">' +
      '<span class="hw-val" id="hw-pitch-v">' + prefs.pitch + '</span></div>' +
    '<div class="hw-row"><span class="hw-label">Delay</span>' +
      '<input type="range" id="hw-delay" min="0.3" max="4" step="0.1" value="' + prefs.delay + '">' +
      '<span class="hw-val" id="hw-delay-v">' + prefs.delay + 's</span></div>' +
    '<div class="hw-row"><span class="hw-label">Read</span>' +
      '<select id="hw-read-mode">' +
        '<option value="auto"' + (prefs.readMode === "auto" ? " selected" : "") + '>Auto</option>' +
        '<option value="sentence"' + (prefs.readMode === "sentence" ? " selected" : "") + '>Sentence</option>' +
        '<option value="paragraph"' + (prefs.readMode === "paragraph" ? " selected" : "") + '>Paragraph</option>' +
      '</select></div>' +
    '<div class="hw-toggle"><span class="hw-label">Karaoke</span>' +
      '<div class="hw-switch' + (prefs.karaoke ? " on" : "") + '" id="hw-karaoke"></div></div>' +
    '<div class="hw-toggle"><span class="hw-label">Enabled</span>' +
      '<div class="hw-switch on" id="hw-enabled"></div></div>' +
    '<div id="huvrtwk-stop">Stop Speaking</div>';
  document.body.appendChild(panel);

  /* -- Progress ring -- */
  var ring = document.createElement("div");
  ring.id = "huvrtwk-ring";
  ring.innerHTML = '<svg viewBox="0 0 26 26"><circle cx="13" cy="13" r="11"/></svg>';
  document.body.appendChild(ring);

  /* -- Touch canvas -- */
  var canvas = document.createElement("canvas");
  canvas.id = "huvrtwk-canvas";
  document.body.appendChild(canvas);
  var ctx = canvas.getContext("2d");
  function resizeCanvas() { canvas.width = window.innerWidth; canvas.height = window.innerHeight; }
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);

  /* ========== Panel refs ========== */
  var voiceSel  = document.getElementById("hw-voice");
  var speedR    = document.getElementById("hw-speed");
  var speedV    = document.getElementById("hw-speed-v");
  var pitchR    = document.getElementById("hw-pitch");
  var pitchV    = document.getElementById("hw-pitch-v");
  var delayR    = document.getElementById("hw-delay");
  var delayV    = document.getElementById("hw-delay-v");
  var readMSel  = document.getElementById("hw-read-mode");
  var karToggle = document.getElementById("hw-karaoke");
  var enToggle  = document.getElementById("hw-enabled");
  var stopBtn   = document.getElementById("huvrtwk-stop");

  /* ========== Voices ========== */
  function loadVoices() {
    var voices = synth.getVoices();
    voiceSel.innerHTML = "";
    voices.forEach(function(v, i) {
      var o = document.createElement("option");
      o.value = i;
      o.textContent = v.name.replace(/Microsoft |Google |Apple /g, "") + " (" + v.lang + ")";
      if (prefs.voiceName && v.name === prefs.voiceName) o.selected = true;
      else if (!prefs.voiceName && v.default) o.selected = true;
      voiceSel.appendChild(o);
    });
  }
  loadVoices();
  if (synth.onvoiceschanged !== undefined) synth.onvoiceschanged = loadVoices;

  /* ========== Pill positioning & drag ========== */
  var pillPos = prefs.pillPos || { right: 8, top: Math.round(window.innerHeight * 0.4) };
  positionPill();

  function positionPill() {
    pill.style.right = pillPos.right + "px";
    pill.style.top   = pillPos.top   + "px";
    panel.style.right = (pillPos.right + 52) + "px";
    panel.style.top   = pillPos.top + "px";
  }

  var dragData = null;
  pill.addEventListener("pointerdown", function(e) {
    dragData = { startY: e.clientY, origTop: pillPos.top, moved: false };
    pill.setPointerCapture(e.pointerId);
  });
  pill.addEventListener("pointermove", function(e) {
    if (!dragData) return;
    var dy = e.clientY - dragData.startY;
    if (Math.abs(dy) > 5) dragData.moved = true;
    pillPos.top = Math.max(0, Math.min(window.innerHeight - 50, dragData.origTop + dy));
    positionPill();
  });
  pill.addEventListener("pointerup", function() {
    if (dragData && !dragData.moved) togglePanel();
    if (dragData) { prefs.pillPos = pillPos; savePrefs(); }
    dragData = null;
  });

  /* ========== Panel toggle ========== */
  function togglePanel() {
    panelOpen = !panelOpen;
    panel.classList.toggle("open", panelOpen);
  }

  document.addEventListener("pointerdown", function(e) {
    if (panelOpen && !panel.contains(e.target) && e.target !== pill && !pill.contains(e.target)) {
      panelOpen = false;
      panel.classList.remove("open");
    }
  });

  /* ========== Panel controls ========== */
  speedR.addEventListener("input", function() {
    prefs.rate = Number(speedR.value).toFixed(1);
    speedV.textContent = prefs.rate; savePrefs();
  });
  pitchR.addEventListener("input", function() {
    prefs.pitch = Number(pitchR.value).toFixed(1);
    pitchV.textContent = prefs.pitch; savePrefs();
  });
  delayR.addEventListener("input", function() {
    prefs.delay = Number(delayR.value).toFixed(1);
    delayV.textContent = prefs.delay + "s"; savePrefs();
  });
  voiceSel.addEventListener("change", function() {
    var voices = synth.getVoices();
    var idx = parseInt(voiceSel.value, 10);
    if (voices[idx]) prefs.voiceName = voices[idx].name;
    savePrefs();
  });
  readMSel.addEventListener("change", function() { prefs.readMode = readMSel.value; savePrefs(); });

  karToggle.addEventListener("click", function() {
    prefs.karaoke = !prefs.karaoke;
    karToggle.classList.toggle("on", prefs.karaoke);
    savePrefs();
  });

  enToggle.addEventListener("click", function() {
    enabled = !enabled;
    enToggle.classList.toggle("on", enabled);
    pill.classList.toggle("off", !enabled);
    if (!enabled) cancelAll();
  });

  stopBtn.addEventListener("click", function() { synth.cancel(); });

  /* ========== Preferences ========== */
  function loadPrefs() {
    try {
      var s = localStorage.getItem(PREFS_KEY);
      if (s) return JSON.parse(s);
    } catch (e) {}
    return { rate: "1.0", pitch: "1.0", delay: "1.5", voiceName: "", readMode: "auto", karaoke: true, pillPos: null };
  }
  function savePrefs() {
    try { localStorage.setItem(PREFS_KEY, JSON.stringify(prefs)); } catch (e) {}
  }

  /* ========== Speech engine ========== */
  function speak(text, sourceEls) {
    if (!text) return;
    synth.cancel();
    clearHighlights();

    var utter  = new SpeechSynthesisUtterance(text);
    var voices = synth.getVoices();
    var idx    = parseInt(voiceSel.value, 10);
    if (voices[idx]) utter.voice = voices[idx];
    utter.rate  = parseFloat(prefs.rate);
    utter.pitch = parseFloat(prefs.pitch);

    utter.onstart = function() {
      speaking = true;
      pill.classList.add("speaking");
      stopBtn.classList.add("visible");
    };
    utter.onend = function() {
      speaking = false;
      pill.classList.remove("speaking");
      stopBtn.classList.remove("visible");
      clearKaraoke();
    };
    utter.onerror = utter.onend;

    /* Karaoke: highlight words as they're spoken */
    if (prefs.karaoke && sourceEls && sourceEls.length === 1) {
      var el = sourceEls[0];
      var origHTML = el.innerHTML;
      wrapWords(el);
      var wordSpans = el.querySelectorAll(".hw-word");

      utter.onboundary = function(ev) {
        if (ev.name !== "word") return;
        wordSpans.forEach(function(s) { s.classList.remove("huvrtwk-word-active"); });
        var charCount = 0;
        for (var k = 0; k < wordSpans.length; k++) {
          var len = wordSpans[k].textContent.length;
          if (charCount <= ev.charIndex && ev.charIndex < charCount + len + 1) {
            wordSpans[k].classList.add("huvrtwk-word-active");
            break;
          }
          charCount += len + 1;
        }
      };

      var restore = function() { el.innerHTML = origHTML; };
      utter.onend = function() {
        speaking = false;
        pill.classList.remove("speaking");
        stopBtn.classList.remove("visible");
        setTimeout(restore, 600);
      };
      utter.onerror = function() {
        restore(); speaking = false;
        pill.classList.remove("speaking");
        stopBtn.classList.remove("visible");
      };
    }

    synth.speak(utter);
  }

  function wrapWords(el) {
    var text = el.textContent;
    var words = text.split(/(\s+)/);
    el.innerHTML = "";
    words.forEach(function(w) {
      if (/^\s+$/.test(w)) {
        el.appendChild(document.createTextNode(w));
      } else {
        var span = document.createElement("span");
        span.className = "hw-word";
        span.textContent = w;
        el.appendChild(span);
      }
    });
  }

  function clearKaraoke() {
    document.querySelectorAll(".huvrtwk-word-active").forEach(function(el) {
      el.classList.remove("huvrtwk-word-active");
    });
  }

  /* ========== Helpers ========== */
  function cancelAll() {
    clearTimeout(hoverTimer); hoverTimer = null;
    synth.cancel();
    speaking = false;
    pill.classList.remove("speaking");
    stopBtn.classList.remove("visible");
    ring.style.display = "none";
    clearHighlights();
    clearKaraoke();
    currentEl = null;
  }

  function clearHighlights() {
    document.querySelectorAll(".huvrtwk-highlight, .huvrtwk-circle-sel").forEach(function(el) {
      el.classList.remove("huvrtwk-highlight", "huvrtwk-circle-sel");
    });
  }

  var HW_IDS = { "huvrtwk-pill": 1, "huvrtwk-panel": 1, "huvrtwk-ring": 1, "huvrtwk-canvas": 1 };
  function readableEl(el) {
    while (el) {
      if (el.id && HW_IDS[el.id]) return null;
      if (el === panel || el === pill) return null;
      var tag = el.tagName;
      if (tag === "P" || tag === "LI" || tag === "H1" || tag === "H2" || tag === "H3" ||
          tag === "H4" || tag === "H5" || tag === "H6" || tag === "TD" || tag === "TH" ||
          tag === "BLOCKQUOTE" || tag === "FIGCAPTION" || tag === "CAPTION" ||
          tag === "DD" || tag === "DT" || tag === "LABEL" || tag === "A" || tag === "SPAN" ||
          tag === "DIV") {
        /* For DIV, only use it if it has no block-level children (leaf div) */
        if (tag === "DIV") {
          var hasBlock = el.querySelector("p, div, ul, ol, table, h1, h2, h3, h4, h5, h6, blockquote");
          if (hasBlock) { el = el.parentElement; continue; }
        }
        return el;
      }
      el = el.parentElement;
    }
    return null;
  }

  function getReadText(el) {
    var full = (el.textContent || "").trim();
    var mode = prefs.readMode;
    if (mode === "auto" || mode === "paragraph") return full;
    if (mode === "sentence") {
      var m = full.match(/[^.!?]+[.!?]+/g);
      return m ? m[0].trim() : full;
    }
    return full;
  }

  /* ==========================================================
     DESKTOP: Hover-to-speak
     ========================================================== */
  document.addEventListener("mousemove", function(e) {
    if (!enabled || isTouch) return;

    ring.style.left = (e.clientX + 14) + "px";
    ring.style.top  = (e.clientY - 14) + "px";

    var target = readableEl(e.target);
    if (!target || target === currentEl) return;

    cancelAll();
    currentEl = target;
    currentEl.classList.add("huvrtwk-highlight");

    var delay = parseFloat(prefs.delay) * 1000 || 1500;
    ring.style.display = "block";

    var c = ring.querySelector("circle");
    c.style.transition = "none";
    c.style.strokeDashoffset = "69.12";
    void c.offsetWidth;
    c.style.transition = "stroke-dashoffset " + (delay / 1000) + "s linear";
    c.style.strokeDashoffset = "0";

    hoverTimer = setTimeout(function() {
      ring.style.display = "none";
      var txt = getReadText(currentEl);
      speak(txt, [currentEl]);
    }, delay);
  });

  document.addEventListener("mouseout", function(e) {
    if (isTouch || !enabled) return;
    var target = readableEl(e.target);
    if (target && target === currentEl) cancelAll();
  });

  /* ==========================================================
     MOBILE: Touch-hold + circle-draw
     ========================================================== */
  document.addEventListener("touchstart", function(e) {
    if (!enabled) return;
    var t = e.touches[0];
    if (pill.contains(e.target) || panel.contains(e.target)) return;

    touchStart = { x: t.clientX, y: t.clientY };
    clearHighlights();
    clearCanvas();

    holdTimer = setTimeout(function() {
      drawing = true;
      touchPoints = [];
      canvas.classList.add("active");
      if (navigator.vibrate) navigator.vibrate(30);
    }, 350);
  }, { passive: true });

  document.addEventListener("touchmove", function(e) {
    if (!drawing) {
      if (touchStart && holdTimer) {
        var t = e.touches[0];
        var d = Math.hypot(t.clientX - touchStart.x, t.clientY - touchStart.y);
        if (d > 12) { clearTimeout(holdTimer); holdTimer = null; }
      }
      return;
    }
    e.preventDefault();
    var t2 = e.touches[0];
    touchPoints.push({ x: t2.clientX, y: t2.clientY });
    drawCirclePath();
  }, { passive: false });

  document.addEventListener("touchend", function() {
    clearTimeout(holdTimer); holdTimer = null;

    if (!drawing || touchPoints.length < 12) {
      drawing = false;
      canvas.classList.remove("active");
      clearCanvas();
      return;
    }

    drawing = false;
    canvas.classList.remove("active");

    var bbox = getBBox(touchPoints);
    var candidates = document.querySelectorAll("p, li, h1, h2, h3, h4, h5, h6, td, th, blockquote, figcaption, dd, dt, label, span, a");
    var captured = [];

    candidates.forEach(function(el) {
      if (pill.contains(el) || panel.contains(el)) return;
      var r = el.getBoundingClientRect();
      if (rectsOverlap(bbox, r) && ptInPoly(r.left + r.width / 2, r.top + r.height / 2, touchPoints)) {
        captured.push(el);
        el.classList.add("huvrtwk-circle-sel");
      }
    });

    fadeCanvas();

    if (!captured.length) return;

    if (navigator.vibrate) navigator.vibrate(50);

    var text = captured.map(function(el) { return el.textContent.trim(); }).join(". ");
    speak(text, captured);

    var cleanup = setInterval(function() {
      if (!synth.speaking) { clearInterval(cleanup); setTimeout(clearHighlights, 1200); }
    }, 400);
  }, { passive: true });

  /* ---------- Canvas helpers ---------- */
  function drawCirclePath() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (touchPoints.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(touchPoints[0].x, touchPoints[0].y);
    for (var i = 1; i < touchPoints.length; i++) ctx.lineTo(touchPoints[i].x, touchPoints[i].y);
    ctx.strokeStyle = "rgba(68, 138, 255, 0.65)";
    ctx.lineWidth = 2.5;
    ctx.lineCap = "round";
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(touchPoints[touchPoints.length - 1].x, touchPoints[touchPoints.length - 1].y);
    ctx.lineTo(touchPoints[0].x, touchPoints[0].y);
    ctx.strokeStyle = "rgba(68, 138, 255, 0.25)";
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(touchPoints[0].x, touchPoints[0].y);
    for (var j = 1; j < touchPoints.length; j++) ctx.lineTo(touchPoints[j].x, touchPoints[j].y);
    ctx.closePath();
    ctx.fillStyle = "rgba(68, 138, 255, 0.06)";
    ctx.fill();
  }

  function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    touchPoints = [];
  }

  function fadeCanvas() {
    var op = 0.65;
    var pts = touchPoints.slice();
    var iv = setInterval(function() {
      op -= 0.065;
      if (op <= 0 || pts.length < 2) { clearInterval(iv); clearCanvas(); return; }
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (var i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
      ctx.closePath();
      ctx.fillStyle = "rgba(68, 138, 255, " + (op * 0.12) + ")";
      ctx.fill();
      ctx.strokeStyle = "rgba(68, 138, 255, " + op + ")";
      ctx.lineWidth = 2;
      ctx.stroke();
    }, 45);
  }

  /* ---------- Geometry ---------- */
  function getBBox(pts) {
    var x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
    pts.forEach(function(p) { x1 = Math.min(x1, p.x); y1 = Math.min(y1, p.y); x2 = Math.max(x2, p.x); y2 = Math.max(y2, p.y); });
    return { left: x1, top: y1, right: x2, bottom: y2 };
  }
  function rectsOverlap(a, b) {
    return !(a.right < b.left || a.left > b.right || a.bottom < b.top || a.top > b.bottom);
  }
  function ptInPoly(px, py, poly) {
    var ins = false;
    for (var i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      var xi = poly[i].x, yi = poly[i].y, xj = poly[j].x, yj = poly[j].y;
      if (((yi > py) !== (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) ins = !ins;
    }
    return ins;
  }

})();
