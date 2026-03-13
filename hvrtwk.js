/**
 * HvrTwk -- Text-to-Speech overlay for any webpage.
 * Desktop: hover over text to hear it. Mobile: touch-hold + circle text.
 * Continuous mode: reads from the selected element through the rest of the page.
 * Android-first with chunked speech to avoid Chrome's 15s cutoff.
 *
 * Usage: inject via bookmarklet or <script src="hvrtwk.js"></script>
 * Zero dependencies -- uses the browser's built-in Web Speech API.
 */
(function HvrTwk() {
  "use strict";

  if (window.__hvrtwk_loaded) return;
  window.__hvrtwk_loaded = true;

  var synth = window.speechSynthesis;
  if (!synth) { console.warn("HvrTwk: Web Speech API not supported"); return; }

  /* ========== Platform detection ========== */
  var ua = navigator.userAgent || "";
  var isAndroid = /android/i.test(ua);
  var isTouch   = ("ontouchstart" in window) || (navigator.maxTouchPoints > 0);
  /* Android Chrome kills utterances after ~15s, so we chunk */
  var CHUNK_SIZE = isAndroid ? 120 : 300; /* words per chunk */

  /* ========== State ========== */
  var PREFS_KEY = "hvrtwk_prefs";
  var prefs = loadPrefs();
  var enabled     = true;
  var panelOpen   = false;
  var speaking    = false;
  var hoverTimer  = null;
  var currentEl   = null;
  var contQueue   = [];     /* continuous reading queue of elements */
  var contIndex   = 0;      /* current position in continuous queue */
  var contActive  = false;  /* is continuous reading in progress */
  var utterQueue  = [];     /* chunked utterance queue */
  var utterIndex  = 0;

  /* Touch circle state */
  var drawing     = false;
  var holdTimer   = null;
  var touchStart  = null;
  var touchPoints = [];

  /* ========== Inject styles ========== */
  var css = document.createElement("style");
  css.id = "hvrtwk-styles";
  css.textContent = "\
    #hvrtwk-pill {\
      position: fixed;\
      z-index: 2147483646;\
      width: 46px;\
      height: 46px;\
      border-radius: 50%;\
      background: rgba(20, 20, 40, 0.78);\
      backdrop-filter: blur(14px);\
      -webkit-backdrop-filter: blur(14px);\
      border: 1px solid rgba(255,255,255,0.12);\
      cursor: pointer;\
      display: flex;\
      align-items: center;\
      justify-content: center;\
      box-shadow: 0 4px 24px rgba(0,0,0,0.4);\
      transition: transform 0.2s, background 0.3s;\
      touch-action: none;\
      user-select: none;\
      -webkit-user-select: none;\
      -webkit-tap-highlight-color: transparent;\
    }\
    #hvrtwk-pill:active { transform: scale(0.93); }\
    #hvrtwk-pill.speaking { background: rgba(255, 145, 0, 0.82); }\
    #hvrtwk-pill.continuous { background: rgba(0, 200, 83, 0.8); }\
    #hvrtwk-pill.off { opacity: 0.4; }\
    #hvrtwk-pill svg {\
      width: 22px; height: 22px;\
      fill: none; stroke: #fff; stroke-width: 1.8;\
      stroke-linecap: round; stroke-linejoin: round;\
    }\
    \
    #hvrtwk-panel {\
      position: fixed;\
      z-index: 2147483645;\
      width: 248px;\
      background: rgba(14, 14, 32, 0.92);\
      backdrop-filter: blur(24px);\
      -webkit-backdrop-filter: blur(24px);\
      border: 1px solid rgba(255,255,255,0.08);\
      border-radius: 18px;\
      padding: 0;\
      max-height: 0;\
      overflow: hidden;\
      opacity: 0;\
      transition: max-height 0.3s ease, opacity 0.2s ease, padding 0.3s ease;\
      box-shadow: 0 8px 44px rgba(0,0,0,0.55);\
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;\
      color: #e0e0e0;\
      -webkit-tap-highlight-color: transparent;\
    }\
    #hvrtwk-panel.open {\
      max-height: 560px;\
      opacity: 1;\
      padding: 18px 16px;\
    }\
    #hvrtwk-panel .hw-title {\
      font-size: 14px;\
      font-weight: 800;\
      letter-spacing: 2px;\
      text-transform: uppercase;\
      color: rgba(255,255,255,0.35);\
      margin-bottom: 16px;\
      text-align: center;\
    }\
    #hvrtwk-panel .hw-row {\
      display: flex;\
      align-items: center;\
      justify-content: space-between;\
      margin-bottom: 13px;\
    }\
    #hvrtwk-panel .hw-label {\
      font-size: 12px;\
      color: rgba(255,255,255,0.5);\
      min-width: 52px;\
      flex-shrink: 0;\
    }\
    #hvrtwk-panel select {\
      flex: 1;\
      background: rgba(255,255,255,0.07);\
      color: #e0e0e0;\
      border: 1px solid rgba(255,255,255,0.08);\
      border-radius: 8px;\
      padding: 7px 8px;\
      font-size: 13px;\
      max-width: 172px;\
      outline: none;\
      -webkit-appearance: none;\
      appearance: none;\
    }\
    #hvrtwk-panel input[type='range'] {\
      flex: 1;\
      -webkit-appearance: none;\
      appearance: none;\
      height: 4px;\
      background: rgba(255,255,255,0.12);\
      border-radius: 2px;\
      outline: none;\
      margin: 0 8px;\
    }\
    #hvrtwk-panel input[type='range']::-webkit-slider-thumb {\
      -webkit-appearance: none;\
      width: 18px; height: 18px;\
      border-radius: 50%;\
      background: #e94560;\
      cursor: pointer;\
      box-shadow: 0 2px 6px rgba(233,69,96,0.4);\
    }\
    #hvrtwk-panel .hw-val {\
      font-size: 11px;\
      color: rgba(255,255,255,0.45);\
      width: 30px;\
      text-align: right;\
      flex-shrink: 0;\
    }\
    #hvrtwk-panel .hw-divider {\
      height: 1px;\
      background: rgba(255,255,255,0.06);\
      margin: 6px 0 10px;\
    }\
    #hvrtwk-panel .hw-toggle {\
      display: flex;\
      align-items: center;\
      justify-content: space-between;\
      padding: 6px 0;\
    }\
    .hw-switch {\
      position: relative;\
      width: 42px; height: 24px;\
      background: rgba(255,255,255,0.1);\
      border-radius: 12px;\
      cursor: pointer;\
      transition: background 0.25s;\
      flex-shrink: 0;\
      -webkit-tap-highlight-color: transparent;\
    }\
    .hw-switch.on { background: #00c853; }\
    .hw-switch::after {\
      content: '';\
      position: absolute;\
      top: 2px; left: 2px;\
      width: 20px; height: 20px;\
      border-radius: 50%;\
      background: #fff;\
      transition: transform 0.25s;\
      box-shadow: 0 1px 4px rgba(0,0,0,0.3);\
    }\
    .hw-switch.on::after { transform: translateX(18px); }\
    \
    .hw-btn {\
      width: 100%;\
      padding: 9px 0;\
      border-radius: 10px;\
      font-size: 13px;\
      font-weight: 600;\
      cursor: pointer;\
      text-align: center;\
      transition: background 0.2s;\
      border: none;\
      -webkit-tap-highlight-color: transparent;\
    }\
    #hvrtwk-stop {\
      background: rgba(233, 69, 96, 0.15);\
      border: 1px solid rgba(233, 69, 96, 0.25);\
      color: #e94560;\
      margin-top: 8px;\
      display: none;\
    }\
    #hvrtwk-stop.visible { display: block; }\
    #hvrtwk-stop:active { background: rgba(233, 69, 96, 0.35); }\
    \
    .hvrtwk-hl {\
      background: rgba(233, 69, 96, 0.18) !important;\
      outline: 1px solid rgba(233, 69, 96, 0.3) !important;\
      border-radius: 3px !important;\
      transition: background 0.15s !important;\
    }\
    .hvrtwk-reading {\
      background: rgba(0, 200, 83, 0.12) !important;\
      outline: 2px solid rgba(0, 200, 83, 0.35) !important;\
      border-radius: 3px !important;\
    }\
    .hvrtwk-circle {\
      background: rgba(68, 138, 255, 0.15) !important;\
      outline: 2px solid rgba(68, 138, 255, 0.45) !important;\
      border-radius: 3px !important;\
    }\
    .hvrtwk-word {\
      background: rgba(255, 200, 0, 0.3) !important;\
      border-radius: 2px !important;\
    }\
    \
    #hvrtwk-ring {\
      position: fixed;\
      pointer-events: none;\
      width: 28px; height: 28px;\
      z-index: 2147483647;\
      display: none;\
    }\
    #hvrtwk-ring circle {\
      fill: none;\
      stroke: #e94560;\
      stroke-width: 2.5;\
      stroke-dasharray: 69.12;\
      stroke-dashoffset: 69.12;\
      transform: rotate(-90deg);\
      transform-origin: center;\
    }\
    \
    #hvrtwk-canvas {\
      position: fixed;\
      top: 0; left: 0;\
      width: 100%; height: 100%;\
      pointer-events: none;\
      z-index: 2147483644;\
    }\
    #hvrtwk-canvas.active { pointer-events: auto; }\
    \
    #hvrtwk-progress {\
      position: fixed;\
      bottom: 0; left: 0;\
      height: 3px;\
      background: #00c853;\
      z-index: 2147483645;\
      transition: width 0.3s ease;\
      pointer-events: none;\
      display: none;\
    }\
    #hvrtwk-progress.active { display: block; }\
    \
    @media (max-width: 500px) {\
      #hvrtwk-panel { width: 220px; }\
      #hvrtwk-panel select { max-width: 145px; font-size: 12px; }\
      #hvrtwk-pill { width: 48px; height: 48px; }\
    }\
  ";
  document.head.appendChild(css);

  /* ========== Build DOM ========== */

  var pill = document.createElement("div");
  pill.id = "hvrtwk-pill";
  pill.innerHTML = '<svg viewBox="0 0 24 24">' +
    '<path d="M12 1C9.24 1 7 3.24 7 6v6c0 2.76 2.24 5 5 5s5-2.24 5-5V6c0-2.76-2.24-5-5-5z"/>' +
    '<path d="M19 10v2c0 3.87-3.13 7-7 7s-7-3.13-7-7v-2"/>' +
    '<line x1="12" y1="19" x2="12" y2="23"/>' +
    '<line x1="8" y1="23" x2="16" y2="23"/>' +
    '</svg>';
  document.body.appendChild(pill);

  var panel = document.createElement("div");
  panel.id = "hvrtwk-panel";
  panel.innerHTML =
    '<div class="hw-title">HvrTwk</div>' +

    '<div class="hw-row"><span class="hw-label">Voice</span>' +
      '<select id="hw-voice"></select></div>' +

    '<div class="hw-row"><span class="hw-label">Speed</span>' +
      '<input type="range" id="hw-speed" min="0.4" max="2.5" step="0.1" value="' + prefs.rate + '">' +
      '<span class="hw-val" id="hw-speed-v">' + prefs.rate + '</span></div>' +

    '<div class="hw-row"><span class="hw-label">Pitch</span>' +
      '<input type="range" id="hw-pitch" min="0" max="2" step="0.1" value="' + prefs.pitch + '">' +
      '<span class="hw-val" id="hw-pitch-v">' + prefs.pitch + '</span></div>' +

    '<div class="hw-row"><span class="hw-label">Delay</span>' +
      '<input type="range" id="hw-delay" min="0.3" max="4" step="0.1" value="' + prefs.delay + '">' +
      '<span class="hw-val" id="hw-delay-v">' + prefs.delay + 's</span></div>' +

    '<div class="hw-divider"></div>' +

    '<div class="hw-toggle"><span class="hw-label">Continuous</span>' +
      '<div class="hw-switch' + (prefs.continuous ? ' on' : '') + '" id="hw-cont"></div></div>' +

    '<div class="hw-toggle"><span class="hw-label">Karaoke</span>' +
      '<div class="hw-switch' + (prefs.karaoke ? ' on' : '') + '" id="hw-karaoke"></div></div>' +

    '<div class="hw-toggle"><span class="hw-label">Enabled</span>' +
      '<div class="hw-switch on" id="hw-enabled"></div></div>' +

    '<div id="hvrtwk-stop" class="hw-btn">Stop</div>';
  document.body.appendChild(panel);

  var ring = document.createElement("div");
  ring.id = "hvrtwk-ring";
  ring.innerHTML = '<svg viewBox="0 0 28 28"><circle cx="14" cy="14" r="11"/></svg>';
  document.body.appendChild(ring);

  var canvas = document.createElement("canvas");
  canvas.id = "hvrtwk-canvas";
  document.body.appendChild(canvas);
  var ctx = canvas.getContext("2d");

  var progressBar = document.createElement("div");
  progressBar.id = "hvrtwk-progress";
  document.body.appendChild(progressBar);

  function resizeCanvas() { canvas.width = window.innerWidth; canvas.height = window.innerHeight; }
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);

  /* ========== Panel refs ========== */
  var voiceSel   = document.getElementById("hw-voice");
  var speedR     = document.getElementById("hw-speed");
  var speedV     = document.getElementById("hw-speed-v");
  var pitchR     = document.getElementById("hw-pitch");
  var pitchV     = document.getElementById("hw-pitch-v");
  var delayR     = document.getElementById("hw-delay");
  var delayV     = document.getElementById("hw-delay-v");
  var contToggle = document.getElementById("hw-cont");
  var karToggle  = document.getElementById("hw-karaoke");
  var enToggle   = document.getElementById("hw-enabled");
  var stopBtn    = document.getElementById("hvrtwk-stop");

  /* ========== Voices ========== */
  function loadVoices() {
    var voices = synth.getVoices();
    voiceSel.innerHTML = "";
    /* Sort: prioritize lang matching device locale, then alphabetical */
    var lang = (navigator.language || "en").slice(0, 2).toLowerCase();
    var sorted = voices.map(function(v, i) { return { v: v, i: i }; });
    sorted.sort(function(a, b) {
      var aMatch = a.v.lang.toLowerCase().startsWith(lang) ? 0 : 1;
      var bMatch = b.v.lang.toLowerCase().startsWith(lang) ? 0 : 1;
      if (aMatch !== bMatch) return aMatch - bMatch;
      return a.v.name.localeCompare(b.v.name);
    });
    sorted.forEach(function(item) {
      var o = document.createElement("option");
      o.value = item.i;
      o.textContent = item.v.name.replace(/Microsoft |Google |Apple /g, "") + " (" + item.v.lang + ")";
      if (prefs.voiceName && item.v.name === prefs.voiceName) o.selected = true;
      else if (!prefs.voiceName && item.v.default) o.selected = true;
      voiceSel.appendChild(o);
    });
  }
  loadVoices();
  if (synth.onvoiceschanged !== undefined) synth.onvoiceschanged = loadVoices;

  /* ========== Pill positioning & drag ========== */
  var pillPos = prefs.pillPos || { right: 6, top: Math.round(window.innerHeight * 0.38) };
  positionPill();

  function positionPill() {
    pill.style.right = pillPos.right + "px";
    pill.style.top   = pillPos.top + "px";
    /* Panel anchors to pill */
    var panelRight = pillPos.right + 54;
    /* If panel would go off left edge, flip to right of pill */
    if (window.innerWidth - panelRight < 260) {
      panel.style.right = "auto";
      panel.style.left = "8px";
    } else {
      panel.style.left = "auto";
      panel.style.right = panelRight + "px";
    }
    panel.style.top = Math.min(pillPos.top, window.innerHeight - 420) + "px";
  }

  var dragData = null;
  pill.addEventListener("pointerdown", function(e) {
    dragData = { startY: e.clientY, startX: e.clientX, origTop: pillPos.top, moved: false };
    pill.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  pill.addEventListener("pointermove", function(e) {
    if (!dragData) return;
    var dy = e.clientY - dragData.startY;
    if (Math.abs(dy) > 6) dragData.moved = true;
    pillPos.top = Math.max(0, Math.min(window.innerHeight - 52, dragData.origTop + dy));
    positionPill();
  });
  pill.addEventListener("pointerup", function() {
    if (dragData && !dragData.moved) {
      /* Tap: if continuous reading is active, stop it; otherwise toggle panel */
      if (contActive) {
        stopContinuous();
      } else {
        togglePanel();
      }
    }
    if (dragData && dragData.moved) { prefs.pillPos = pillPos; savePrefs(); }
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
  contToggle.addEventListener("click", function() {
    prefs.continuous = !prefs.continuous;
    contToggle.classList.toggle("on", prefs.continuous);
    savePrefs();
  });
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
  stopBtn.addEventListener("click", function() { stopContinuous(); cancelAll(); });

  /* ========== Preferences ========== */
  function loadPrefs() {
    try {
      var s = localStorage.getItem(PREFS_KEY);
      if (s) return JSON.parse(s);
    } catch (e) {}
    return { rate: "1.0", pitch: "1.0", delay: "1.5", voiceName: "", continuous: false, karaoke: true, pillPos: null };
  }
  function savePrefs() {
    try { localStorage.setItem(PREFS_KEY, JSON.stringify(prefs)); } catch (e) {}
  }

  /* ==========================================================
     SPEECH ENGINE -- Android-safe chunked utterances
     ==========================================================
     Android Chrome silently kills SpeechSynthesisUtterance after
     ~15 seconds. We split long text into word-count chunks and
     chain them, so each utterance is short enough to survive.
     ========================================================== */

  function getVoice() {
    var voices = synth.getVoices();
    var idx = parseInt(voiceSel.value, 10);
    return voices[idx] || null;
  }

  function chunkText(text) {
    var words = text.split(/\s+/);
    if (words.length <= CHUNK_SIZE) return [text];
    var chunks = [];
    for (var i = 0; i < words.length; i += CHUNK_SIZE) {
      chunks.push(words.slice(i, i + CHUNK_SIZE).join(" "));
    }
    return chunks;
  }

  /**
   * Speak text, optionally with karaoke on a single source element.
   * Calls onDone() when finished (for continuous reading chaining).
   */
  function speakText(text, karaokeEl, onDone) {
    synth.cancel();
    utterQueue = chunkText(text);
    utterIndex = 0;
    speaking = true;
    pill.classList.add("speaking");
    stopBtn.classList.add("visible");

    var origHTML = null;
    var wordSpans = null;
    var wordOffset = 0; /* word offset for multi-chunk karaoke */

    if (prefs.karaoke && karaokeEl) {
      origHTML = karaokeEl.innerHTML;
      wrapWords(karaokeEl);
      wordSpans = karaokeEl.querySelectorAll(".hw-w");
    }

    function speakChunk() {
      if (utterIndex >= utterQueue.length || !speaking) {
        finishSpeaking();
        if (onDone) onDone();
        return;
      }

      var chunk = utterQueue[utterIndex];
      var utter = new SpeechSynthesisUtterance(chunk);
      var v = getVoice();
      if (v) utter.voice = v;
      utter.rate  = parseFloat(prefs.rate);
      utter.pitch = parseFloat(prefs.pitch);

      /* Karaoke word tracking */
      if (wordSpans) {
        var chunkWordCount = chunk.split(/\s+/).length;
        var baseOffset = wordOffset;
        wordOffset += chunkWordCount;

        utter.onboundary = function(ev) {
          if (ev.name !== "word" || !wordSpans) return;
          /* Map charIndex within this chunk to word index */
          var before = chunk.slice(0, ev.charIndex);
          var wIdx = before.split(/\s+/).filter(function(w) { return w.length > 0; }).length;
          var globalIdx = baseOffset + wIdx;
          for (var k = 0; k < wordSpans.length; k++) {
            wordSpans[k].classList.toggle("hvrtwk-word", k === globalIdx);
          }
        };
      }

      utter.onend = function() {
        utterIndex++;
        /* Small gap between chunks for natural pacing */
        setTimeout(speakChunk, 80);
      };

      utter.onerror = function() {
        finishSpeaking();
        if (onDone) onDone();
      };

      synth.speak(utter);

      /* Android keep-alive: Chrome sometimes pauses synthesis.
         Periodically calling resume() prevents it. */
      if (isAndroid) startKeepAlive();
    }

    function finishSpeaking() {
      speaking = false;
      pill.classList.remove("speaking");
      stopBtn.classList.remove("visible");
      stopKeepAlive();
      if (origHTML !== null && karaokeEl) {
        setTimeout(function() {
          try { karaokeEl.innerHTML = origHTML; } catch (e) {}
        }, 400);
      }
    }

    speakChunk();
  }

  /* Android keep-alive timer */
  var keepAliveId = null;
  function startKeepAlive() {
    stopKeepAlive();
    keepAliveId = setInterval(function() {
      if (synth.speaking && !synth.paused) {
        synth.pause();
        synth.resume();
      }
    }, 10000);
  }
  function stopKeepAlive() {
    if (keepAliveId) { clearInterval(keepAliveId); keepAliveId = null; }
  }

  function wrapWords(el) {
    var text = el.textContent;
    var parts = text.split(/(\s+)/);
    el.innerHTML = "";
    parts.forEach(function(p) {
      if (/^\s+$/.test(p)) {
        el.appendChild(document.createTextNode(p));
      } else {
        var s = document.createElement("span");
        s.className = "hw-w";
        s.textContent = p;
        el.appendChild(s);
      }
    });
  }

  /* ==========================================================
     CONTINUOUS READING
     ==========================================================
     Collects all readable text elements from the starting element
     through the end of the page, then speaks them sequentially.
     Highlights the current element being read in green.
     Progress bar shows position.
     ========================================================== */

  function getAllReadableEls() {
    var sel = "p, li, h1, h2, h3, h4, h5, h6, td, th, blockquote, figcaption, caption, dd, dt, label";
    var all = document.querySelectorAll(sel);
    var result = [];
    for (var i = 0; i < all.length; i++) {
      if (pill.contains(all[i]) || panel.contains(all[i])) continue;
      var text = (all[i].textContent || "").trim();
      if (text.length > 1) result.push(all[i]);
    }
    return result;
  }

  function startContinuous(fromEl) {
    stopContinuous();
    var allEls = getAllReadableEls();
    if (!allEls.length) return;

    /* Find starting index */
    var startIdx = 0;
    if (fromEl) {
      for (var i = 0; i < allEls.length; i++) {
        if (allEls[i] === fromEl || allEls[i].contains(fromEl) || fromEl.contains(allEls[i])) {
          startIdx = i;
          break;
        }
      }
    }

    contQueue  = allEls;
    contIndex  = startIdx;
    contActive = true;
    pill.classList.add("continuous");
    progressBar.classList.add("active");

    readNext();
  }

  function readNext() {
    if (!contActive || contIndex >= contQueue.length) {
      stopContinuous();
      return;
    }

    /* Clear previous highlight */
    clearHighlights();

    var el = contQueue[contIndex];
    el.classList.add("hvrtwk-reading");

    /* Scroll element into view (smooth, centered) */
    try {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
    } catch (e) {
      el.scrollIntoView(false);
    }

    /* Update progress bar */
    var pct = ((contIndex + 1) / contQueue.length) * 100;
    progressBar.style.width = pct + "%";

    var text = el.textContent.trim();
    var karEl = (prefs.karaoke && contQueue.length > 0) ? el : null;

    speakText(text, karEl, function() {
      if (!contActive) return;
      contIndex++;
      /* Small pause between elements for natural reading */
      setTimeout(readNext, 250);
    });
  }

  function stopContinuous() {
    contActive = false;
    contQueue  = [];
    contIndex  = 0;
    synth.cancel();
    speaking = false;
    pill.classList.remove("continuous");
    pill.classList.remove("speaking");
    stopBtn.classList.remove("visible");
    progressBar.classList.remove("active");
    progressBar.style.width = "0";
    stopKeepAlive();
    clearHighlights();
  }

  /* ========== Helpers ========== */
  function cancelAll() {
    clearTimeout(hoverTimer); hoverTimer = null;
    stopContinuous();
    synth.cancel();
    speaking = false;
    pill.classList.remove("speaking");
    stopBtn.classList.remove("visible");
    ring.style.display = "none";
    clearHighlights();
    currentEl = null;
    stopKeepAlive();
  }

  function clearHighlights() {
    var cls = ["hvrtwk-hl", "hvrtwk-reading", "hvrtwk-circle", "hvrtwk-word"];
    cls.forEach(function(c) {
      document.querySelectorAll("." + c).forEach(function(el) { el.classList.remove(c); });
    });
  }

  var HW_IDS = { "hvrtwk-pill": 1, "hvrtwk-panel": 1, "hvrtwk-ring": 1, "hvrtwk-canvas": 1, "hvrtwk-progress": 1 };
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

  /* ==========================================================
     DESKTOP: Hover-to-speak
     ========================================================== */
  document.addEventListener("mousemove", function(e) {
    if (!enabled || isTouch || contActive) return;

    ring.style.left = (e.clientX + 14) + "px";
    ring.style.top  = (e.clientY - 14) + "px";

    var target = readableEl(e.target);
    if (!target || target === currentEl) return;

    /* New element */
    clearTimeout(hoverTimer);
    synth.cancel();
    ring.style.display = "none";
    if (currentEl) currentEl.classList.remove("hvrtwk-hl");
    currentEl = target;
    currentEl.classList.add("hvrtwk-hl");

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
      if (prefs.continuous) {
        startContinuous(currentEl);
      } else {
        var txt = (currentEl.textContent || "").trim();
        speakText(txt, currentEl, null);
      }
    }, delay);
  });

  document.addEventListener("mouseout", function(e) {
    if (isTouch || !enabled || contActive) return;
    var target = readableEl(e.target);
    if (target && target === currentEl) {
      clearTimeout(hoverTimer); hoverTimer = null;
      ring.style.display = "none";
      synth.cancel();
      speaking = false;
      pill.classList.remove("speaking");
      stopBtn.classList.remove("visible");
      if (currentEl) currentEl.classList.remove("hvrtwk-hl");
      currentEl = null;
      stopKeepAlive();
    }
  });

  /* ==========================================================
     MOBILE: Touch-hold + circle-draw
     ========================================================== */
  document.addEventListener("touchstart", function(e) {
    if (!enabled) return;
    var t = e.touches[0];
    if (pill.contains(e.target) || panel.contains(e.target)) return;

    /* If continuous reading is active, tap anywhere to stop */
    if (contActive) {
      stopContinuous();
      return;
    }

    touchStart = { x: t.clientX, y: t.clientY };
    clearHighlights();
    clearCanvas();

    holdTimer = setTimeout(function() {
      drawing = true;
      touchPoints = [];
      canvas.classList.add("active");
      if (navigator.vibrate) navigator.vibrate(25);
    }, 350);
  }, { passive: true });

  document.addEventListener("touchmove", function(e) {
    if (!drawing) {
      if (touchStart && holdTimer) {
        var t = e.touches[0];
        var d = Math.hypot(t.clientX - touchStart.x, t.clientY - touchStart.y);
        if (d > 14) { clearTimeout(holdTimer); holdTimer = null; }
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

    if (!drawing || touchPoints.length < 10) {
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
        el.classList.add("hvrtwk-circle");
      }
    });

    fadeCanvas();

    if (!captured.length) return;
    if (navigator.vibrate) navigator.vibrate(40);

    if (prefs.continuous && captured.length > 0) {
      /* Start continuous reading from the first circled element */
      startContinuous(captured[0]);
    } else {
      var text = captured.map(function(el) { return el.textContent.trim(); }).join(". ");
      speakText(text, captured.length === 1 ? captured[0] : null, function() {
        setTimeout(clearHighlights, 1000);
      });
    }
  }, { passive: true });

  /* ---------- Canvas helpers ---------- */
  function drawCirclePath() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (touchPoints.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(touchPoints[0].x, touchPoints[0].y);
    for (var i = 1; i < touchPoints.length; i++) ctx.lineTo(touchPoints[i].x, touchPoints[i].y);
    ctx.strokeStyle = "rgba(68, 138, 255, 0.6)";
    ctx.lineWidth = 2.5;
    ctx.lineCap = "round";
    ctx.stroke();
    /* Closing dash */
    ctx.beginPath();
    ctx.moveTo(touchPoints[touchPoints.length - 1].x, touchPoints[touchPoints.length - 1].y);
    ctx.lineTo(touchPoints[0].x, touchPoints[0].y);
    ctx.strokeStyle = "rgba(68, 138, 255, 0.2)";
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
    /* Light fill */
    ctx.beginPath();
    ctx.moveTo(touchPoints[0].x, touchPoints[0].y);
    for (var j = 1; j < touchPoints.length; j++) ctx.lineTo(touchPoints[j].x, touchPoints[j].y);
    ctx.closePath();
    ctx.fillStyle = "rgba(68, 138, 255, 0.05)";
    ctx.fill();
  }

  function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    touchPoints = [];
  }

  function fadeCanvas() {
    var op = 0.6;
    var pts = touchPoints.slice();
    var iv = setInterval(function() {
      op -= 0.06;
      if (op <= 0 || pts.length < 2) { clearInterval(iv); clearCanvas(); return; }
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (var i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
      ctx.closePath();
      ctx.fillStyle = "rgba(68, 138, 255, " + (op * 0.1) + ")";
      ctx.fill();
      ctx.strokeStyle = "rgba(68, 138, 255, " + op + ")";
      ctx.lineWidth = 2;
      ctx.stroke();
    }, 40);
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
