import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";


const TIPS = [4, 8, 12, 16, 20];
const PIPS = [2, 6, 10, 14, 18];

function dist(lm, a, b) {
  return Math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y);
}

const SIGN_GUIDE = [
  ["HELLO (OPEN PALM)", "✋", "Open your whole hand, all five fingers straight and spread apart, palm facing the camera."],
  ["YES", "👍", "Make a fist, then stick the thumb straight UP."],
  ["NO", "👎", "Make a fist, then point the thumb straight DOWN."],
  ["A (FIST)", "✊", "Close all four fingers into a fist. Thumb rests against the side, not up or down."],
  ["B", "🤚", "All four fingers straight and held TOGETHER (no gaps), thumb folded across the palm."],
  ["C", "🤏", "Curve the whole hand like holding a cup — fingers curved, not straight, not a fist."],
  ["O", "👌", "Curl all fingers so the fingertips meet the thumb tip and form a round O."],
  ["U", "✌️", "Index and middle finger straight up, held TOGETHER. Other fingers folded."],
  ["PEACE (V)", "✌️", "Index and middle finger straight up, SPREAD apart into a V. Other fingers folded."],
  ["W", "🖖", "Index, middle and ring finger straight up and spread. Thumb and pinky folded."],
  ["L", "👆", "Thumb out sideways, index finger straight up, making an L. Other fingers folded."],
  ["Y", "🤙", "Thumb and pinky stretched out, three middle fingers folded — like 'call me'."],
  ["I (PINKY)", "☝️", "Only the pinky finger straight up. Everything else folded."],
  ["ONE", "☝️", "Only the index finger pointing straight up. Everything else folded."],
  ["THREE", "🤟", "Thumb, index and middle finger extended. Ring and pinky folded."],
  ["FOUR", "🖐️", "Four fingers straight up and spread, thumb folded across the palm."],
  ["OK", "👌", "Touch thumb tip to index fingertip to make a circle. Other three fingers straight."],
  ["ROCK", "🤘", "Index finger and pinky up, middle and ring folded, thumb tucked in."],
  ["I LOVE YOU", "🤟", "Thumb, index finger and pinky all extended at once. Middle and ring folded."],
];
const BUILT_IN_LABELS = SIGN_GUIDE.map(e => e[0]);

function ruleClassify(lm) {
  const scale = dist(lm, 0, 9);
  if (scale < 1e-4) return null;

  const ext = [false, false, false, false, false];
  ext[0] = dist(lm, 4, 17) > dist(lm, 2, 17) * 1.12;
  for (let f = 1; f <= 4; f++) ext[f] = dist(lm, TIPS[f], 0) > dist(lm, PIPS[f], 0) * 1.15;
  const [t, i, m, r, p] = ext;

  const thumbTouchesIndex = dist(lm, 4, 8) / scale < 0.40;
  const indexMiddleSpread = dist(lm, 8, 12) / scale > 0.45;
  const avgTipDist = (dist(lm, 8, 0) + dist(lm, 12, 0) + dist(lm, 16, 0) + dist(lm, 20, 0)) / (4 * scale);
  const looseCurl = avgTipDist > 1.05;
  const thumbDY = lm[4].y - lm[0].y;
  const thumbUp = thumbDY < -0.10;
  const thumbDown = thumbDY > 0.10;
  const thumbIndexAngleWide = dist(lm, 4, 8) / scale > 1.15;

  if (t && i && p && !m && !r) return ["I LOVE YOU", 0.93];
  if (i && p && !m && !r && !t) return ["ROCK", 0.90];
  if (t && p && !i && !m && !r) return ["Y", 0.90];
  if (t && i && !m && !r && !p) return thumbIndexAngleWide ? ["L", 0.92] : ["ONE", 0.75];
  if (thumbTouchesIndex && m && r && p) return ["OK", 0.90];

  if (!i && !m && !r && !p) {
    if (thumbTouchesIndex && looseCurl) return ["O", 0.85];
    if (t && thumbUp) return ["YES", 0.95];
    if (t && thumbDown) return ["NO", 0.95];
    return ["A (FIST)", 0.88];
  }

  if (i && m && !r && !p) return indexMiddleSpread ? ["PEACE (V)", 0.92] : ["U", 0.85];
  if (i && m && r && !p) return ["W", 0.88];
  if (t && i && m && !r && !p) return ["THREE", 0.86];

  if (i && m && r && p) {
    if (t) return ["HELLO (OPEN PALM)", 0.93];
    return indexMiddleSpread ? ["FOUR", 0.86] : ["B", 0.86];
  }

  if (p && !i && !m && !r) return ["I (PINKY)", 0.88];
  if (i && !m && !r && !p) return ["ONE", 0.88];
  if (looseCurl && !thumbTouchesIndex) return ["C", 0.70];

  return null;
}

class SignClassifier {
  constructor() {
    this.samples = [];       
    this.k = 5;
    this.maxDistance = 0.85;
    this.load();
  }
  static normalize(lm) {
    if (lm.length < 21) return null;
    const baseX = lm[0].x, baseY = lm[0].y;
    const out = new Array(42);
    let max = 0;
    for (let i = 0; i < 21; i++) {
      const dx = lm[i].x - baseX, dy = lm[i].y - baseY;
      out[i * 2] = dx; out[i * 2 + 1] = dy;
      if (Math.abs(dx) > max) max = Math.abs(dx);
      if (Math.abs(dy) > max) max = Math.abs(dy);
    }
    if (max < 1e-6) return null;
    for (let i = 0; i < out.length; i++) out[i] /= max;
    return out;
  }
  addSample(label, features) { this.samples.push({ label, features }); }
  countFor(label) { return this.samples.filter(s => s.label === label).length; }
  trainedLabels() { return [...new Set(this.samples.map(s => s.label))].sort(); }
  totalSamples() { return this.samples.length; }
  clearLabel(label) { this.samples = this.samples.filter(s => s.label !== label); this.save(); }
  clearAll() { this.samples = []; this.save(); }
  predict(features) {
    if (this.samples.length === 0) return null;
    const scored = this.samples
      .map(s => [s.label, euclidean(features, s.features)])
      .sort((a, b) => a[1] - b[1])
      .slice(0, this.k);
    if (scored[0][1] > this.maxDistance) return null;
    const votes = {};
    let total = 0;
    for (const [label, d] of scored) {
      const w = 1 / (d + 1e-3);
      votes[label] = (votes[label] || 0) + w;
      total += w;
    }
    let best = null, bestW = -1;
    for (const label in votes) if (votes[label] > bestW) { bestW = votes[label]; best = label; }
    return [best, bestW / total];
  }
  save() { localStorage.setItem("signlang_dataset", JSON.stringify(this.samples)); }
  load() {
    try {
      const raw = localStorage.getItem("signlang_dataset");
      if (raw) this.samples = JSON.parse(raw);
    } catch (e) { this.samples = []; }
  }
}
function euclidean(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s);
}

class AdminManager {
  constructor() {
    this.loggedIn = false;
    if (!localStorage.getItem("signlang_pin_hash")) {
      localStorage.setItem("signlang_pin_hash", this.hash("1234"));
    }
  }
  hash(s) { let h = 0; for (let i = 0; i < s.length; i++) { h = (h * 31 + s.charCodeAt(i)) | 0; } return String(h); }
  check(pin) { return this.hash(pin) === localStorage.getItem("signlang_pin_hash"); }
  setPin(pin) { localStorage.setItem("signlang_pin_hash", this.hash(pin)); }
}

const classifier = new SignClassifier();
const admin = new AdminManager();

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const octx = overlay.getContext("2d");
const statusPill = document.getElementById("statusPill");
const signText = document.getElementById("signText");
const confText = document.getElementById("confText");
const sentenceBox = document.getElementById("sentenceBox");
const trainBar = document.getElementById("trainBar");
const trainBtn = document.getElementById("trainBtn");
const labelSelect = document.getElementById("labelSelect");
const recordBtn = document.getElementById("recordBtn");
const loadingEl = document.getElementById("loading");
const loadingText = document.getElementById("loadingText");

let handLandmarker = null;
let trainingMode = false;
let samplesLeft = 0;
let sentence = "";
let lastLabel = "", stableCount = 0, committedLabel = "";
const STABLE_FRAMES = 8;
const SAMPLES_PER_SIGN = 40;

const trainableLabels = [
  ...BUILT_IN_LABELS,
  "D","E","F","G","H","K","M","N","P","Q","R","S","T","X","Z"
];

function refreshLabelSelect() {
  const learned = classifier.trainedLabels().filter(l => !trainableLabels.includes(l));
  const all = [...trainableLabels, ...learned];
  labelSelect.innerHTML = all.map(l => `<option value="${l}">${l}</option>`).join("");
}
refreshLabelSelect();

function updateStatus() {
  const builtIn = BUILT_IN_LABELS.length;
  const learned = classifier.trainedLabels().length;
  statusPill.textContent = `${builtIn} built-in · ${learned} learned` + (admin.loggedIn ? " · ADMIN" : "");
  statusPill.className = "status-pill" + (admin.loggedIn ? " admin" : "");
}
updateStatus();

function toast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg; t.style.display = "block";
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.style.display = "none", 2200);
}

/* ---- camera + model setup ---- */
async function setup() {
  try {
    loadingText.textContent = "Requesting camera access…";
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    video.srcObject = stream;
    await new Promise(res => video.onloadedmetadata = res);
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;

    loadingText.textContent = "Loading hand-tracking model…";
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    loadingEl.style.display = "none";
    confText.textContent = "Show a hand sign";
    requestAnimationFrame(loop);
  } catch (err) {
    loadingText.textContent = "Error: " + err.message;
    console.error(err);
    if (err.name === "NotAllowedError") {
      loadingText.textContent = "Camera permission denied. Allow it and reload the page.";
    }
  }
}

const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],[0,17]
];

function drawLandmarks(lm) {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  octx.strokeStyle = "#ffffff"; octx.lineWidth = 2;
  for (const [a, b] of CONNECTIONS) {
    octx.beginPath();
    octx.moveTo(lm[a].x * overlay.width, lm[a].y * overlay.height);
    octx.lineTo(lm[b].x * overlay.width, lm[b].y * overlay.height);
    octx.stroke();
  }
  octx.fillStyle = "#4dd0e1";
  for (const p of lm) {
    octx.beginPath();
    octx.arc(p.x * overlay.width, p.y * overlay.height, 4, 0, Math.PI * 2);
    octx.fill();
  }
}

function loop() {
  if (handLandmarker && video.readyState >= 2) {
    const result = handLandmarker.detectForVideo(video, performance.now());
    if (result.landmarks && result.landmarks.length > 0) {
      const lm = result.landmarks[0];
      drawLandmarks(lm);
      handleFrame(lm);
    } else {
      octx.clearRect(0, 0, overlay.width, overlay.height);
      signText.textContent = "—";
      confText.textContent = "No hand detected";
      stableCount = 0;
    }
  }
  requestAnimationFrame(loop);
}

function handleFrame(lm) {
  if (trainingMode) { handleTrainingFrame(lm); return; }

  const features = SignClassifier.normalize(lm);
  const learned = features ? classifier.predict(features) : null;
  const prediction = learned || ruleClassify(lm);
  const source = learned ? "learned" : "built-in";

  if (!prediction) {
    signText.textContent = "?";
    confText.textContent = "Sign not recognised — check the Guide";
    stableCount = 0;
    return;
  }
  const [label, score] = prediction;
  signText.textContent = label;
  confText.textContent = `${Math.round(score * 100)}%  ·  ${source}`;

  if (label === lastLabel) stableCount++; else { lastLabel = label; stableCount = 0; }
  if (stableCount === STABLE_FRAMES && label !== committedLabel) {
    committedLabel = label;
    const word = label.split(" (")[0];
    sentence += (word.length === 1 ? word : word + " ");
    renderSentence();
  }
  if (stableCount === 0) committedLabel = "";
}

function handleTrainingFrame(lm) {
  if (samplesLeft <= 0) {
    signText.textContent = "READY";
    confText.textContent = "Hold the sign, then press Record";
    return;
  }
  const features = SignClassifier.normalize(lm);
  if (!features) return;
  const label = labelSelect.value;
  classifier.addSample(label, features);
  samplesLeft--;
  signText.textContent = label;
  confText.textContent = `Recording… ${samplesLeft} left`;
  if (samplesLeft === 0) {
    classifier.save();
    confText.textContent = `Saved — ${classifier.countFor(label)} samples for ${label}`;
    recordBtn.disabled = false;
    updateStatus();
  }
}

function renderSentence() {
  sentenceBox.innerHTML = `Sentence: ${sentence}<span class="hint">Click to clear · signs are appended automatically when held steady</span>`;
}
sentenceBox.addEventListener("click", () => { sentence = ""; renderSentence(); });

function setTrainingMode(on) {
  trainingMode = on;
  trainBar.classList.toggle("show", on);
  trainBtn.textContent = on ? "Recognise" : "Train 🔒";
  samplesLeft = 0;
  recordBtn.disabled = false;
  updateStatus();
}

trainBtn.addEventListener("click", () => {
  if (trainingMode) { setTrainingMode(false); return; }
  if (admin.loggedIn) { setTrainingMode(true); toast("Admin mode"); return; }
  showPinModal("Admin login", "Training mode is restricted to authorised users.", (pin) => {
    if (admin.check(pin)) { admin.loggedIn = true; setTrainingMode(true); toast("Admin mode"); return true; }
    toast("Wrong PIN"); return false;
  });
});

recordBtn.addEventListener("click", () => {
  recordBtn.disabled = true;
  confText.textContent = "Get ready…";
  setTimeout(() => { samplesLeft = SAMPLES_PER_SIGN; }, 1500);
});

const guideModal = document.getElementById("guideModal");
document.getElementById("guideBtn").addEventListener("click", () => {
  document.getElementById("guideList").innerHTML = SIGN_GUIDE.map(([name, emoji, how]) =>
    `<div class="guide-entry"><div class="name">${emoji} ${name}</div><div class="how">${how}</div></div>`
  ).join("");
  guideModal.classList.add("show");
});
document.getElementById("guideClose").addEventListener("click", () => guideModal.classList.remove("show"));

const pinModal = document.getElementById("pinModal");
const pinInput = document.getElementById("pinInput");
let pinCallback = null;
function showPinModal(title, msg, onSubmit) {
  document.getElementById("pinTitle").textContent = title;
  document.getElementById("pinMsg").textContent = msg;
  pinInput.value = "";
  pinCallback = onSubmit;
  pinModal.classList.add("show");
  setTimeout(() => pinInput.focus(), 50);
}
document.getElementById("pinOk").addEventListener("click", () => {
  if (pinCallback && pinCallback(pinInput.value)) pinModal.classList.remove("show");
  else if (pinCallback) pinModal.classList.remove("show"); // close either way, toast already shown
});
document.getElementById("pinCancel").addEventListener("click", () => pinModal.classList.remove("show"));
pinInput.addEventListener("keydown", e => { if (e.key === "Enter") document.getElementById("pinOk").click(); });

const infoModal = document.getElementById("infoModal");
function showInfo(title, msg) {
  document.getElementById("infoTitle").textContent = title;
  document.getElementById("infoMsg").textContent = msg;
  infoModal.classList.add("show");
}
document.getElementById("infoClose").addEventListener("click", () => infoModal.classList.remove("show"));

const addSignModal = document.getElementById("addSignModal");
document.getElementById("addSignCancel").addEventListener("click", () => addSignModal.classList.remove("show"));
document.getElementById("addSignOk").addEventListener("click", () => {
  const name = document.getElementById("addSignInput").value.trim().toUpperCase();
  if (!name) return;
  if (!trainableLabels.includes(name)) trainableLabels.push(name);
  refreshLabelSelect();
  labelSelect.value = name;
  addSignModal.classList.remove("show");
  toast(`Now record samples for ${name}`);
});

const menuModal = document.getElementById("menuModal");
document.getElementById("menuBtn").addEventListener("click", () => {
  const items = [
    ["Sign guide", () => { menuModal.classList.remove("show"); document.getElementById("guideBtn").click(); }],
    ["Clear sentence", () => { sentence = ""; renderSentence(); menuModal.classList.remove("show"); }],
    ["Which signs are learned?", () => {
      const learned = classifier.trainedLabels();
      menuModal.classList.remove("show");
      showInfo(`Learned signs (${learned.length})`,
        learned.length ? learned.join(", ") : `None yet. The app is using its ${BUILT_IN_LABELS.length} built-in signs.`);
    }],
  ];
  if (admin.loggedIn) {
    items.push(
      ["Add a new sign name", () => { menuModal.classList.remove("show"); document.getElementById("addSignInput").value = ""; addSignModal.classList.add("show"); }],
      ["Delete samples of selected sign", () => {
        classifier.clearLabel(labelSelect.value); updateStatus(); menuModal.classList.remove("show");
        toast(`Cleared ${labelSelect.value}`);
      }],
      ["Delete ALL learned data", () => {
        classifier.clearAll(); refreshLabelSelect(); updateStatus(); menuModal.classList.remove("show");
        toast("Learned data cleared");
      }],
      ["Change admin PIN", () => {
        menuModal.classList.remove("show");
        showPinModal("Change admin PIN", "Enter a new 4+ digit PIN.", (pin) => {
          if (pin.length < 4) { toast("PIN must be at least 4 digits"); return false; }
          admin.setPin(pin); toast("PIN updated"); return true;
        });
      }],
      ["Log out", () => { admin.loggedIn = false; setTrainingMode(false); updateStatus(); menuModal.classList.remove("show"); toast("Logged out"); }]
    );
  }
  document.getElementById("menuList").innerHTML = "";
  items.forEach(([label, fn]) => {
    const div = document.createElement("div");
    div.className = "menu-item"; div.textContent = label;
    div.addEventListener("click", fn);
    document.getElementById("menuList").appendChild(div);
  });
  menuModal.classList.add("show");
});
document.getElementById("menuClose").addEventListener("click", () => menuModal.classList.remove("show"));

setup();