const landingPage = document.getElementById("landingPage");
const loginPage = document.getElementById("loginPage");
const controlPage = document.getElementById("controlPage");
const joinButton = document.getElementById("joinButton");
const loginForm = document.getElementById("loginForm");
const loginId = document.getElementById("loginId");
const loginPassword = document.getElementById("loginPassword");
const loginError = document.getElementById("loginError");
const commandInput = document.getElementById("commandInput");
const logList = document.getElementById("logList");
const statusPanel = document.getElementById("statusPanel");
const statusText = document.getElementById("statusText");
const statusSpinner = document.getElementById("statusSpinner");
const statusActions = document.getElementById("statusActions");
const stopButton = document.getElementById("stopButton");
const logoutButton = document.getElementById("logoutButton");
const micButton = document.getElementById("micButton");
const sendButton = document.getElementById("sendButton");
const videoCanvas = document.getElementById("videoCanvas");
const setupOverlay = document.getElementById("setupOverlay");
const videoStatus = document.getElementById("videoStatus");
const scenarioModeText = document.getElementById("scenarioModeText");
const scenarioModeFlow = document.getElementById("scenarioModeFlow");

let isRecording = false;
let isRobotRunning = false;
let logSocket = null;
let mediaRecorder = null;
let audioChunks = [];
let pendingMode = null;
let jsmpegPlayer = null;
let videoDecodeCount = 0;
let inputLocked = false;
let ttsQueue = Promise.resolve();

const JSMPEG_SOURCES = [
  "/static/jsmpg.min.js",
  "https://cdnjs.cloudflare.com/ajax/libs/jsmpeg/0.2/jsmpg.min.js",
  "https://cdn.jsdelivr.net/npm/jsmpeg@0.2.1/jsmpeg.min.js",
  "https://cdn.jsdelivr.net/gh/phoboslab/jsmpeg@master/jsmpeg.min.js",
];

const SCENARIO_MODE_LABELS = {
  CLEANUP: "Organize",
  SETUP: "Setting",
  PACKING: "Packing",
};

const MODE_DISPLAY_NAMES = {
  CLEANUP: "정리 모드",
  SETUP: "세팅 모드",
  PACKING: "패킹 모드",
};

const SETUP_POLYGONS = [
  { label: "Battery", color: "#1e90ff", points: [[420, 195], [525, 195], [525, 240], [420, 240]] },
  { label: "Cup", color: "#ff2d2d", points: [[470, 270], [535, 270], [535, 330], [470, 330]] },
  { label: "Driver", color: "#ffd400", points: [[75, 185], [195, 185], [195, 230], [75, 230]] },
];
const SCENARIO_MODE_STEPS = {
  CLEANUP: [
    {
      src: "/static/media/cleannupp-1.mp4",
      label: "cleanup-1 영상",
      caption: "RL이 정리할 물건과 위치를 결정합니다.",
    },
    {
      src: "/static/media/cleanupp-2.mp4",
      label: "cleanup-2 영상",
      caption: "로봇이 RL Decision에 맞게 정리합니다.",
    },
  ],
  SETUP: [
    {
      src: "/static/media/setup-1.mp4",
      label: "setup-1 영상",
      caption: "드라이버를 올바른 위치에 세팅합니다.",
    },
    {
      src: "/static/media/setup-2.mp4",
      label: "setup-2 영상",
      caption: "컵과 배터리를 지정된 위치에 놓습니다.",
    },
  ],
  PACKING: [
    {
      src: "/static/media/packing-1.mp4",
      label: "packing-1 영상",
      caption: "arm1이 공구상자를 준비합니다.",
    },
    {
      src: "/static/media/packing-2.mp4",
      label: "packing-2 영상",
      caption: "arm2가 물건을 공구상자에 담습니다.",
    },
    {
      src: "/static/media/packing-3.mp4",
      label: "packing-3 영상",
      caption: "arm1이 패킹된 공구상자를 이동합니다.",
    },
  ],
};

function resizeSetupOverlayCanvas() {
  if (!setupOverlay) {
    return;
  }

  const width = videoCanvas?.width || 640;
  const height = videoCanvas?.height || 480;

  if (setupOverlay.width !== width || setupOverlay.height !== height) {
    setupOverlay.width = width;
    setupOverlay.height = height;
  }
}

function drawSetupOverlay(enabled) {
  if (!setupOverlay) {
    return;
  }

  resizeSetupOverlayCanvas();
  const ctx = setupOverlay.getContext("2d");
  ctx.clearRect(0, 0, setupOverlay.width, setupOverlay.height);

  if (!enabled) {
    return;
  }

  ctx.save();
  ctx.lineWidth = 3;
  ctx.font = "18px Arial";
  ctx.shadowColor = "rgba(0, 0, 0, 0.75)";
  ctx.shadowBlur = 4;

  for (const polygon of SETUP_POLYGONS) {
    ctx.strokeStyle = polygon.color;
    ctx.fillStyle = polygon.color;
    const [firstPoint, ...restPoints] = polygon.points;
    ctx.beginPath();
    ctx.moveTo(firstPoint[0], firstPoint[1]);
    for (const point of restPoints) {
      ctx.lineTo(point[0], point[1]);
    }
    ctx.closePath();
    ctx.stroke();
    ctx.fillText(polygon.label, firstPoint[0], Math.max(20, firstPoint[1] - 8));
  }

  ctx.restore();
}

function clearScenarioModeMedia() {
  if (!scenarioModeFlow) {
    return;
  }

  for (const video of scenarioModeFlow.querySelectorAll("video")) {
    video.pause();
    video.removeAttribute("src");
    video.load();
  }
  scenarioModeFlow.innerHTML = "";
  scenarioModeFlow.hidden = true;
}

function updateScenarioModeMedia(mode) {
  if (!scenarioModeFlow) {
    return;
  }

  const steps = SCENARIO_MODE_STEPS[mode] || [];
  clearScenarioModeMedia();

  if (steps.length === 0) {
    return;
  }

  steps.forEach((step, index) => {
    const row = document.createElement("div");
    row.className = "scenario-mode-step";

    const stepIndex = document.createElement("div");
    stepIndex.className = "scenario-step-index";
    stepIndex.textContent = String(index + 1);

    const body = document.createElement("div");
    body.className = "scenario-step-body";

    const mediaBox = document.createElement("div");
    mediaBox.className = "scenario-step-media-box";

    const video = document.createElement("video");
    video.className = "scenario-step-media";
    video.autoplay = true;
    video.muted = true;
    video.loop = true;
    video.playsInline = true;
    video.src = step.src;
    video.onerror = () => {
      video.remove();
      const fallback = document.createElement("div");
      fallback.className = "scenario-step-missing";
      fallback.textContent = `${step.label} 파일 없음`;
      mediaBox.appendChild(fallback);
    };
    mediaBox.appendChild(video);

    const caption = document.createElement("div");
    caption.className = "scenario-step-caption";
    caption.textContent = step.caption;

    body.appendChild(mediaBox);
    body.appendChild(caption);
    row.appendChild(stepIndex);
    row.appendChild(body);
    scenarioModeFlow.appendChild(row);

    video.play().catch(() => {});
  });

  scenarioModeFlow.hidden = false;
}

function setScenarioMode(mode) {
  if (!scenarioModeText) {
    return;
  }

  scenarioModeText.textContent = SCENARIO_MODE_LABELS[mode] || "None";
  updateScenarioModeMedia(mode);
  drawSetupOverlay(mode === "SETUP");
}

function syncScenarioModeFromStatus(text) {
  if (!text) {
    return;
  }

  if (text === "무슨 모드를 실행할까요?") {
    setScenarioMode(null);
    return;
  }

  if (text.includes("정리 모드") || text.includes("정리모드")) {
    setScenarioMode("CLEANUP");
    return;
  }

  if (text.includes("세팅 모드") || text.includes("세팅모드")) {
    setScenarioMode("SETUP");
    return;
  }

  if (text.includes("패킹 모드") || text.includes("패킹모드")) {
    setScenarioMode("PACKING");
  }
}

function setVideoStatus(message) {
  if (videoStatus) {
    videoStatus.textContent = message || "";
  }
}

function playTts(text) {
  if (!text) {
    return;
  }

  ttsQueue = ttsQueue
    .then(() => playTtsNow(text))
    .catch((error) => {
      console.warn("TTS playback failed:", error);
    });
}

function playTtsNow(text) {
  return new Promise((resolve, reject) => {
    const audio = new Audio(`/tts?text=${encodeURIComponent(text)}&ts=${Date.now()}`);
    audio.preload = "auto";
    audio.onended = resolve;
    audio.onerror = () => reject(new Error("브라우저 TTS 오디오를 재생하지 못했습니다."));
    audio.play().catch(reject);
  });
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[data-src="${src}"]`);
    if (existing) {
      existing.addEventListener("load", resolve, { once: true });
      existing.addEventListener("error", reject, { once: true });
      return;
    }

    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.dataset.src = src;
    script.onload = resolve;
    script.onerror = () => reject(new Error(`JSMpeg 로드 실패: ${src}`));
    document.head.appendChild(script);
  });
}

async function ensureJSMpegLoaded() {
  if (window.JSMpeg) {
    return;
  }

  const errors = [];
  for (const src of JSMPEG_SOURCES) {
    try {
      await loadScript(src);
      if (window.JSMpeg) {
        console.info(`JSMpeg loaded from ${src}`);
        return;
      }
      errors.push(`${src}: window.JSMpeg 없음`);
    } catch (error) {
      errors.push(String(error.message || error));
    }
  }

  throw new Error(`JSMpeg 라이브러리를 불러오지 못했습니다. ${errors.join(" / ")}`);
}

function goToControlPage() {
  landingPage.classList.remove("active");
  loginPage.classList.remove("active");
  controlPage.classList.add("active");

  connectLogSocket();
  startJSMpegStream();
}

function goToLoginPage() {
  if (!landingPage || !loginPage || !controlPage) {
    console.error("Page elements are missing; cannot open login page.");
    return;
  }

  landingPage.classList.remove("active");
  controlPage.classList.remove("active");
  loginPage.classList.add("active");
  if (loginError) {
    loginError.textContent = "";
  }
  window.setTimeout(() => loginId?.focus(), 0);
}

function goToLandingPage() {
  loginPage.classList.remove("active");
  controlPage.classList.remove("active");
  landingPage.classList.add("active");
}

async function checkAuthStatus() {
  try {
    const response = await fetch("/auth_status");
    const result = await response.json();
    if (result.authenticated) {
      goToControlPage();
    }
  } catch (_) {
  }
}

async function submitLogin(event) {
  event.preventDefault();
  if (loginError) {
    loginError.textContent = "";
  }

  try {
    const response = await fetch("/login", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_id: loginId.value.trim(),
        password: loginPassword.value,
      }),
    });
    const result = await response.json();

    if (result.status !== "ok") {
      if (loginError) {
        loginError.textContent = result.message || "로그인에 실패했습니다.";
      }
      return;
    }

    loginPassword.value = "";
    goToControlPage();
  } catch (error) {
    if (loginError) {
      loginError.textContent = `로그인 요청 중 오류가 발생했습니다: ${error}`;
    }
  }
}

async function logout() {
  try {
    await fetch("/logout", { method: "POST" });
  } catch (_) {
  }

  if (logSocket) {
    logSocket.close();
    logSocket = null;
  }

  if (jsmpegPlayer) {
    try {
      jsmpegPlayer.destroy();
    } catch (_) {
    }
    jsmpegPlayer = null;
  }

  clearLog();
  setScenarioMode(null);
  setVideoStatus("대기 중");
  setControlsDisabled(false);
  updateStatusPanel({
    text: "무슨 모드를 실행할까요?",
    loading: false,
    showStopButton: false,
  });
  goToLandingPage();
}

function clearLog() {
  logList.innerHTML = "";
}

function appendLog(text, options = {}) {
  const item = document.createElement("div");
  item.className = "log-item robot";

  if (options.loading) {
    const spinner = document.createElement("span");
    spinner.className = "loading-spinner";
    item.appendChild(spinner);
  }

  const textNode = document.createElement("span");
  textNode.textContent = text;
  item.appendChild(textNode);

  if (options.actions) {
    item.classList.add("has-actions");
    const actions = document.createElement("div");
    actions.className = "log-actions";

    for (const action of options.actions) {
      const button = document.createElement("button");
      button.className = "choice-button";
      button.textContent = action.label;
      button.onclick = action.onClick;
      actions.appendChild(button);
    }

    item.appendChild(actions);
  }

  logList.appendChild(item);
  logList.scrollTop = logList.scrollHeight;
  return item;
}

function setControlsDisabled(disabled) {
  inputLocked = disabled;
  isRobotRunning = disabled;
  commandInput.disabled = disabled;
  sendButton.disabled = disabled;
  micButton.disabled = disabled;
}

function clearStatusActions() {
  if (!statusActions) return;

  statusActions.innerHTML = "";
  statusActions.hidden = true;
}

function updateStatusPanel({ text, loading = false, showStopButton = false, actions = null }) {
  statusText.textContent = text || "무슨 모드를 실행할까요?";
  statusSpinner.hidden = !loading;
  stopButton.hidden = !showStopButton;

  clearStatusActions();
  if (actions && statusActions) {
    for (const action of actions) {
      const button = document.createElement("button");
      button.className = "status-choice-button";
      button.textContent = action.label;
      button.onclick = action.onClick;
      statusActions.appendChild(button);
    }
    statusActions.hidden = false;
  }
}

function handleEnter(event) {
  if (event.key === "Enter") {
    sendCommand();
  }
}

function getWebSocketBaseUrl() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${window.location.host}`;
}

async function startJSMpegStream() {
  try {
    setVideoStatus("영상 플레이어 준비 중...");
    await ensureJSMpegLoaded();

    if (jsmpegPlayer) {
      try {
        jsmpegPlayer.destroy();
      } catch (_) {
      }
      jsmpegPlayer = null;
    }

    const wsUrl = `${getWebSocketBaseUrl()}/ws/video?ts=${Date.now()}`;
    console.info(`JSMpeg video WebSocket URL: ${wsUrl}`);
    videoDecodeCount = 0;
    setVideoStatus("영상 연결 중...");

    jsmpegPlayer = new window.JSMpeg.Player(wsUrl, {
      canvas: videoCanvas,
      autoplay: true,
      audio: false,
      loop: false,
      videoBufferSize: 1024 * 1024,
      preserveDrawingBuffer: true,
      disableGl: true,
      onSourceEstablished: () => {
        setVideoStatus("영상 데이터 수신 중...");
      },
      onVideoDecode: () => {
        videoDecodeCount += 1;
        if (videoDecodeCount === 1) {
          console.info("JSMpeg first video frame decoded");
          setVideoStatus("영상 표시 중");
        }
      },
      onSourceCompleted: () => {
        setVideoStatus("영상 연결이 종료되었습니다.");
      },
      onStalled: () => {
        setVideoStatus("영상 데이터 대기 중...");
      },
    });

    window.setTimeout(() => {
      if (videoDecodeCount === 0 && videoStatus) {
        if (videoStatus.textContent === "영상 연결 중...") {
          setVideoStatus("영상 WebSocket 연결 대기 중...");
        } else if (videoStatus.textContent === "영상 데이터 수신 중...") {
          setVideoStatus("영상 데이터는 받았지만 아직 디코딩된 프레임이 없습니다.");
        }
      }
    }, 2500);
  } catch (error) {
    const message = `영상 연결 오류: ${error.message || error}`;
    setVideoStatus(message);
    appendLog(message);
    console.error(error);
  }
}

function connectLogSocket() {
  if (logSocket && logSocket.readyState === WebSocket.OPEN) {
    return;
  }

  const wsUrl = `${getWebSocketBaseUrl()}/ws/log`;
  logSocket = new WebSocket(wsUrl);

  logSocket.onopen = () => {
    updateStatusPanel({
      text: "무슨 모드를 실행할까요?",
      loading: false,
      showStopButton: false,
    });
  };

  logSocket.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.event === "setup_overlay") {
      drawSetupOverlay(data.enabled);
      return;
    }

    if (data.event === "robot_state") {
      setControlsDisabled(Boolean(data.running));
      return;
    }

    if (data.event === "scenario_mode") {
      setScenarioMode(data.mode);
      return;
    }

    if (data.event === "tts") {
      playTts(data.text || "");
      return;
    }

    if (data.event === "status") {
      syncScenarioModeFromStatus(data.text);
      updateStatusPanel({
        text: data.text,
        loading: Boolean(data.loading),
        showStopButton: Boolean(data.showStopButton),
      });
      return;
    }

    if (data.event === "input_lock") {
      setControlsDisabled(Boolean(data.locked));
      return;
    }

    if (data.event === "command_log") {
      appendLog(data.text || data.message || "");
      return;
    }

    if (data.message) {
      appendLog(data.message);
    }
  };

  logSocket.onerror = () => {
    appendLog("Command log WebSocket 오류가 발생했습니다.");
  };

  logSocket.onclose = () => {
    logSocket = null;
  };
}

if (loginForm) {
  loginForm.addEventListener("submit", submitLogin);
}

if (joinButton) {
  joinButton.addEventListener("click", goToLoginPage);
}

window.goToLoginPage = goToLoginPage;

async function sendCommand() {
  if (inputLocked) return;

  const command = commandInput.value.trim();
  if (!command) return;

  commandInput.value = "";
  showRecognizingMode();

  try {
    const response = await fetch("/command", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ command }),
    });

    const result = await response.json();

    if (result.status !== "ok") {
      appendLog(result.message || "모드를 인식하지 못했습니다. 다시 말씀해 주세요.");
      updateStatusPanel({
        text: "무슨 모드를 실행할까요?",
        loading: false,
        showStopButton: false,
      });
      return;
    }

    showModeConfirmation(result.mode, result.mode_name);
  } catch (error) {
    appendLog(`명령 처리 중 오류가 발생했습니다: ${error}`);
    updateStatusPanel({
      text: "무슨 모드를 실행할까요?",
      loading: false,
      showStopButton: false,
    });
  }
}

async function toggleMic() {
  if (inputLocked) return;

  if (!isRecording) {
    await startRecording();
    return;
  }

  stopRecording();
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
      stream.getTracks().forEach((track) => track.stop());
      await sendAudioToSTT(audioBlob);
    };

    mediaRecorder.start();
    isRecording = true;
    micButton.classList.add("recording");
    appendLog("음성 녹음이 시작되었습니다.");
  } catch (error) {
    appendLog(`음성 녹음을 시작할 수 없습니다: ${error}`);
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }

  isRecording = false;
  micButton.classList.remove("recording");
  showRecognizingMode();
}

function showRecognizingMode() {
  updateStatusPanel({
    text: "모드 인식 중",
    loading: true,
    showStopButton: false,
  });
}

function showModeConfirmation(mode, modeName) {
  pendingMode = mode;
  const displayName = MODE_DISPLAY_NAMES[mode] || modeName;
  updateStatusPanel({
    text: `${displayName}를 실행할까요?`,
    loading: false,
    showStopButton: false,
    actions: [
      {
        label: "예",
        onClick: confirmModeExecution,
      },
      {
        label: "아니오",
        onClick: cancelModeExecution,
      },
    ],
  });
}

function cancelModeExecution() {
  pendingMode = null;
  setScenarioMode(null);
  updateStatusPanel({
    text: "무슨 모드를 실행할까요?",
    loading: false,
    showStopButton: false,
  });
}

async function confirmModeExecution() {
  if (!pendingMode || inputLocked) return;

  const mode = pendingMode;
  pendingMode = null;
  setControlsDisabled(true);
  setScenarioMode(mode);
  updateStatusPanel({
    text: `${MODE_DISPLAY_NAMES[mode] || "모드"} 준비 중 ...`,
    loading: true,
    showStopButton: true,
  });

  try {
    const response = await fetch("/execute_mode", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ mode }),
    });

    const result = await response.json();

    if (result.status === "ok") {
      setScenarioMode(result.mode || mode);
      return;
    }

    appendLog(result.message || "모드 실행을 시작하지 못했습니다.");
    setControlsDisabled(false);
    setScenarioMode(null);
    updateStatusPanel({
      text: "무슨 모드를 실행할까요?",
      loading: false,
      showStopButton: false,
    });
  } catch (error) {
    appendLog(`모드 실행 요청 중 오류가 발생했습니다: ${error}`);
    setControlsDisabled(false);
    setScenarioMode(null);
    updateStatusPanel({
      text: "무슨 모드를 실행할까요?",
      loading: false,
      showStopButton: false,
    });
  }
}

async function stopMode() {
  try {
    stopButton.disabled = true;
    await fetch("/stop_mode", { method: "POST" });
  } catch (error) {
    appendLog(`모드 중지 요청 중 오류가 발생했습니다: ${error}`);
  } finally {
    stopButton.disabled = false;
  }
}

async function sendAudioToSTT(audioBlob) {
  const formData = new FormData();
  formData.append("audio", audioBlob, "voice.webm");

  try {
    const response = await fetch("/stt", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();

    if (result.status !== "ok") {
      appendLog(result.message || "모드를 인식하지 못했습니다. 다시 말씀해 주세요.");
      updateStatusPanel({
        text: "무슨 모드를 실행할까요?",
        loading: false,
        showStopButton: false,
      });
      return;
    }

    if (result.auto_command) {
      showModeConfirmation(result.mode, result.mode_name);
    } else {
      appendLog("모드를 인식하지 못했습니다. 다시 말씀해 주세요.");
      updateStatusPanel({
        text: "무슨 모드를 실행할까요?",
        loading: false,
        showStopButton: false,
      });
    }
  } catch (error) {
    appendLog(`음성 인식 요청 중 오류가 발생했습니다: ${error}`);
    updateStatusPanel({
      text: "무슨 모드를 실행할까요?",
      loading: false,
      showStopButton: false,
    });
  }
}

window.addEventListener("beforeunload", () => {
  if (jsmpegPlayer) {
    try {
      jsmpegPlayer.destroy();
    } catch (_) {
    }
  }
});

checkAuthStatus();
