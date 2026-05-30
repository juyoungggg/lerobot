# LeRobot Scenario 5 실행 명령어

## 1. 웹 서버 실행

```bash
cd lerobot_project/lerobot_scenario7


conda activate web_test

export CONDA_EXE=conda
export LEROBOT_CONDA_ENV=project
export LEROBOT_REPO_ROOT=/home/oran/lerobot_project
export LEROBOT_RECORD_SCRIPT=/home/oran/lerobot_project/src/lerobot/scripts/lerobot_record_web.py
export LEROBOT_ROBOT_COMMAND_ENABLED=true

export GLOBAL_CAM=/dev/GLOBAL_CAM
export RIGHT_TOP_CAM=/dev/RIGHT_TOP
export RIGHT_WRIST_CAM=/dev/RIGHT_WRIST
export LEFT_WRIST_CAM=/dev/LEFT_WRIST

export FOLLOWER_LEFT_PORT=/dev/ttyACM_FOLLOWER
export FOLLOWER_RIGHT_PORT=/dev/ttyACM_FOLLOWER_2

export LEROBOT_DATASET_REPO_ID=juyoungggg/web_mode_run
export LEROBOT_DATASET_NUM_EPISODES=1
export LEROBOT_DATASET_PUSH_TO_HUB=false

export STREAM_WIDTH=640
export STREAM_HEIGHT=480
export STREAM_CAPTURE_FPS=10
export STREAM_OUTPUT_FPS=30
export STREAM_BITRATE=600k
export STREAM_CAMERA_FORMAT=mjpeg
export FFMPEG_LOG_LEVEL=warning
export VIDEO_PAUSE_WHEN_STT_BUSY=false

export WHISPER_MODEL_ID=openai/whisper-tiny

# STT -> Ollama LLM -> 모드 결정
export OLLAMA_MODE_RECOGNITION_ENABLED=true
export OLLAMA_BASE_URL=http://OLLAMA_컴퓨터_IP:11434
export OLLAMA_MODE_MODEL=qwen2.5:1.5b
export OLLAMA_MODE_TIMEOUT_S=2

# 외부 HTTPS 접속자 브라우저에서 TTS 재생
export LEROBOT_TTS_ENABLED=true
export LEROBOT_TTS_OUTPUT=web

python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

## 2. Ollama 컴퓨터 설정

Ollama를 실행하는 다른 컴퓨터에서:

```bash
ollama pull qwen2.5:1.5b
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

웹 서버 컴퓨터에서 `OLLAMA_BASE_URL`은 Ollama 컴퓨터 IP로 맞춥니다.

```bash
export OLLAMA_BASE_URL=http://192.168.x.x:11434
```

## 3. 외부 HTTPS 열기

```bash
sudo tailscale funnel --bg --https=443 http://127.0.0.1:8000
```

브라우저에서는 `tailscale serve status`에 나온 `https://...ts.net` 주소로 접속합니다.

## 4. 지연이 클 때 권장값

Ollama 응답이 늦으면 fallback까지 오래 기다리므로 timeout을 낮춥니다.

```bash
export OLLAMA_MODE_TIMEOUT_S=2
```

원격 영상이 버벅이면 웹 서버 실행 전에 아래 값으로 낮춰서 실행합니다.

```bash
export STREAM_WIDTH=480
export STREAM_HEIGHT=360
export STREAM_CAPTURE_FPS=8
export STREAM_OUTPUT_FPS=15
export STREAM_BITRATE=350k
```

## 5. TTS 출력 위치

```bash
export LEROBOT_TTS_OUTPUT=web    # 외부 접속자 브라우저에서 재생
export LEROBOT_TTS_OUTPUT=both   # 브라우저 + 서버 로컬 스피커
export LEROBOT_TTS_OUTPUT=local  # 서버 로컬 스피커만
export LEROBOT_TTS_OUTPUT=none   # TTS 끄기
```
