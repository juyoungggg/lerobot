#!/bin/bash
# scenario7 웹 서버 실행 스크립트
# 사용법: ./start.sh
set -e

cd "$(dirname "$0")"

# ============================================================
# conda 환경 활성화
# ============================================================
source ~/miniconda3/etc/profile.d/conda.sh
conda activate web_test

# ============================================================
# 로봇 프로세스 설정 (main.py가 conda run으로 실행)
# ============================================================
export CONDA_EXE=${CONDA_EXE:-conda}
export LEROBOT_CONDA_ENV=${LEROBOT_CONDA_ENV:-project}
export LEROBOT_REPO_ROOT=${LEROBOT_REPO_ROOT:-$(cd .. && pwd)}
# 🔴 데몬 방식: main.py가 시작 시 lerobot_record_daemon.py를 자동으로 띄웁니다.
# LEROBOT_RECORD_SCRIPT는 더 이상 사용하지 않습니다.
export LEROBOT_ROBOT_COMMAND_ENABLED=${LEROBOT_ROBOT_COMMAND_ENABLED:-true}

# 🔴 HF Hub 온라인 요청 차단 (policy 로드 시 캐시에서만 읽음, 로딩 속도 대폭 향상)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ============================================================
# 카메라/포트 장치명
# ============================================================
export GLOBAL_CAM=${GLOBAL_CAM:-/dev/GLOBAL_CAM}
export RIGHT_TOP_CAM=${RIGHT_TOP_CAM:-/dev/RIGHT_TOP}
export RIGHT_WRIST_CAM=${RIGHT_WRIST_CAM:-/dev/RIGHT_WRIST}
export LEFT_WRIST_CAM=${LEFT_WRIST_CAM:-/dev/LEFT_WRIST}

export FOLLOWER_LEFT_PORT=${FOLLOWER_LEFT_PORT:-/dev/ttyACM_FOLLOWER}
export FOLLOWER_RIGHT_PORT=${FOLLOWER_RIGHT_PORT:-/dev/ttyACM_FOLLOWER_2}

# ============================================================
# 데이터셋 설정
# ============================================================
export LEROBOT_DATASET_REPO_ID=${LEROBOT_DATASET_REPO_ID:-juyoungggg/web_mode_run}
export LEROBOT_DATASET_NUM_EPISODES=${LEROBOT_DATASET_NUM_EPISODES:-1}
export LEROBOT_DATASET_PUSH_TO_HUB=${LEROBOT_DATASET_PUSH_TO_HUB:-false}

# ============================================================
# JSMpeg 영상 스트리밍
# ============================================================
export STREAM_WIDTH=${STREAM_WIDTH:-640}
export STREAM_HEIGHT=${STREAM_HEIGHT:-480}
export STREAM_CAPTURE_FPS=${STREAM_CAPTURE_FPS:-10}
export STREAM_OUTPUT_FPS=${STREAM_OUTPUT_FPS:-30}
export STREAM_BITRATE=${STREAM_BITRATE:-600k}
export STREAM_CAMERA_FORMAT=${STREAM_CAMERA_FORMAT:-mjpeg}
export FFMPEG_LOG_LEVEL=${FFMPEG_LOG_LEVEL:-warning}
export VIDEO_PAUSE_WHEN_STT_BUSY=${VIDEO_PAUSE_WHEN_STT_BUSY:-false}

# ============================================================
# STT (Whisper)
# ============================================================
export WHISPER_MODEL_ID=${WHISPER_MODEL_ID:-openai/whisper-tiny}

# ============================================================
# Ollama LLM 모드 분류
# Ollama가 다른 PC에서 돌고 있으면 IP를 바꾸세요.
# 같은 PC면 127.0.0.1 그대로 두면 됩니다.
# ============================================================
export OLLAMA_MODE_RECOGNITION_ENABLED=${OLLAMA_MODE_RECOGNITION_ENABLED:-true}
export OLLAMA_BASE_URL=${OLLAMA_BASE_URL:-http://127.0.0.1:11434}
export OLLAMA_MODE_MODEL=${OLLAMA_MODE_MODEL:-qwen2.5:1.5b}
export OLLAMA_MODE_TIMEOUT_S=${OLLAMA_MODE_TIMEOUT_S:-2}

# ============================================================
# TTS 설정
# web: 브라우저 스피커 / local: 서버 스피커 / both / none
# ============================================================
export LEROBOT_TTS_ENABLED=${LEROBOT_TTS_ENABLED:-true}
export LEROBOT_TTS_OUTPUT=${LEROBOT_TTS_OUTPUT:-web}

# ============================================================
# 서버 실행
# ============================================================
echo ""
echo "=========================================="
echo "  scenario_final 웹 서버 시작 (데몬 방식)"
echo "  http://127.0.0.1:8000"
echo ""
echo "  서버 시작 시 로봇 연결 + 모든 policy 미리 로드"
echo "  모드 선택 시 즉시 실행 (로드 대기 없음)"
echo ""
echo "  외부 접속: sudo tailscale funnel --bg --https=443 http://127.0.0.1:8000"
echo "  종료: Ctrl+C"
echo "=========================================="
echo ""

python -m uvicorn main:app --host 127.0.0.1 --port 8000 --log-level info
