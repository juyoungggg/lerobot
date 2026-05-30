import asyncio
import io
import json
import os
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Set
from collections import deque

import av
import edge_tts
import numpy as np
import torch
from fastapi import FastAPI, File, Query, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import logging as transformers_logging

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

log_clients: Set[WebSocket] = set()
video_clients: Set[WebSocket] = set()

# ============================================================
# Shared JSMpeg video broadcaster
# ============================================================
# 여러 브라우저가 동시에 /ws/video에 접속해도 카메라는 한 번만 엽니다.
# FFmpeg 프로세스 1개가 /dev/GLOBAL_CAM을 읽고, stdout chunk를 모든 WebSocket에게 복사합니다.
video_process: asyncio.subprocess.Process | None = None
video_stderr_task: asyncio.Task | None = None
video_broadcast_task: asyncio.Task | None = None
video_lock = asyncio.Lock()
video_client_queues: Dict[WebSocket, asyncio.Queue] = {}
video_recent_chunks = deque(maxlen=int(os.getenv("STREAM_REPLAY_CHUNKS", "180")))

robot_running = False
# 현재 실행 중인 로봇 제어 프로세스입니다.
# 🔴 데몬 방식: 서버 시작 시 한 번만 띄우고, stdin으로 MODE:X/STOP/QUIT 명령을 보냅니다.
robot_process = None
robot_process_mode = None
robot_output_task = None
robot_stop_requested = False
robot_start_cancel_requested = False
robot_motion_started = False
robot_daemon_ready = False  # 🔴 데몬이 READY 상태인지
process_transition_lock = asyncio.Lock()
current_mode = "IDLE"
current_mode_name = "대기모드"
current_task_prompt = ""

LOGIN_ID = os.getenv("LEROBOT_LOGIN_ID", "teleop150")
LOGIN_PASSWORD = os.getenv("LEROBOT_LOGIN_PASSWORD", "1234")
AUTH_COOKIE_NAME = "lerobot_auth"
AUTH_TOKEN = secrets.token_urlsafe(32)

def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


# ============================================================
# 5080 Ubuntu 환경 설정값
# 실제 5080 컴퓨터 값으로 바꿔야 하는 부분입니다.
# ============================================================

# 웹사이트 JSMpeg 스트리밍용 카메라입니다.
# 숫자 인덱스 대신 /dev/GLOBAL_CAM 같은 symlink를 사용합니다.
GLOBAL_CAM = os.getenv("GLOBAL_CAM", "/dev/GLOBAL_CAM")

# 실제 로봇 명령 실행 여부입니다.
# 안전을 위해 기본값은 false입니다. 실제 제어 테스트 시 true로 켜세요.
ROBOT_COMMAND_ENABLED = env_bool("LEROBOT_ROBOT_COMMAND_ENABLED", False)

# 웹에서 선택된 모드를 실행할 lerobot_record_web_jsmpeg.py입니다.
# 기본값은 이 웹 폴더 안의 lerobot_record_web_jsmpeg.py입니다.
# 다른 위치를 쓰려면 run_server.sh 또는 .env에서 LEROBOT_RECORD_SCRIPT를 바꾸면 됩니다.
DEFAULT_LEROBOT_REPO_ROOT = os.getenv("LEROBOT_REPO_ROOT", "/home/oran/lerobot")
DEFAULT_RECORD_SCRIPT = BASE_DIR / "lerobot_record_web_jsmpeg.py"
LEROBOT_RECORD_SCRIPT = Path(os.getenv("LEROBOT_RECORD_SCRIPT", str(DEFAULT_RECORD_SCRIPT))).expanduser()

# web_test에서 서버를 켜도 로봇 실행만 lerobot 가상환경으로 보내기 위한 설정입니다.
CONDA_EXE = os.getenv("CONDA_EXE", "conda")
LEROBOT_CONDA_ENV = os.getenv("LEROBOT_CONDA_ENV", "lerobot")
LEROBOT_REPO_ROOT = Path(os.getenv("LEROBOT_REPO_ROOT", DEFAULT_LEROBOT_REPO_ROOT)).expanduser()

 # 로봇/카메라 장치명입니다.
FOLLOWER_LEFT_PORT = os.getenv("FOLLOWER_LEFT_PORT", "/dev/ttyACM_FOLLOWER")
FOLLOWER_RIGHT_PORT = os.getenv("FOLLOWER_RIGHT_PORT", "/dev/ttyACM_FOLLOWER_2")
LEADER_LEFT_PORT = os.getenv("LEADER_LEFT_PORT", "/dev/ttyACM_LEADER")
LEADER_RIGHT_PORT = os.getenv("LEADER_RIGHT_PORT", "/dev/ttyACM_LEADER_2")

LEFT_WRIST_CAM = os.getenv("LEFT_WRIST_CAM", "/dev/LEFT_WRIST")
RIGHT_WRIST_CAM = os.getenv("RIGHT_WRIST_CAM", "/dev/RIGHT_WRIST")
RIGHT_TOP_CAM = os.getenv("RIGHT_TOP_CAM", "/dev/RIGHT_TOP")

ROBOT_ID = os.getenv("ROBOT_ID", "bimanual_follower")
TELEOP_ID = os.getenv("TELEOP_ID", "bimanual_leader")

# lerobot_record_web.py 실행용 데이터셋/녹화 기본값입니다.
# 실제 repo_id는 반드시 본인 값으로 바꿔서 export 하세요.
DATASET_REPO_ID = os.getenv("LEROBOT_DATASET_REPO_ID", "juyoungggg/web_mode_run")
DATASET_UNIQUE_PER_RUN = env_bool("LEROBOT_DATASET_UNIQUE_PER_RUN", True)
DATASET_ROOT = os.getenv("LEROBOT_DATASET_ROOT", "")
DATASET_NUM_EPISODES = os.getenv("LEROBOT_DATASET_NUM_EPISODES", "1")
DATASET_EPISODE_TIME_S = os.getenv("LEROBOT_DATASET_EPISODE_TIME_S", "80")
DATASET_RESET_TIME_S = os.getenv("LEROBOT_DATASET_RESET_TIME_S", "5")
DATASET_FPS = os.getenv("LEROBOT_DATASET_FPS", "30")
DATASET_PUSH_TO_HUB = os.getenv("LEROBOT_DATASET_PUSH_TO_HUB", "false")
DATASET_STREAMING_ENCODING = os.getenv("LEROBOT_DATASET_STREAMING_ENCODING", "true")
DATASET_ENCODER_THREADS = os.getenv("LEROBOT_DATASET_ENCODER_THREADS", "2")

# ============================================================
# JSMpeg 영상 스트리밍 설정값
# Tailscale Funnel의 HTTPS/WSS 경로로 그대로 흘릴 수 있도록 WebSocket을 사용합니다.
# ============================================================

STREAM_WIDTH = int(os.getenv("STREAM_WIDTH", "640"))
STREAM_HEIGHT = int(os.getenv("STREAM_HEIGHT", "480"))
STREAM_CAPTURE_FPS = int(os.getenv("STREAM_CAPTURE_FPS", "10"))
STREAM_OUTPUT_FPS = int(os.getenv("STREAM_OUTPUT_FPS", "30"))
STREAM_BITRATE = os.getenv("STREAM_BITRATE", "600k")
STREAM_CAMERA_FORMAT = os.getenv("STREAM_CAMERA_FORMAT", "mjpeg").strip().lower()
STREAM_READ_CHUNK_SIZE = int(os.getenv("STREAM_READ_CHUNK_SIZE", "1316"))
VIDEO_PAUSE_WHEN_STT_BUSY = env_bool("VIDEO_PAUSE_WHEN_STT_BUSY", False)

# STT 중일 때 영상 송출을 잠깐 늦출 수 있습니다.
STT_BUSY = False
stt_lock = asyncio.Lock()

# ============================================================
# TTS 설정값
# 기본값은 외부 HTTPS 접속자의 웹 브라우저 스피커 재생입니다.
# LEROBOT_TTS_OUTPUT=web/local/both/none 으로 출력 위치를 고를 수 있습니다.
# LEROBOT_TTS_ENABLED=false로 끌 수 있습니다.
# ============================================================

TTS_ENABLED = env_bool("LEROBOT_TTS_ENABLED", True)
TTS_OUTPUT = os.getenv("LEROBOT_TTS_OUTPUT", "web").strip().lower()
TTS_VOICE = os.getenv("LEROBOT_TTS_VOICE", "ko-KR-SunHiNeural")
TTS_PLAYER = os.getenv(
    "LEROBOT_TTS_PLAYER",
    str(Path.home() / "miniconda3" / "envs" / "lerobot" / "bin" / "ffplay")
    if (Path.home() / "miniconda3" / "envs" / "lerobot" / "bin" / "ffplay").exists()
    else "ffplay",
)
tts_lock = asyncio.Lock()


def tts_output_enabled(target: str) -> bool:
    if not TTS_ENABLED:
        return False
    if TTS_OUTPUT in {"", "none", "off", "false", "0"}:
        return False
    if TTS_OUTPUT == "both":
        return True
    return TTS_OUTPUT == target

# ============================================================
# STT / Whisper 설정값
# 브라우저에서 녹음한 음성(webm)을 /stt로 보내면 5080에서 Whisper로 한국어 STT를 수행합니다.
# 모바일/외부 브라우저에서 마이크를 쓰려면 HTTPS가 필요하므로 Tailscale Funnel 또는 ngrok 사용을 권장합니다.
# ============================================================

WHISPER_MODEL_ID = os.getenv("WHISPER_MODEL_ID", "openai/whisper-tiny")
STT_SAMPLING_RATE = 16000
STT_VOLUME_THRESHOLD = 0.02

# STT/텍스트 명령 후 모드 결정을 Ollama LLM에 먼저 맡깁니다.
# Ollama가 꺼져 있거나 응답이 불안정하면 기존 키워드 판정으로 fallback합니다.
OLLAMA_MODE_RECOGNITION_ENABLED = env_bool("OLLAMA_MODE_RECOGNITION_ENABLED", True)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODE_MODEL = os.getenv("OLLAMA_MODE_MODEL", "qwen2.5:1.5b")
OLLAMA_MODE_TIMEOUT_S = float(os.getenv("OLLAMA_MODE_TIMEOUT_S", "8"))

stt_device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_processor = None
whisper_model = None



# ============================================================
# 모드별 SmolVLA task prompt
# 실제 학습 데이터의 dataset.single_task 문장과 최대한 비슷하게 맞추는 것이 좋습니다.
# ============================================================

TASK_PROMPTS = {
    "CLEANUP": "Organize the desk by moving objects to their proper places.",
    "SETUP": "Set up the desk by placing the required objects in the correct positions.",
    "PACKING": "Pack the objects into the toolbox.",
}


# ============================================================
# 웹 표시용 이벤트 전송
# ============================================================

def print_server_log(message: str):
    print(message, flush=True)


async def send_command_log(message: str):
    await broadcast_ui_event("command_log", text=message)


async def send_status_panel(text: str, loading: bool = False, show_stop_button: bool = False):
    await broadcast_ui_event(
        "status",
        text=text,
        loading=loading,
        showStopButton=show_stop_button,
    )


async def send_input_lock(locked: bool):
    await broadcast_ui_event("input_lock", locked=locked)


async def send_scenario_mode(mode: str):
    await broadcast_ui_event("scenario_mode", mode=mode)


async def broadcast_log(message: str):
    await send_command_log(message)


async def send_web_tts(text: str):
    if tts_output_enabled("web") and text:
        await broadcast_ui_event("tts", text=text)


async def broadcast_ui_event(event: str, **data):
    dead_clients = []

    payload = {
        "event": event,
        "time": time.strftime("%H:%M:%S"),
        **data,
    }

    for ws in list(log_clients):
        try:
            await ws.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            dead_clients.append(ws)

    for ws in dead_clients:
        log_clients.discard(ws)


def is_authenticated_request(request: Request) -> bool:
    return request.cookies.get(AUTH_COOKIE_NAME) == AUTH_TOKEN


def is_authenticated_websocket(websocket: WebSocket) -> bool:
    return websocket.cookies.get(AUTH_COOKIE_NAME) == AUTH_TOKEN


class LoginRequest(BaseModel):
    user_id: str
    password: str


# ============================================================
# 로컬 TTS 출력
# ============================================================

async def speak_local_tts(text: str):
    """
    서버가 실행 중인 컴퓨터의 로컬 스피커에서 TTS를 재생합니다.
    LEROBOT_TTS_OUTPUT=local 또는 both일 때만 동작합니다.
    """
    if not tts_output_enabled("local") or not text:
        if text:
            print_server_log("Local TTS disabled. Set LEROBOT_TTS_OUTPUT=local or both to enable local speaker playback.")
        return

    async with tts_lock:
        output_path = None
        try:
            print_server_log(f"TTS start: {text}")
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                output_path = f.name

            communicate = edge_tts.Communicate(text=text, voice=TTS_VOICE)
            await communicate.save(output_path)

            process = await asyncio.create_subprocess_exec(
                TTS_PLAYER,
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "quiet",
                output_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()
            if process.returncode != 0:
                decoded = stderr.decode(errors="replace").strip() if stderr else ""
                print_server_log(f"TTS player exited with code {process.returncode}: {decoded}")
            else:
                print_server_log("TTS done")

        except FileNotFoundError:
            print_server_log(f"TTS player를 찾지 못했습니다: {TTS_PLAYER}. ffmpeg/ffplay 설치를 확인해 주세요.")
        except Exception as e:
            print_server_log(f"TTS error: {e}")
        finally:
            if output_path and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass


async def tts_audio_stream(text: str):
    communicate = edge_tts.Communicate(text=text, voice=TTS_VOICE)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]


@app.get("/tts")
async def tts_endpoint(request: Request, text: str = Query(..., min_length=1, max_length=500)):
    if not is_authenticated_request(request):
        return Response("로그인이 필요합니다.", status_code=401)

    if not tts_output_enabled("web"):
        return Response("Web TTS is disabled.", status_code=404)

    return StreamingResponse(
        tts_audio_stream(text),
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-store"},
    )


async def announce_tts(text: str):
    await send_web_tts(text)
    if tts_output_enabled("local"):
        await speak_local_tts(text)


def speak_mode_started(mode_name: str):
    asyncio.create_task(announce_tts(f"{mode_name}를 실행하겠습니다."))


# ============================================================
# 웹 페이지
# ============================================================

@app.get("/")
async def index():
    html_path = BASE_DIR / "templates" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/auth_status")
async def auth_status(request: Request):
    return {"authenticated": is_authenticated_request(request)}


@app.post("/login")
async def login(request: LoginRequest, response: Response):
    if request.user_id != LOGIN_ID or request.password != LOGIN_PASSWORD:
        return {"status": "error", "message": "ID 또는 Password가 올바르지 않습니다."}

    response.set_cookie(
        AUTH_COOKIE_NAME,
        AUTH_TOKEN,
        httponly=True,
        samesite="lax",
        secure=env_bool("LEROBOT_AUTH_SECURE_COOKIE", False),
        max_age=60 * 60 * 12,
    )
    return {"status": "ok"}


@app.post("/logout")
async def logout(response: Response):
    response.delete_cookie(AUTH_COOKIE_NAME)
    return {"status": "ok"}


# ============================================================
# WebSocket 로그
# ============================================================

@app.websocket("/ws/log")
async def websocket_log(websocket: WebSocket):
    if not is_authenticated_websocket(websocket):
        print("[log] websocket rejected: not authenticated", flush=True)
        await websocket.close(code=1008)
        return

    await websocket.accept()
    log_clients.add(websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        log_clients.discard(websocket)


# ============================================================
# JSMpeg 영상 WebSocket
# ============================================================

def build_ffmpeg_command() -> list[str]:
    """
    GLOBAL_CAM을 FFmpeg가 직접 읽고, JSMpeg가 재생할 수 있는
    MPEG-TS / MPEG-1 video 스트림으로 stdout에 내보냅니다.
    """
    fmt = STREAM_CAMERA_FORMAT
    if fmt in {"", "none", "auto"}:
        input_format_args = []
    else:
        input_format_args = ["-input_format", fmt]

    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", os.getenv("FFMPEG_LOG_LEVEL", "warning"),
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-f", "v4l2",
        *input_format_args,
        "-framerate", str(STREAM_CAPTURE_FPS),
        "-video_size", f"{STREAM_WIDTH}x{STREAM_HEIGHT}",
        "-i", str(GLOBAL_CAM),
        "-an",
        "-f", "mpegts",
        "-codec:v", "mpeg1video",
        "-b:v", str(STREAM_BITRATE),
        "-maxrate", str(STREAM_BITRATE),
        "-r", str(STREAM_OUTPUT_FPS),
        "-g", str(max(1, STREAM_OUTPUT_FPS // 2)),
        "-bf", "0",
        "-muxdelay", "0.001",
        "-muxpreload", "0",
        "-mpegts_flags", "+resend_headers",
        "-",
    ]


async def drain_ffmpeg_stderr(process: asyncio.subprocess.Process):
    if process.stderr is None:
        return

    while True:
        line = await process.stderr.readline()
        if not line:
            break
        decoded = line.decode(errors="replace").rstrip()
        if decoded:
            print(f"[ffmpeg] {decoded}", flush=True)


async def stop_shared_video_stream(reason: str = ""):
    """공유 FFmpeg 스트림을 안전하게 종료합니다."""
    global video_process, video_stderr_task, video_broadcast_task

    process = video_process
    video_process = None

    if video_stderr_task is not None and not video_stderr_task.done():
        video_stderr_task.cancel()
    video_stderr_task = None

    # 이 함수를 broadcaster 자기 자신이 부를 수도 있으므로 자기 task는 cancel하지 않습니다.
    current_task = asyncio.current_task()
    if (
        video_broadcast_task is not None
        and video_broadcast_task is not current_task
        and not video_broadcast_task.done()
    ):
        video_broadcast_task.cancel()
    if video_broadcast_task is not current_task:
        video_broadcast_task = None

    if process is not None and process.returncode is None:
        print(f"[video] stopping shared ffmpeg stream: {reason}", flush=True)
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=2)
        except asyncio.TimeoutError:
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass
        except ProcessLookupError:
            pass
        except Exception as e:
            print(f"[video] ffmpeg 종료 중 오류: {e}", flush=True)


async def _queue_video_chunk(queue: asyncio.Queue, chunk: bytes):
    """느린 클라이언트 때문에 전체 스트림이 멈추지 않도록 오래된 chunk를 버리고 최신 chunk를 넣습니다."""
    try:
        queue.put_nowait(chunk)
    except asyncio.QueueFull:
        try:
            queue.get_nowait()
            queue.task_done()
        except Exception:
            pass
        try:
            queue.put_nowait(chunk)
        except asyncio.QueueFull:
            pass


async def send_video_to_client(websocket: WebSocket, queue: asyncio.Queue):
    """클라이언트별 전송 루프입니다. WebSocket send를 클라이언트마다 분리해 한 명의 지연이 전체를 막지 않게 합니다."""
    try:
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            await websocket.send_bytes(chunk)
            queue.task_done()
    except Exception:
        pass
    finally:
        video_clients.discard(websocket)
        video_client_queues.pop(websocket, None)
        try:
            await websocket.close()
        except Exception:
            pass
        print(f"[video] client sender stopped. active={len(video_clients)}", flush=True)


async def broadcast_ffmpeg_stdout(process: asyncio.subprocess.Process):
    """FFmpeg stdout에서 읽은 MPEG-TS chunk를 모든 접속자의 queue로 broadcast합니다."""
    global video_broadcast_task

    if process.stdout is None:
        print("[video] ffmpeg stdout이 없습니다.", flush=True)
        return

    first_chunk_sent = False

    try:
        while True:
            if VIDEO_PAUSE_WHEN_STT_BUSY and STT_BUSY:
                await asyncio.sleep(0.15)
                continue

            chunk = await process.stdout.read(STREAM_READ_CHUNK_SIZE)
            if not chunk:
                return_code = await process.wait()
                print(f"[video] shared ffmpeg stopped with code {return_code}", flush=True)
                break

            video_recent_chunks.append(chunk)

            if not first_chunk_sent:
                first_chunk_sent = True
                print(f"[video] first shared chunk: {len(chunk)} bytes", flush=True)

            if not video_clients:
                await stop_shared_video_stream("no video clients")
                break

            for queue in list(video_client_queues.values()):
                await _queue_video_chunk(queue, chunk)

            if not video_clients:
                await stop_shared_video_stream("all video clients disconnected")
                break

    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"[video] broadcast error: {e}", flush=True)
    finally:
        if video_process is process:
            await stop_shared_video_stream("broadcast task ended")
        if video_broadcast_task is asyncio.current_task():
            video_broadcast_task = None

async def ensure_shared_video_stream_started():
    """첫 접속자 때만 FFmpeg를 실행하고, 이후 접속자는 같은 스트림을 공유합니다."""
    global video_process, video_stderr_task, video_broadcast_task

    async with video_lock:
        if video_process is not None and video_process.returncode is None:
            return

        command = build_ffmpeg_command()
        print("[video] starting shared ffmpeg stream", flush=True)
        print("[video] ffmpeg command:", " ".join(command), flush=True)

        try:
            video_process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
        except FileNotFoundError:
            print("[video] ffmpeg를 찾지 못했습니다. sudo apt install -y ffmpeg 를 실행해 주세요.", flush=True)
            raise

        video_stderr_task = asyncio.create_task(drain_ffmpeg_stderr(video_process))
        video_broadcast_task = asyncio.create_task(broadcast_ffmpeg_stdout(video_process))


@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    """
    브라우저 JSMpeg player로 MPEG-TS binary chunk를 전송합니다.
    여러 명이 접속해도 FFmpeg는 1개만 실행하고 같은 chunk를 각 클라이언트 queue로 broadcast합니다.
    """
    if not is_authenticated_websocket(websocket):
        print("[video] websocket rejected: not authenticated", flush=True)
        await websocket.close(code=1008)
        return

    await websocket.accept()

    queue: asyncio.Queue = asyncio.Queue(maxsize=int(os.getenv("VIDEO_CLIENT_QUEUE_SIZE", "45")))
    video_clients.add(websocket)
    video_client_queues[websocket] = queue
    sender_task = asyncio.create_task(send_video_to_client(websocket, queue))
    print(f"[video] client connected. active={len(video_clients)}", flush=True)

    try:
        await ensure_shared_video_stream_started()

        # 새로 접속한 브라우저가 MPEG 스트림의 중간에서 시작해 검은 화면이 되는 것을 줄이기 위해
        # 최근 chunk를 먼저 보내고 이후 live chunk를 이어 받게 합니다.
        for cached_chunk in list(video_recent_chunks):
            await _queue_video_chunk(queue, cached_chunk)

        await sender_task

    except FileNotFoundError:
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
    except Exception as e:
        print(f"[video] websocket error: {e}", flush=True)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        if not sender_task.done():
            sender_task.cancel()
        video_clients.discard(websocket)
        video_client_queues.pop(websocket, None)
        print(f"[video] client removed. active={len(video_clients)}", flush=True)
        if not video_clients:
            async with video_lock:
                if not video_clients:
                    await stop_shared_video_stream("last video client disconnected")

@app.get("/video_debug")
async def video_debug():
    return {
        "camera": str(GLOBAL_CAM),
        "width": STREAM_WIDTH,
        "height": STREAM_HEIGHT,
        "capture_fps": STREAM_CAPTURE_FPS,
        "output_fps": STREAM_OUTPUT_FPS,
        "bitrate": STREAM_BITRATE,
        "camera_format": STREAM_CAMERA_FORMAT,
        "active_video_clients": len(video_clients),
        "active_video_queues": len(video_client_queues),
        "cached_video_chunks": len(video_recent_chunks),
        "shared_ffmpeg_running": video_process is not None and video_process.returncode is None,
        "ffmpeg_command": build_ffmpeg_command(),
    }


# ============================================================
# STT / Whisper 처리
# ============================================================

def load_whisper_model_once():
    """
    Whisper 모델은 무거우므로 서버 시작 시가 아니라 첫 STT 요청 때 한 번만 로드합니다.
    첫 음성 인식은 모델 로드 때문에 시간이 오래 걸릴 수 있습니다.
    """
    global whisper_processor, whisper_model

    if whisper_processor is not None and whisper_model is not None:
        return whisper_processor, whisper_model

    transformers_logging.set_verbosity_error()

    whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_ID)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_MODEL_ID,
        torch_dtype=torch.float16 if stt_device == "cuda" else torch.float32,
    ).to(stt_device)
    whisper_model.eval()

    return whisper_processor, whisper_model


def decode_audio_to_16k_mono(audio_bytes: bytes) -> np.ndarray:
    """
    브라우저 MediaRecorder가 보낸 webm/opus 음성을 Whisper 입력용 16kHz mono float32로 변환합니다.
    """
    container = av.open(io.BytesIO(audio_bytes))

    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono",
        rate=STT_SAMPLING_RATE,
    )

    pcm_chunks = []

    for frame in container.decode(audio=0):
        resampled_frames = resampler.resample(frame)

        if not isinstance(resampled_frames, list):
            resampled_frames = [resampled_frames]

        for resampled in resampled_frames:
            array = resampled.to_ndarray()
            pcm_chunks.append(array.reshape(-1))

    if not pcm_chunks:
        return np.zeros(0, dtype=np.float32)

    pcm = np.concatenate(pcm_chunks).astype(np.float32)
    pcm = pcm / 32768.0

    return pcm


def transcribe_korean(audio_data: np.ndarray) -> str:
    if audio_data.size == 0:
        return ""

    volume_norm = float(np.linalg.norm(audio_data) * 10)

    if volume_norm < STT_VOLUME_THRESHOLD:
        return ""

    processor, model = load_whisper_model_once()

    input_features = processor(
        audio_data,
        sampling_rate=STT_SAMPLING_RATE,
        return_tensors="pt",
    ).input_features.to(stt_device)

    if stt_device == "cuda":
        input_features = input_features.to(torch.float16)

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language="ko",
            task="transcribe",
            no_repeat_ngram_size=2,
        )

    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
    )

    if not transcription:
        return ""

    return transcription[0].strip()


def process_stt_bytes(audio_bytes: bytes) -> str:
    audio_data = decode_audio_to_16k_mono(audio_bytes)
    return transcribe_korean(audio_data)


@app.post("/stt")
async def stt_endpoint(request: Request, audio: UploadFile = File(...)):
    """
    브라우저에서 녹음한 음성 파일을 받아 Whisper로 텍스트 변환합니다.
    무거운 디코딩/추론은 thread로 보내서 영상 WebSocket과 로그가 덜 막히게 합니다.
    """
    global STT_BUSY

    if not is_authenticated_request(request):
        return {"status": "error", "message": "로그인이 필요합니다.", "text": ""}

    try:
        audio_bytes = await audio.read()

        async with stt_lock:
            STT_BUSY = True
            try:
                text = await asyncio.to_thread(process_stt_bytes, audio_bytes)
            finally:
                STT_BUSY = False

        if not text:
            return {
                "status": "error",
                "message": "음성이 너무 작거나 인식 결과가 없습니다.",
                "text": "",
            }

        mode, recognition_source = await classify_mode(text)

        if mode == "UNKNOWN":
            return {
                "status": "ok",
                "text": text,
                "mode": "UNKNOWN",
                "mode_name": "알 수 없는 모드",
                "task_prompt": "",
                "recognition_source": recognition_source,
                "auto_command": False,
            }

        mode_name = get_mode_name(mode)
        task_prompt = build_task_prompt(mode)

        return {
            "status": "ok",
            "text": text,
            "mode": mode,
            "mode_name": mode_name,
            "task_prompt": task_prompt,
            "recognition_source": recognition_source,
            "auto_command": True,
        }

    except Exception as e:
        STT_BUSY = False
        print(f"STT error: {e}")
        return {
            "status": "error",
            "message": str(e),
            "text": "",
        }


# ============================================================
# 자연어 명령 처리
# ============================================================

class CommandRequest(BaseModel):
    command: str


class ExecuteModeRequest(BaseModel):
    mode: str


def normalize_command(command: str) -> str:
    return command.strip().lower().replace(" ", "")


def parse_mode(command: str) -> str:
    """
    입력 문장을 정리모드 / 세팅모드 / 패킹모드 중 하나로 분류합니다.

    중요:
    "공구상자 세팅해줘"처럼 공구상자와 세팅이 같이 들어간 문장은
    패킹이 아니라 세팅 의도가 더 강하므로 SETUP을 먼저 판정합니다.
    "공구상자에 넣어줘/담아줘"처럼 넣기 동사가 있으면 PACKING으로 갑니다.
    """
    text = normalize_command(command)

    setup_keywords = [
        "세팅", "셋팅", "준비", "책상세팅", "책상셋팅", "작업준비",
        "준비해줘", "꺼내줘", "배치", "배치해줘", "놓아줘",
        "책상위에놓아줘", "작업할수있게", "시작준비",
    ]

    packing_keywords = [
        "패킹", "포장", "짐싸", "짐싸줘", "담아", "담아줘", "넣어", "넣어줘",
        "상자에넣", "박스에넣", "공구함에넣", "공구상자에넣",
        "챙겨줘", "싸줘", "보관", "수납",
    ]

    cleanup_keywords = [
        "정리", "정돈", "치워", "치워줘", "책상정리", "책상좀정리",
        "내책상정리", "내책상정리좀", "책상치워", "책상치워줘",
        "깨끗하게", "청소", "어질러진거", "어질러진것", "원래자리", "제자리",
    ]

    # 세팅은 "공구상자 세팅"처럼 공구상자 단어와 같이 올 수 있으므로 먼저 잡습니다.
    if any(keyword in text for keyword in setup_keywords):
        return "SETUP"

    # 공구상자라는 명사만으로는 세팅/패킹이 갈릴 수 있으므로,
    # 넣기/담기/수납 같은 동작 의도가 있을 때만 패킹으로 봅니다.
    if any(keyword in text for keyword in packing_keywords):
        return "PACKING"

    if any(keyword in text for keyword in cleanup_keywords):
        return "CLEANUP"

    return "UNKNOWN"


def extract_mode_from_llm_text(text: str) -> str:
    normalized = text.strip().upper()

    if not normalized:
        return "UNKNOWN"

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            normalized = str(parsed.get("mode", "")).strip().upper()
        elif isinstance(parsed, str):
            normalized = parsed.strip().upper()
    except json.JSONDecodeError:
        pass

    valid_modes = {"CLEANUP", "SETUP", "PACKING", "UNKNOWN"}
    if normalized in valid_modes:
        return normalized

    for mode in valid_modes:
        if mode in normalized:
            return mode

    return "UNKNOWN"


def classify_mode_with_ollama_sync(command: str) -> str:
    prompt = f"""
You classify a Korean or English robot command into exactly one mode.

Modes:
- CLEANUP: organize, clean, tidy, return objects to proper places.
- SETUP: set up the desk, prepare objects, place required objects for work.
- PACKING: pack objects into a box/toolbox/container, put objects away into a box.
- UNKNOWN: not enough information or unrelated command.

Rules:
- Return only JSON, no markdown, no explanation.
- JSON schema: {{"mode":"CLEANUP|SETUP|PACKING|UNKNOWN"}}
- If the command says to put objects into a box/toolbox/container, choose PACKING.
- If the command says to set up or prepare the desk/workspace, choose SETUP.
- If the command says to clean/tidy/organize the desk, choose CLEANUP.

Command: {command}
""".strip()

    payload = {
        "model": OLLAMA_MODE_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 24,
        },
    }

    request = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=OLLAMA_MODE_TIMEOUT_S) as response:
        body = response.read().decode("utf-8", errors="replace")

    data = json.loads(body)
    return extract_mode_from_llm_text(str(data.get("response", "")))


async def classify_mode(command: str) -> tuple[str, str]:
    """
    모드 인식 흐름:
    STT/텍스트 -> Ollama(qwen2.5:1.5b) -> 모드 결정.
    Ollama가 실패하면 기존 키워드 기반 parse_mode로 fallback합니다.
    """
    if OLLAMA_MODE_RECOGNITION_ENABLED:
        try:
            mode = await asyncio.to_thread(classify_mode_with_ollama_sync, command)
            if mode in {"CLEANUP", "SETUP", "PACKING"}:
                return mode, "ollama"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
            print_server_log(f"[mode] Ollama mode recognition failed, fallback to keywords: {e}")
        except Exception as e:
            print_server_log(f"[mode] Unexpected Ollama mode recognition error, fallback to keywords: {e}")

    return parse_mode(command), "keyword"


def get_mode_name(mode: str) -> str:
    mode_name_map = {
        "CLEANUP": "정리모드",
        "SETUP": "세팅모드",
        "PACKING": "패킹모드",
        "UNKNOWN": "알 수 없는 모드",
    }
    return mode_name_map.get(mode, "알 수 없는 모드")


def get_mode_display_name(mode: str) -> str:
    mode_name_map = {
        "CLEANUP": "정리 모드",
        "SETUP": "세팅 모드",
        "PACKING": "패킹 모드",
    }
    return mode_name_map.get(mode, "알 수 없는 모드")


def get_mode_prepare_log(mode: str) -> str:
    mode_name_map = {
        "CLEANUP": "로봇이 정리모드를 준비중입니다.",
        "SETUP": "로봇이 세팅모드를 준비중입니다.",
        "PACKING": "로봇이 패킹모드를 준비중입니다.",
    }
    return mode_name_map.get(mode, "로봇이 모드를 준비중입니다.")


def get_mode_complete_status(mode: str) -> str:
    mode_name_map = {
        "CLEANUP": "정리를 완료했습니다.",
        "SETUP": "세팅을 완료했습니다.",
        "PACKING": "패킹을 완료했습니다.",
    }
    return mode_name_map.get(mode, "완료했습니다.")


def build_task_prompt(mode: str) -> str:
    return TASK_PROMPTS.get(mode, "Unknown command.")


def mode_to_record_value(mode: str) -> str:
    """lerobot_record_web.py 기준: 1=정리, 2=패킹, 3=세팅"""
    mode_input_map = {
        "CLEANUP": "1",
        "PACKING": "2",
        "SETUP": "3",
    }
    return mode_input_map[mode]


def build_run_dataset_repo_id(mode: str) -> str:
    """
    LeRobotDataset.create()는 같은 repo_id의 로컬 dataset/cache가 이미 있으면 실패합니다.
    웹 실행은 여러 모드를 연속으로 시험하는 흐름이므로 기본적으로 매 실행마다 repo_id를 다르게 만듭니다.
    """
    if not DATASET_UNIQUE_PER_RUN:
        return DATASET_REPO_ID

    mode_slug_map = {
        "CLEANUP": "cleanup",
        "SETUP": "setup",
        "PACKING": "packing",
    }
    mode_slug = mode_slug_map.get(mode, mode.lower())
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if "/" not in DATASET_REPO_ID:
        return f"{DATASET_REPO_ID}_{mode_slug}_{timestamp}"

    namespace, name = DATASET_REPO_ID.split("/", 1)
    return f"{namespace}/{name}_{mode_slug}_{timestamp}"


def build_record_command(mode: str, task_prompt: str) -> tuple[list[str], dict[str, str]]:
    left_camera_json = (
        "{"
        f'"wrist": {{"type": "opencv", "index_or_path": "{LEFT_WRIST_CAM}", "width": 640, "height": 480, "fps": 30}}'
        "}"
    )

    right_camera_items = [
        f'"wrist": {{"type": "opencv", "index_or_path": "{RIGHT_WRIST_CAM}", "width": 640, "height": 480, "fps": 30}}'
    ]

    if env_bool("LEROBOT_USE_RIGHT_TOP_IN_ROBOT", True):
        right_camera_items.append(
            f'"top": {{"type": "opencv", "index_or_path": "{RIGHT_TOP_CAM}", "width": 640, "height": 480, "fps": 30}}'
        )

    right_camera_json = "{" + ", ".join(right_camera_items) + "}"

    run_dataset_repo_id = build_run_dataset_repo_id(mode)

    command = [
        CONDA_EXE,
        "run",
        "--no-capture-output",
        "-n",
        LEROBOT_CONDA_ENV,
        "python",
        "-u",
        str(LEROBOT_RECORD_SCRIPT),
        "--play_sounds=false",
        "--robot.type=bi_so_follower",
        f"--robot.left_arm_config.port={FOLLOWER_LEFT_PORT}",
        f"--robot.right_arm_config.port={FOLLOWER_RIGHT_PORT}",
        f"--robot.id={ROBOT_ID}",
        f"--robot.left_arm_config.cameras={left_camera_json}",
        f"--robot.right_arm_config.cameras={right_camera_json}",
        f"--dataset.repo_id={run_dataset_repo_id}",
        f"--dataset.single_task={task_prompt}",
        f"--dataset.num_episodes={DATASET_NUM_EPISODES}",
        f"--dataset.episode_time_s={DATASET_EPISODE_TIME_S}",
        f"--dataset.reset_time_s={DATASET_RESET_TIME_S}",
        f"--dataset.fps={DATASET_FPS}",
        f"--dataset.push_to_hub={DATASET_PUSH_TO_HUB}",
        f"--dataset.streaming_encoding={DATASET_STREAMING_ENCODING}",
        f"--dataset.encoder_threads={DATASET_ENCODER_THREADS}",
        "--display_data=false",
    ]

    if DATASET_ROOT:
        command.append(f"--dataset.root={DATASET_ROOT}")

    child_env = os.environ.copy()
    child_env["LEROBOT_MODE_SELECT"] = mode_to_record_value(mode)
    child_env["LEROBOT_MODE_NAME"] = mode
    child_env["LEROBOT_RUN_DATASET_REPO_ID"] = run_dataset_repo_id

    # web_test 환경에서 uvicorn을 켜도, 자식 프로세스가 LeRobot repo를 찾도록 해 줍니다.
    repo_root_str = str(LEROBOT_REPO_ROOT)
    existing_pythonpath = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = repo_root_str if not existing_pythonpath else repo_root_str + os.pathsep + existing_pythonpath

    return command, child_env


async def reset_status_after_complete(mode: str):
    await send_status_panel(get_mode_complete_status(mode), loading=False, show_stop_button=False)
    await broadcast_ui_event("setup_overlay", enabled=False)
    await asyncio.sleep(3)
    await send_status_panel("무슨 모드를 실행할까요?", loading=False, show_stop_button=False)


async def monitor_robot_process(process: asyncio.subprocess.Process, mode: str):
    """🔴 데몬 방식: 데몬의 stdout을 계속 읽으면서 WEB_STATUS 메시지를 처리합니다."""
    global robot_running, robot_process_mode, robot_motion_started, robot_stop_requested, robot_daemon_ready

    completed_message_seen = False

    try:
        while True:
            if process.stdout is None:
                break
            raw = await process.stdout.readline()
            if not raw:
                # 프로세스가 종료됨
                print_server_log("[robot-daemon] stdout closed — daemon exited")
                robot_daemon_ready = False
                break

            decoded = raw.decode("utf-8", errors="replace").rstrip()
            if not decoded:
                continue

            if "WEB_STATUS:" in decoded:
                message = decoded.split("WEB_STATUS:", 1)[1].strip()

                if message == "READY":
                    robot_daemon_ready = True
                    print_server_log("[robot-daemon] READY — policies preloaded")
                    await send_status_panel("무슨 모드를 실행할까요?", loading=False, show_stop_button=False)
                elif message == "DONE":
                    if not robot_stop_requested:
                        if not completed_message_seen:
                            mode_name = get_mode_name(robot_process_mode or "")
                            await send_command_log(f"{mode_name}를 다 했어요.")
                        await reset_status_after_complete(robot_process_mode or mode)
                    robot_running = False
                    robot_process_mode = None
                    robot_motion_started = False
                    robot_stop_requested = False
                    completed_message_seen = False
                    await broadcast_ui_event("robot_state", running=False)
                    await send_input_lock(False)
                elif message == "STOPPED":
                    robot_running = False
                    robot_process_mode = None
                    robot_motion_started = False
                    robot_stop_requested = False
                    await broadcast_ui_event("robot_state", running=False)
                    await send_input_lock(False)
                elif "정리모드를 다 했어요" in message or "패킹모드를 다 했어요" in message or "세팅모드를 다 했어요" in message:
                    completed_message_seen = True
                    await send_command_log(message)
                    await send_web_tts(message)
                elif "정리하고 있어요" in message or "넣고 있어요" in message or "세팅하고 있어요" in message:
                    if not robot_motion_started:
                        robot_motion_started = True
                        mode_display = get_mode_display_name(robot_process_mode or mode)
                        await send_status_panel(f"{mode_display} 실행 중", loading=False, show_stop_button=True)
                    await send_command_log(message)
                    await send_web_tts(message)
                elif message.startswith("ERROR:"):
                    await send_command_log(message.replace("ERROR:", ""))
                else:
                    await send_command_log(message)
                    await send_web_tts(message)
            elif "WEB_RUNNING" in decoded:
                if not robot_motion_started:
                    robot_motion_started = True
                    mode_display = get_mode_display_name(robot_process_mode or mode)
                    await send_status_panel(f"{mode_display} 실행 중", loading=False, show_stop_button=True)
            elif "정책을 로딩하고 있습니다" in decoded:
                await send_status_panel("정책을 로딩하고 있습니다...", loading=True, show_stop_button=False)
            else:
                print_server_log(decoded)

    except asyncio.CancelledError:
        raise
    except Exception as e:
        print_server_log(f"[robot-daemon] stdout 모니터링 오류: {e}")


async def stop_current_robot_process(reason: str = ""):
    """🔴 데몬 방식: stdin에 STOP을 보내서 현재 모드를 중단합니다."""
    global robot_running, robot_process_mode, robot_stop_requested, robot_motion_started

    if not robot_running:
        robot_process_mode = None
        robot_motion_started = False
        return

    if reason:
        print_server_log(f"로봇 모드 중지 요청: {reason}")

    robot_stop_requested = True

    # 데몬 프로세스의 stdin에 STOP 명령 전송
    if robot_process is not None and robot_process.stdin is not None:
        try:
            robot_process.stdin.write(b"STOP\n")
            await robot_process.stdin.drain()
            print_server_log("[robot-daemon] STOP 명령 전송 완료")
        except Exception as e:
            print_server_log(f"[robot-daemon] STOP 전송 실패: {e}")

    # DONE 응답을 기다림 (최대 10초)
    for _ in range(100):
        if not robot_running:
            break
        await asyncio.sleep(0.1)

    robot_process_mode = None
    robot_running = False
    robot_stop_requested = False
    robot_motion_started = False
    await broadcast_ui_event("robot_state", running=False)
    await send_input_lock(False)


async def run_lerobot_command(mode: str, task_prompt: str):
    """
    🔴 데몬 방식: 이미 떠있는 데몬 프로세스의 stdin에 MODE:X를 보냅니다.
    데몬이 policy를 미리 로드해두었으므로 즉시 실행됩니다.
    """
    global robot_running, robot_process_mode, robot_stop_requested, robot_start_cancel_requested
    global current_mode, current_mode_name, current_task_prompt

    mode_name = get_mode_name(mode)
    current_mode = mode
    current_mode_name = mode_name
    current_task_prompt = task_prompt

    print_server_log(f"[robot-daemon] run_lerobot_command: mode={mode}, mode_name={mode_name}")

    async with process_transition_lock:
        # 이전 모드가 실행 중이면 먼저 중지
        if robot_running:
            await stop_current_robot_process(f"새 모드({mode_name})가 선택되었습니다.")

        if robot_start_cancel_requested:
            print_server_log("[robot-daemon] start cancelled before command send")
            robot_start_cancel_requested = False
            robot_stop_requested = False
            robot_running = False
            await broadcast_ui_event("robot_state", running=False)
            await send_input_lock(False)
            await send_status_panel("무슨 모드를 실행할까요?", loading=False, show_stop_button=False)
            return

        robot_stop_requested = False

        if not ROBOT_COMMAND_ENABLED:
            print_server_log("[robot-daemon] disabled: LEROBOT_ROBOT_COMMAND_ENABLED is false")
            await send_command_log("로봇 실행 기능이 꺼져 있습니다.")
            await send_status_panel("무슨 모드를 실행할까요?", loading=False, show_stop_button=False)
            await send_input_lock(False)
            await broadcast_ui_event("robot_state", running=False)
            return

        if not robot_daemon_ready:
            print_server_log("[robot-daemon] daemon not ready yet")
            await send_command_log("로봇 데몬이 아직 준비되지 않았습니다. 잠시 후 다시 시도해주세요.")
            await send_status_panel("무슨 모드를 실행할까요?", loading=False, show_stop_button=False)
            await send_input_lock(False)
            await broadcast_ui_event("robot_state", running=False)
            return

        if robot_process is None or robot_process.returncode is not None:
            print_server_log("[robot-daemon] daemon process is not running!")
            await send_command_log("로봇 데몬이 종료되었습니다. 서버를 재시작해주세요.")
            await send_status_panel("무슨 모드를 실행할까요?", loading=False, show_stop_button=False)
            await send_input_lock(False)
            await broadcast_ui_event("robot_state", running=False)
            return

        # 실제 "실행 중" 표시는 데몬이 WEB_RUNNING을 보낸 뒤에만 띄웁니다.
        # 이렇게 해야 "준비 중" -> "실행 중" -> 세부 작업 문장/TTS 순서가 안정적으로 유지됩니다.
        await send_status_panel(f"{get_mode_display_name(mode)} 준비 중 ...", loading=True, show_stop_button=True)

        # 데몬에 MODE 명령 전송
        mode_num = mode_to_record_value(mode)
        try:
            robot_process.stdin.write(f"MODE:{mode_num}\n".encode())
            await robot_process.stdin.drain()
            robot_process_mode = mode
            robot_running = True
            print_server_log(f"[robot-daemon] MODE:{mode_num} 명령 전송 완료")
            await broadcast_ui_event("robot_state", running=True)
        except Exception as e:
            print_server_log(f"[robot-daemon] MODE 명령 전송 실패: {e}")
            robot_running = False
            await send_command_log("로봇 명령 전송 중 오류가 발생했습니다.")
            await send_status_panel("무슨 모드를 실행할까요?", loading=False, show_stop_button=False)
            await send_input_lock(False)
            await broadcast_ui_event("robot_state", running=False)


@app.post("/command")
async def receive_command(http_request: Request, request: CommandRequest):
    if not is_authenticated_request(http_request):
        return {"status": "error", "message": "로그인이 필요합니다."}

    command = request.command

    mode, recognition_source = await classify_mode(command)

    if mode == "UNKNOWN":
        return {
            "status": "error",
            "mode": "UNKNOWN",
            "mode_name": "알 수 없는 모드",
            "recognition_source": recognition_source,
            "message": "알 수 없는 명령입니다. 예: 정리해줘, 책상 세팅해줘, 공구상자에 담아줘",
        }

    mode_name = get_mode_name(mode)
    task_prompt = build_task_prompt(mode)

    return {
        "status": "ok",
        "mode": mode,
        "mode_name": mode_name,
        "task_prompt": task_prompt,
        "recognition_source": recognition_source,
        "message": f"{mode_name}를 실행할까요?",
    }


@app.post("/execute_mode")
async def execute_mode(http_request: Request, request: ExecuteModeRequest):
    global robot_running, robot_start_cancel_requested

    if not is_authenticated_request(http_request):
        return {"status": "error", "message": "로그인이 필요합니다."}

    mode = request.mode
    print_server_log(f"[robot] /execute_mode requested: mode={mode}, robot_running={robot_running}")

    if robot_running:
        return {
            "status": "error",
            "message": "로봇이 이미 모드를 실행중입니다.",
        }

    if mode not in {"CLEANUP", "SETUP", "PACKING"}:
        return {
            "status": "error",
            "message": "알 수 없는 모드입니다.",
        }

    mode_name = get_mode_name(mode)

    task_prompt = build_task_prompt(mode)
    robot_running = True
    robot_start_cancel_requested = False
    speak_mode_started(mode_name)
    await send_input_lock(True)
    await broadcast_ui_event("robot_state", running=True)
    await send_scenario_mode(mode)
    if mode == "SETUP": #추가
        await broadcast_ui_event("setup_overlay", enabled=True)
    else:
        await broadcast_ui_event("setup_overlay", enabled=False)
    await send_command_log(get_mode_prepare_log(mode))
    await send_status_panel(f"{get_mode_display_name(mode)} 준비 중 ...", loading=True, show_stop_button=True)
    task = asyncio.create_task(run_lerobot_command(mode, task_prompt))
    task.add_done_callback(log_background_task_result)

    return {
        "status": "ok",
        "mode": mode,
        "mode_name": mode_name,
        "task_prompt": task_prompt,
        "message": get_mode_prepare_log(mode),
    }


@app.post("/stop_mode")
async def stop_mode(request: Request):
    global robot_start_cancel_requested

    if not is_authenticated_request(request):
        return {"status": "error", "message": "로그인이 필요합니다."}

    robot_start_cancel_requested = True

    await broadcast_ui_event("setup_overlay", enabled=False)

    stop_message = "모드를 종료합니다." if robot_motion_started else "모드 실행을 중지했습니다. 다시 말씀해 주세요."

    if not robot_running and robot_process is None:
        await send_command_log(stop_message)
        await send_status_panel("무슨 모드를 실행할까요?", loading=False, show_stop_button=False)
        await send_input_lock(False)
        return {"status": "ok", "message": stop_message}

    await stop_current_robot_process("웹에서 중지 버튼을 눌렀습니다.")
    await send_command_log(stop_message)
    await send_status_panel("무슨 모드를 실행할까요?", loading=False, show_stop_button=False)
    await send_input_lock(False)
    return {"status": "ok", "message": stop_message}


def log_background_task_result(task: asyncio.Task):
    try:
        task.result()
    except asyncio.CancelledError:
        print_server_log("[robot] background task cancelled")
    except Exception as e:
        print_server_log(f"[robot] background task failed: {e}")


# ============================================================
# 종료 처리
# ============================================================

@app.on_event("shutdown")
async def on_shutdown():
    """🔴 데몬 방식: QUIT 명령을 보내고 프로세스 종료를 기다립니다."""
    global robot_process
    if robot_process is not None and robot_process.returncode is None:
        try:
            robot_process.stdin.write(b"QUIT\n")
            await robot_process.stdin.drain()
            await asyncio.wait_for(robot_process.wait(), timeout=10)
        except Exception as e:
            print_server_log(f"[robot-daemon] QUIT 전송/종료 실패: {e}")
            try:
                robot_process.terminate()
                await robot_process.wait()
            except Exception:
                pass
    print_server_log("[robot-daemon] 데몬 프로세스 종료 완료")


@app.on_event("startup")
async def on_startup():
    """🔴 데몬 방식: 서버 시작 시 데몬 프로세스를 한 번 띄웁니다."""
    global robot_process, robot_output_task

    if not ROBOT_COMMAND_ENABLED:
        print_server_log("[robot-daemon] ROBOT_COMMAND_ENABLED=false, 데몬을 띄우지 않습니다.")
        return

    daemon_script = LEROBOT_REPO_ROOT / "src" / "lerobot" / "scripts" / "lerobot_record_daemon.py"
    if not daemon_script.exists():
        print_server_log(f"[robot-daemon] 데몬 스크립트를 찾을 수 없습니다: {daemon_script}")
        return

    # 데몬 실행 명령 조립 (기존 build_record_command와 유사하지만 LEROBOT_MODE_SELECT 없음)
    left_camera_json = (
        "{"
        f'"wrist": {{"type": "opencv", "index_or_path": "{LEFT_WRIST_CAM}", "width": 640, "height": 480, "fps": 30}}'
        "}"
    )
    right_camera_items = [
        f'"wrist": {{"type": "opencv", "index_or_path": "{RIGHT_WRIST_CAM}", "width": 640, "height": 480, "fps": 30}}'
    ]
    if env_bool("LEROBOT_USE_RIGHT_TOP_IN_ROBOT", True):
        right_camera_items.append(
            f'"top": {{"type": "opencv", "index_or_path": "{RIGHT_TOP_CAM}", "width": 640, "height": 480, "fps": 30}}'
        )
    right_camera_json = "{" + ", ".join(right_camera_items) + "}"

    command = [
        CONDA_EXE,
        "run",
        "--no-capture-output",
        "-n",
        LEROBOT_CONDA_ENV,
        "python",
        "-u",
        str(daemon_script),
        "--play_sounds=false",
        "--robot.type=bi_so_follower",
        f"--robot.left_arm_config.port={FOLLOWER_LEFT_PORT}",
        f"--robot.right_arm_config.port={FOLLOWER_RIGHT_PORT}",
        f"--robot.id={ROBOT_ID}",
        f"--robot.left_arm_config.cameras={left_camera_json}",
        f"--robot.right_arm_config.cameras={right_camera_json}",
        f"--dataset.repo_id={DATASET_REPO_ID}",
        f"--dataset.single_task=daemon",
        f"--dataset.num_episodes={DATASET_NUM_EPISODES}",
        f"--dataset.episode_time_s={DATASET_EPISODE_TIME_S}",
        f"--dataset.reset_time_s={DATASET_RESET_TIME_S}",
        f"--dataset.fps={DATASET_FPS}",
        f"--dataset.push_to_hub={DATASET_PUSH_TO_HUB}",
        f"--dataset.streaming_encoding={DATASET_STREAMING_ENCODING}",
        f"--dataset.encoder_threads={DATASET_ENCODER_THREADS}",
        "--display_data=false",
    ]

    child_env = os.environ.copy()
    child_env["LEROBOT_REPO_ROOT"] = str(LEROBOT_REPO_ROOT)
    repo_root_str = str(LEROBOT_REPO_ROOT)
    existing_pythonpath = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = repo_root_str if not existing_pythonpath else repo_root_str + os.pathsep + existing_pythonpath

    print_server_log("[robot-daemon] 데몬 프로세스 시작 중...")
    print_server_log("[robot-daemon] command: " + " ".join(command))

    try:
        robot_process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(LEROBOT_REPO_ROOT if LEROBOT_REPO_ROOT.exists() else BASE_DIR),
            env=child_env,
        )
        print_server_log(f"[robot-daemon] 프로세스 시작됨: pid={robot_process.pid}")

        # stdout 모니터링 시작 (READY 메시지를 기다림)
        robot_output_task = asyncio.create_task(monitor_robot_process(robot_process, "DAEMON"))

    except Exception as e:
        print_server_log(f"[robot-daemon] 데몬 시작 실패: {e}")
        robot_process = None
