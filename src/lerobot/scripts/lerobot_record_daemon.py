# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Daemon-style robot recording script.

Unlike lerobot_record_web.py which reads LEROBOT_MODE_SELECT once and exits,
this daemon:
  1. Starts up, connects robot, loads ALL policies (Mode 1, 2, 3) upfront
  2. Prints WEB_STATUS:READY to stdout when ready
  3. Enters a loop reading commands from stdin (one line at a time)
  4. When it receives "MODE:1", "MODE:2", or "MODE:3", executes that mode immediately
  5. When it receives "STOP", stops the current mode execution gracefully
  6. When it receives "QUIT", disconnects and exits
  7. After each mode completes, prints WEB_STATUS:DONE and goes back to waiting
"""

# 🔴 HF Hub/Transformers 온라인 요청 차단 (캐시에서만 로드, 로딩 속도 대폭 향상)
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 캐시 정리 경로
EVAL_CACHE_PATTERN = os.path.expanduser("~/.cache/huggingface/lerobot/juyoungggg/web_*")


# ============================================================
# 캐시 정리 함수
# ============================================================

def clean_eval_cache():
    # 실행 시 ~/.cache/huggingface/lerobot/juyoungggg/web_* 삭제
    targets = glob.glob(EVAL_CACHE_PATTERN)
    for path in targets:
        try:
            shutil.rmtree(path)
            print(f"  [캐시 삭제] {path}")
        except Exception as e:
            print(f"  [캐시 삭제 실패] {path} ({e})")
    if not targets:
        print("  [캐시] 정리할 항목 없음.")

import logging
import time
import select
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera.configuration_reachy2_camera import Reachy2CameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_so_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    so_leader,
    unitree_g1,
)
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# 🔴 추가: import
import numpy as np
import torch
import torch.nn as nn
import sys
import os
import re
import signal
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from lerobot.scripts.yolo_detect import detect_object, detected_to_obs
from lerobot.scripts.gym_env import DeskCleanEnv
from lerobot.scripts.RL_deploy import decide_action, update_bin_weights
from lerobot.scripts.tts import play_tts
import cv2

LEROBOT_REPO_ROOT = Path(os.environ.get("LEROBOT_REPO_ROOT", Path(BASE_DIR).parent))
MODEL_POLICIES_DIR = Path(
    os.environ.get(
        "LEROBOT_MODEL_POLICIES_DIR",
        LEROBOT_REPO_ROOT / "src" / "lerobot" / "model_policies",
    )
)

# 🔴 추가: 사용할 전역변수
# current bin state
bin_weights = {
    "first_drawer": 0.0,
    "second_drawer": 0.0,
    "gray_bin": 0.0,
    "white_bin": 0.0,
}

STOP_REQUESTED = False
CURRENT_EVENTS = None


def request_stop_from_signal(signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    if CURRENT_EVENTS is not None:
        CURRENT_EVENTS["stop_recording"] = True
        CURRENT_EVENTS["exit_early"] = True
    print("\nWEB_STATUS:모드를 종료합니다.", flush=True)


signal.signal(signal.SIGINT, request_stop_from_signal)
signal.signal(signal.SIGTERM, request_stop_from_signal)


def web_status(message: str) -> None:
    print(f"WEB_STATUS:{message}", flush=True)


def web_running() -> None:
    print("WEB_RUNNING", flush=True)


OBJECT_KO_MAP = {
    "Battery": "배터리",
    "Cup": "컵",
    "Screwdriver": "드라이버",
    "Tape": "테이프",
}

TARGET_KO_MAP = {
    "gray_bin": "공구상자",
    "white_bin": "바구니",
    "first_drawer": "첫 번째 서랍",
    "second_drawer": "두 번째 서랍",
}


def task_to_korean_cleanup_message(task: str) -> str:
    pattern = r"Pick up the (.*?) and place it into the (.*?)\."
    match = re.match(pattern, task.strip())
    if not match:
        return "로봇이 물건을 정리하고 있어요."
    obj_en = match.group(1)
    target_en = match.group(2)
    obj_ko = OBJECT_KO_MAP.get(obj_en, obj_en)
    target_ko = TARGET_KO_MAP.get(target_en, target_en)
    return f"로봇이 {obj_ko}를 {target_ko}에 정리하고 있어요."


def announce_to_web_and_tts(message: str) -> None:
    web_status(message)
    play_tts(message)


def request_stop_from_stdin() -> None:
    """Called when 'STOP' is received from stdin during mode execution."""
    global STOP_REQUESTED
    STOP_REQUESTED = True
    if CURRENT_EVENTS is not None:
        CURRENT_EVENTS["stop_recording"] = True
        CURRENT_EVENTS["exit_early"] = True
    print("\nWEB_STATUS:모드를 종료합니다.", flush=True)


class StdinMonitor:
    """
    Background thread that monitors stdin for 'STOP' commands during mode execution.
    This allows the web server to interrupt a running mode.
    """

    def __init__(self):
        self._thread = None
        self._running = False
        self._command = None
        self._lock = threading.Lock()

    def start(self):
        self._running = True
        self._command = None
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def get_command(self):
        """Return and clear any pending command."""
        with self._lock:
            cmd = self._command
            self._command = None
            return cmd

    def _monitor_loop(self):
        while self._running:
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                line = sys.stdin.readline().strip()
                if line:
                    with self._lock:
                        self._command = line
                    if line.upper() == "STOP":
                        request_stop_from_stdin()
                        break
                    elif line.upper() == "QUIT":
                        request_stop_from_stdin()
                        break


@dataclass
class DatasetRecordConfig:
    repo_id: str
    single_task: str
    root: str | Path | None = None
    fps: int = 30
    episode_time_s: int | float = 60
    reset_time_s: int | float = 60
    num_episodes: int = 50
    video: bool = True
    push_to_hub: bool = True
    private: bool = False
    tags: list[str] | None = None
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    vcodec: str = "libsvtav1"
    streaming_encoding: bool = False
    encoder_queue_maxsize: int = 30
    encoder_threads: int | None = None
    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError(
                "You need to provide a task as argument in `single_task`."
            )


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    teleop: TeleoperatorConfig | None = None
    policy: PreTrainedConfig | None = None
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    play_sounds: bool = True
    resume: bool = False

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(
                policy_path, cli_overrides=cli_overrides
            )
            self.policy.pretrained_path = policy_path

        # Daemon mode: teleop/policy not required at CLI level
        # Policies are loaded inside record() for each mode.

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]

# 추가아 : 욜로 완료
SETTING_TARGET_POLYGONS = {
    "Battery": np.array([
    [120, 250],
    [190, 250],
    [190, 320],
    [120, 320]
    ], dtype=np.int32),

    "Cup": np.array([
    [95, 110],
    [165, 110],
    [165, 180],
    [95, 180]
    ], dtype=np.int32),
}

def check_setting_done(robot):
    cam = robot.cameras["top"]
    img = cam.read()

    tmp_path = "setting_check_top.png"
    Image.fromarray(img, mode="RGB").save(tmp_path)

    detected_objects = detect_object(tmp_path)
    print("\n===== YOLO DETECTION RESULT =====", flush=True)
    for obj in detected_objects:
        print(obj, flush=True)

    found = {}

    for obj in detected_objects:
        name = obj["name"]

        if name not in SETTING_TARGET_POLYGONS:
            continue

        x, y = obj["location"]

        polygon = SETTING_TARGET_POLYGONS[name]

        inside = cv2.pointPolygonTest(
            polygon,
            (float(x), float(y)),
            False
        ) >= 0

        print(
            f"[SETTING CHECK] {name}: ({int(x)}, {int(y)}) -> {inside}",
            flush=True
        )

        if inside:
            found[name] = True

    print("[SETTING RESULT]", found, flush=True)

    # return all(
    #     found.get(name, False)
    #     for name in SETTING_TARGET_POLYGONS.keys()
    # )
    return (
        found.get("Battery", False)
        and found.get("Cup", False)
    )

@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
    enable_setting_check: bool = False,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(
            f"The dataset fps should be equal to requested fps "
            f"({dataset.fps} != {fps})."
        )
    last_done_check_t = 0.0
    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next(
            (t for t in teleop if isinstance(t, KeyboardTeleop)), None
        )
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so_leader.SO100Leader
                        | so_leader.SO101Leader
                        | koch_leader.KochLeader
                        | omx_leader.OmxLeader
                    ),
                )
            ),
            None,
        )
        if not (
            teleop_arm
            and teleop_keyboard
            and len(teleop) == 2
            and robot.name == "lekiwi_client"
        ):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one "
                "KeyboardTeleop and one arm teleoperator."
            )

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    no_action_count = 0
    timestamp = 0
    start_episode_t = time.perf_counter()
    dbg_prev_time = time.time()
    dbg_frame_count = 0
    dbg_update_interval = 2

    while timestamp < control_time_s and not STOP_REQUESTED and not events["stop_recording"]:
        start_loop_t = time.perf_counter()
        now_t = time.perf_counter()

        if enable_setting_check and now_t - last_done_check_t >= 5.0:
            last_done_check_t = now_t

            if check_setting_done(robot):
                print("세팅 완료 감지 -> record_loop 종료", flush=True)
                break

        if STOP_REQUESTED or events["exit_early"] or events["stop_recording"]:
            events["exit_early"] = False
            break

        obs = robot.get_observation()

        if STOP_REQUESTED or events["stop_recording"]:
            break

        obs_processed = robot_observation_processor(obs)

        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(
                dataset.features, obs_processed, prefix=OBS_STR
            )

        if policy is not None and preprocessor is not None and postprocessor is not None:
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )
            act_processed_policy: RobotAction = make_robot_action(
                action_values, dataset.features
            )
        elif policy is None and isinstance(teleop, Teleoperator):
            act = teleop.get_action()
            act_processed_teleop = teleop_action_processor((act, obs))
        elif policy is None and isinstance(teleop, list):
            arm_action = teleop_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)
            act = (
                {**arm_action, **base_action}
                if len(base_action) > 0
                else arm_action
            )
            act_processed_teleop = teleop_action_processor((act, obs))
        else:
            no_action_count += 1
            if no_action_count == 1 or no_action_count % 10 == 0:
                logging.warning(
                    "No policy or teleoperator provided, skipping action."
                )
            continue

        if policy is not None and act_processed_policy is not None:
            action_values = act_processed_policy
            robot_action_to_send = robot_action_processor(
                (act_processed_policy, obs)
            )
        else:
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor(
                (act_processed_teleop, obs)
            )

        if STOP_REQUESTED or events["stop_recording"]:
            break

        _sent_action = robot.send_action(robot_action_to_send)

        if dataset is not None:
            action_frame = build_dataset_frame(
                dataset.features, action_values, prefix=ACTION
            )
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data(
                observation=obs_processed,
                action=action_values,
                compress_images=display_compressed_images,
            )

        dt_s = time.perf_counter() - start_loop_t
        sleep_time_s: float = 1 / fps - dt_s
        # if sleep_time_s < 0:
        #     print()
        #     logging.warning(
        #         f"Record loop running slower ({1 / dt_s:.1f} Hz) "
        #         f"than target FPS ({fps} Hz)."
        #     )

        precise_sleep(max(sleep_time_s, 0.0))
        timestamp = time.perf_counter() - start_episode_t

        dbg_frame_count += 1
        if dbg_frame_count % dbg_update_interval == 0:
            dbg_current_time = time.time()
            elapsed = dbg_current_time - dbg_prev_time
            current_fps = dbg_update_interval / elapsed
            # if current_fps < 25.0:
            #     print(
            #         f"\r⚠️ [FPS Monitor] Current FPS: {current_fps:.2f} Hz\033[K",
            #         end="",
            #         flush=True,
            #     )
            # else:
            #     print(
            #         f"\r✅ [FPS Monitor] Current FPS: {current_fps:.2f} Hz\033[K",
            #         end="",
            #         flush=True,
            #     )
            dbg_prev_time = dbg_current_time


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    global bin_weights, STOP_REQUESTED, CURRENT_EVENTS
    STOP_REQUESTED = False
    CURRENT_EVENTS = None
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(
            session_name="recording", ip=cfg.display_ip, port=cfg.display_port
        )
    display_compressed_images = (
        True
        if (
            cfg.display_data
            and cfg.display_ip is not None
            and cfg.display_port is not None
        )
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = (
        make_teleoperator_from_config(cfg.teleop)
        if cfg.teleop is not None
        else None
    )

    (
        teleop_action_processor,
        robot_action_processor,
        robot_observation_processor,
    ) = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features
            ),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(
                observation=robot.observation_features
            ),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = None
    listener = None
    stdin_monitor = StdinMonitor()

    def load_policy_set(policy_path):
        """Load a policy + pre/post processors from a given path."""
        policy_cfg = PreTrainedConfig.from_pretrained(
            policy_path, local_files_only=True
        )
        policy_cfg.pretrained_path = policy_path

        policy_obj = make_policy(policy_cfg, ds_meta=dataset.meta)

        preprocessor_obj, postprocessor_obj = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=policy_cfg.pretrained_path,
            dataset_stats=rename_stats(
                dataset.meta.stats, cfg.dataset.rename_map
            ),
            preprocessor_overrides={
                "device_processor": {"device": policy_cfg.device},
                "rename_observations_processor": {
                    "rename_map": cfg.dataset.rename_map
                },
            },
        )

        return policy_obj, preprocessor_obj, postprocessor_obj

    def reset_stop_state():
        """Reset STOP_REQUESTED and events for next mode execution."""
        global STOP_REQUESTED
        STOP_REQUESTED = False
        if CURRENT_EVENTS is not None:
            CURRENT_EVENTS["stop_recording"] = False
            CURRENT_EVENTS["exit_early"] = False
            CURRENT_EVENTS["rerecord_episode"] = False

    try:
        if cfg.resume:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )
            if hasattr(robot, "cameras") and len(robot.cameras) > 0:
                dataset.start_image_writer(
                    num_processes=cfg.dataset.num_image_writer_processes,
                    num_threads=(
                        cfg.dataset.num_image_writer_threads_per_camera
                        * len(robot.cameras)
                    ),
                )
            sanity_check_dataset_robot_compatibility(
                dataset, robot, cfg.dataset.fps, dataset_features
            )
        else:
            sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
            # 🔴 이전 실행에서 남은 캐시 폴더 삭제 (FileExistsError 방지)
            import shutil
            from lerobot.utils.constants import HF_LEROBOT_HOME
            dataset_cache_path = Path(cfg.dataset.root) if cfg.dataset.root else HF_LEROBOT_HOME / cfg.dataset.repo_id
            if dataset_cache_path.exists():
                shutil.rmtree(dataset_cache_path)
                print(f"  [캐시 삭제] {dataset_cache_path}", flush=True)

            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=(
                    cfg.dataset.num_image_writer_threads_per_camera
                    * len(robot.cameras)
                ),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )

        robot.connect()
        if teleop is not None:
            teleop.connect()

        listener, events = init_keyboard_listener()
        CURRENT_EVENTS = events

        # ============================================================
        # 2. Load ALL policies upfront
        # ============================================================
        web_status("정책을 로딩하고 있습니다...")

        # Mode 1: 4 organize policies (YOLO + RL)
        policy_sets_mode1 = {
            0: load_policy_set(
                str(MODEL_POLICIES_DIR / "organize_screwdriver")
            ),
            1: load_policy_set(
                str(MODEL_POLICIES_DIR / "organize_battery")
            ),
            2: load_policy_set(
                str(MODEL_POLICIES_DIR / "organize_screwdriver")
            ),
            3: load_policy_set(
                str(MODEL_POLICIES_DIR / "organize_screwdriver")
            ),
        }

        # Mode 2: packing policy
        policy_mode2, pre_mode2, post_mode2 = load_policy_set(
            str(MODEL_POLICIES_DIR / "packing_mode")
        )

        # Mode 3: setting 1st + 2nd policies
        policy_mode3_1, pre_mode3_1, post_mode3_1 = load_policy_set(
            str(MODEL_POLICIES_DIR / "setting_mode_1st")
        )
        policy_mode3_2, pre_mode3_2, post_mode3_2 = load_policy_set(
            str(MODEL_POLICIES_DIR / "setting_mode_2nd")
        )

        # ============================================================
        # 3. Signal READY
        # ============================================================
        web_status("READY")

        # ============================================================
        # 4. Main daemon loop: wait for commands from stdin
        # ============================================================
        quit_requested = False

        with VideoEncodingManager(dataset):
            while not quit_requested:
                # Non-blocking stdin read with select
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    # Check if we got a signal-based stop
                    if STOP_REQUESTED:
                        break
                    continue

                line = sys.stdin.readline().strip()
                if not line:
                    continue

                cmd = line.upper()

                if cmd == "QUIT":
                    quit_requested = True
                    break
                elif cmd == "STOP":
                    # Nothing running, just acknowledge
                    web_status("STOPPED")
                    continue
                elif cmd.startswith("MODE:"):
                    try:
                        mode_select = int(cmd.split(":")[1])
                    except (ValueError, IndexError):
                        web_status("ERROR:잘못된 모드 명령입니다.")
                        continue

                    if mode_select not in (1, 2, 3):
                        web_status("ERROR:존재하지 않는 모드입니다.")
                        play_tts("존재하지 않는 모드입니다.")
                        continue

                    # Reset stop state before executing mode
                    reset_stop_state()

                    # Start stdin monitor thread for STOP during execution
                    stdin_monitor.start()

                    # Execute the requested mode
                    if mode_select == 1:
                        _execute_mode1(
                            cfg, robot, teleop, events, dataset,
                            teleop_action_processor,
                            robot_action_processor,
                            robot_observation_processor,
                            policy_sets_mode1,
                            display_compressed_images,
                        )
                    elif mode_select == 2:
                        _execute_mode2(
                            cfg, robot, teleop, events, dataset,
                            teleop_action_processor,
                            robot_action_processor,
                            robot_observation_processor,
                            policy_mode2, pre_mode2, post_mode2,
                            display_compressed_images,
                        )
                    elif mode_select == 3:
                        _execute_mode3(
                            cfg, robot, teleop, events, dataset,
                            teleop_action_processor,
                            robot_action_processor,
                            robot_observation_processor,
                            policy_mode3_1, pre_mode3_1, post_mode3_1,
                            policy_mode3_2, pre_mode3_2, post_mode3_2,
                            display_compressed_images,
                        )

                    # Stop the stdin monitor
                    stdin_monitor.stop()

                    # Check if QUIT was received during execution
                    pending_cmd = stdin_monitor.get_command()
                    if pending_cmd and pending_cmd.upper() == "QUIT":
                        quit_requested = True
                        break

                    # Signal mode completion
                    web_status("DONE")

                    # Reset for next command
                    reset_stop_state()
                else:
                    web_status(f"ERROR:알 수 없는 명령: {line}")

    finally:
        CURRENT_EVENTS = None
        stdin_monitor.stop()
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()

        if not is_headless() and listener:
            listener.stop()

        if cfg.dataset.push_to_hub:
            dataset.push_to_hub(
                tags=cfg.dataset.tags, private=cfg.dataset.private
            )

        log_say("Exiting", cfg.play_sounds)
    return dataset


# ================================================================
# Mode execution functions
# ================================================================

def _execute_mode1(
    cfg, robot, teleop, events, dataset,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    policy_sets_mode1,
    display_compressed_images,
):
    """Mode 1: YOLO + RL organize mode."""
    global bin_weights

    print("Selected Mode : Organize Mode (YOLO + RL)", flush=True)

    while not STOP_REQUESTED and not events["stop_recording"]:
        log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)

        cam = robot.cameras["top"]
        img = cam.read()
        rl_img_path = "rl_img_top.png"
        Image.fromarray(img, mode="RGB").save(rl_img_path)
        detected_objects = detect_object(rl_img_path)
        rl_obs = detected_to_obs(detected_objects, bin_weights)
        decision = decide_action(rl_obs)
        if decision["success"]:
            print("\n===== RL Decision =====", flush=True)
            print(decision["script"], flush=True)
            bin_weights = update_bin_weights(bin_weights, decision)
            print("Updated bin weights:", bin_weights, flush=True)
        else:
            print("정리가 완료되었으므로 종료합니다.", flush=True)
            break

        object_id = decision["target_object_id"]
        current_policy, current_preprocessor, current_postprocessor = (
            policy_sets_mode1[object_id]
        )

        web_running()
        time.sleep(0.5)
        announce_to_web_and_tts(
            task_to_korean_cleanup_message(decision["script"])
        )
        record_loop(
            robot=robot,
            events=events,
            fps=cfg.dataset.fps,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=teleop,
            policy=current_policy,
            preprocessor=current_preprocessor,
            postprocessor=current_postprocessor,
            dataset=dataset,
            control_time_s=cfg.dataset.episode_time_s,
            single_task=decision["script"],
            display_data=cfg.display_data,
            display_compressed_images=display_compressed_images,
        )

        if STOP_REQUESTED or events["stop_recording"]:
            dataset.clear_episode_buffer()
            break

        dataset.save_episode()

    if not STOP_REQUESTED:
        announce_to_web_and_tts("정리모드를 다 했어요.")


def _execute_mode2(
    cfg, robot, teleop, events, dataset,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    policy_mode2, pre_mode2, post_mode2,
    display_compressed_images,
):
    """Mode 2: Packing mode."""
    print("Selected Mode : Packing Mode", flush=True)
    recorded_episodes = 0

    while (
        recorded_episodes < cfg.dataset.num_episodes
        and not STOP_REQUESTED
        and not events["stop_recording"]
    ):
        log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
        web_running()
        time.sleep(0.5)
        announce_to_web_and_tts(
            "로봇이 배터리랑 드라이버를 공구상자에 넣고 있어요."
        )
        record_loop(
            robot=robot,
            events=events,
            fps=cfg.dataset.fps,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=teleop,
            policy=policy_mode2,
            preprocessor=pre_mode2,
            postprocessor=post_mode2,
            dataset=dataset,
            control_time_s=cfg.dataset.episode_time_s,
            single_task="Place box and put all objects into box.",
            display_data=cfg.display_data,
            display_compressed_images=display_compressed_images,
        )

        # Reset environment between episodes (skip for last)
        if (
            not STOP_REQUESTED
            and not events["stop_recording"]
            and (
                (recorded_episodes < cfg.dataset.num_episodes - 1)
                or events["rerecord_episode"]
            )
        ):
            log_say("Reset the environment", cfg.play_sounds)
            if robot.name == "unitree_g1":
                robot.reset()
            record_loop(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                control_time_s=cfg.dataset.reset_time_s,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
            )

        if events["rerecord_episode"]:
            log_say("Re-record episode", cfg.play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        if STOP_REQUESTED or events["stop_recording"]:
            dataset.clear_episode_buffer()
            break

        dataset.save_episode()
        announce_to_web_and_tts("패킹모드를 다 했어요.")
        recorded_episodes += 1


def _execute_mode3(
    cfg, robot, teleop, events, dataset,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    policy_mode3_1, pre_mode3_1, post_mode3_1,
    policy_mode3_2, pre_mode3_2, post_mode3_2,
    display_compressed_images,
):
    """Mode 3: Setting mode (1st + 2nd)."""
    print("Selected Mode : Setting Mode", flush=True)
    recorded_episodes = 0

    while (
        recorded_episodes < cfg.dataset.num_episodes
        and not STOP_REQUESTED
        and not events["stop_recording"]
    ):
        log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
        web_running()
        time.sleep(0.5)
        announce_to_web_and_tts("로봇이 드라이버를 세팅하고 있어요.")
        record_loop(
            robot=robot,
            events=events,
            fps=cfg.dataset.fps,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=teleop,
            policy=policy_mode3_1,
            preprocessor=pre_mode3_1,
            postprocessor=post_mode3_1,
            dataset=dataset,
            control_time_s=25,
            single_task="Place the screwdriver.",
            display_data=cfg.display_data,
            display_compressed_images=display_compressed_images,
            enable_setting_check=True,
        )

        if not STOP_REQUESTED and not events["stop_recording"]:
            web_running()
            time.sleep(0.5)
            announce_to_web_and_tts(
                "로봇이 컵이랑 배터리를 세팅하고 있어요."
            )
            record_loop(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                policy=policy_mode3_2,
                preprocessor=pre_mode3_2,
                postprocessor=post_mode3_2,
                dataset=dataset,
                control_time_s=50,
                single_task="Place objects to specific location",
                display_data=cfg.display_data,
                display_compressed_images=display_compressed_images,
                enable_setting_check=True,
            )

        # Reset environment between episodes (skip for last)
        if (
            not STOP_REQUESTED
            and not events["stop_recording"]
            and (
                (recorded_episodes < cfg.dataset.num_episodes - 1)
                or events["rerecord_episode"]
            )
        ):
            log_say("Reset the environment", cfg.play_sounds)
            if robot.name == "unitree_g1":
                robot.reset()
            record_loop(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                control_time_s=cfg.dataset.reset_time_s,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
            )

        if events["rerecord_episode"]:
            log_say("Re-record episode", cfg.play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        if STOP_REQUESTED or events["stop_recording"]:
            dataset.clear_episode_buffer()
            break

        dataset.save_episode()
        announce_to_web_and_tts("세팅모드를 다 했어요.")
        recorded_episodes += 1


# ================================================================
# Entry point
# ================================================================

def main():
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()
