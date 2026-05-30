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
lerobot-project: 프로젝트 시연용 deploy 스크립트.
인자 없이 `lerobot-project` 만 입력하면 실행
모드 선택 -> policy deploy -> 완료 -> 다시 모드 선택
"""

# 🔴 스크립트 실행 시에만 HF_HUB_OFFLINE=1 
import os
os.environ["HF_HUB_OFFLINE"] = "1"

import logging
import time
import shutil
import glob
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
from lerobot.robots.bi_so_follower.config_bi_so_follower import BiSOFollowerConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig
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
from lerobot.teleoperators.bi_so_leader.config_bi_so_leader import BiSOLeaderConfig
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderConfig
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
import cv2
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from lerobot.scripts.yolo_detect import detect_object, detected_to_obs
from lerobot.scripts.gym_env import DeskCleanEnv
from lerobot.scripts.RL_deploy import decide_action, update_bin_weights
from lerobot.scripts.tts import play_tts

# 🔴 추가: 사용할 전역변수
# current bin state
bin_weights = {
    "first_drawer": 0.0,
    "second_drawer": 0.0,
    "gray_bin": 20.0,
    "white_bin": 0.0,
}


# ============================================================
# 🔴 하드코딩 설정값 (인자 없이 lerobot-project 로 실행)
# ============================================================

# dataset.repo_id
MODE1_REPO_ID = "juyoungggg/eval_organizing"
MODE2_REPO_ID = "juyoungggg/eval_packing"
MODE3_REPO_ID = "juyoungggg/eval_setting"

# dataset.episode_time_s
MODE1_EPISODE_TIME_S = 60    # 시간 지정 필요
MODE2_EPISODE_TIME_S = 5000    # 시간 지정 필요
MODE3_1ST_EPISODE_TIME_S = 50
MODE3_2ND_EPISODE_TIME_S = 5000  # 시간 지정 필요

# policy.path
POLICY_PATH_PACKING = os.path.abspath(os.path.join(BASE_DIR, "../model_policies/packing_mode"))
POLICY_PATH_SETTING_1ST = os.path.abspath(os.path.join(BASE_DIR, "../model_policies/setting_mode_1st"))
POLICY_PATH_SETTING_2ND = os.path.abspath(os.path.join(BASE_DIR, "../model_policies/setting_mode_2nd"))
# Mode 1 policy.path (물체별로 1개씩, TBD)
POLICY_PATHS_ORGANIZE = {
    0: os.path.abspath(os.path.join(BASE_DIR, "../model_policies/organize_screwdriver")),
    1: os.path.abspath(os.path.join(BASE_DIR, "../model_policies/organize_battery")),
    2: os.path.abspath(os.path.join(BASE_DIR, "../model_policies/organize_tape")),
    3: os.path.abspath(os.path.join(BASE_DIR, "../model_policies/organize_cup")),
}

# 캐시 정리 경로
EVAL_CACHE_PATTERN = os.path.expanduser("~/.cache/huggingface/lerobot/juyoungggg/eval_*")


# ============================================================
# 캐시 정리 함수
# ============================================================

def clean_eval_cache():
    # 실행 시 ~/.cache/huggingface/lerobot/juyoungggg/eval_* 삭제
    targets = glob.glob(EVAL_CACHE_PATTERN)
    for path in targets:
        try:
            shutil.rmtree(path)
            print(f"  [캐시 삭제] {path}")
        except Exception as e:
            print(f"  [캐시 삭제 실패] {path} ({e})")
    if not targets:
        print("  [캐시] 정리할 항목 없음.")


# ============================================================
# 하드코딩 Config 생성 함수
# ============================================================

def build_robot_config():
    # 로봇, 카메라 설정 하드코딩
    return BiSOFollowerConfig(
        left_arm_config=SOFollowerConfig(
            port="/dev/ttyACM_FOLLOWER",
            cameras={
                "wrist": OpenCVCameraConfig(
                    index_or_path="/dev/LEFT_WRIST",
                    width=640,
                    height=480,
                    fps=30,
                ),
            },
        ),
        right_arm_config=SOFollowerConfig(
            port="/dev/ttyACM_FOLLOWER_2",
            cameras={
                "wrist": OpenCVCameraConfig(
                    index_or_path="/dev/RIGHT_WRIST",
                    width=640,
                    height=480,
                    fps=30,
                ),
                "top": OpenCVCameraConfig(
                    index_or_path="/dev/RIGHT_TOP",
                    width=640,
                    height=480,
                    fps=30,
                ),
            },
        ),
        id="bimanual_follower",
    )


def build_teleop_config():
    # Teleop 설정 하드코딩
    return BiSOLeaderConfig(
        left_arm_config=SOLeaderConfig(port="/dev/ttyACM_LEADER"),
        right_arm_config=SOLeaderConfig(port="/dev/ttyACM_LEADER_2"),
        id="bimanual_leader",
    )


@dataclass
class DatasetRecordConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # A short but accurate description of the task performed during the recording (e.g. "Pick the Lego block and drop it in the box on the right.")
    single_task: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path'). If None, defaults to $HF_LEROBOT_HOME/repo_id.
    root: str | Path | None = None
    # Limit the frames per second.
    fps: int = 30
    # Number of seconds for data recording for each episode.
    episode_time_s: int | float = 60
    # Number of seconds for resetting the environment after each episode.
    reset_time_s: int | float = 60
    # Number of episodes to record.
    num_episodes: int = 50
    # Encode frames in the dataset into video
    video: bool = True
    # Upload dataset to Hugging Face hub.
    push_to_hub: bool = True
    # Upload on private repository on the Hugging Face hub.
    private: bool = False
    # Add tags to your dataset on the hub.
    tags: list[str] | None = None
    # Number of subprocesses handling the saving of frames as PNG. Set to 0 to use threads only;
    # set to ≥1 to use subprocesses, each using threads to write images. The best number of processes
    # and threads depends on your system. We recommend 4 threads per camera with 0 processes.
    # If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses.
    num_image_writer_processes: int = 0
    # Number of threads writing the frames as png images on disk, per camera.
    # Too many threads might cause unstable teleoperation fps due to main thread being blocked.
    # Not enough threads might cause low camera fps.
    num_image_writer_threads_per_camera: int = 4
    # Number of episodes to record before batch encoding videos
    # Set to 1 for immediate encoding (default behavior), or higher for batched encoding
    video_encoding_batch_size: int = 1
    # Video codec for encoding videos. Options: 'h264', 'hevc', 'libsvtav1', 'auto',
    # or hardware-specific: 'h264_videotoolbox', 'h264_nvenc', 'h264_vaapi', 'h264_qsv'.
    # Use 'auto' to auto-detect the best available hardware encoder.
    vcodec: str = "libsvtav1"
    # Enable streaming video encoding: encode frames in real-time during capture instead
    # of writing PNG images first. Makes save_episode() near-instant. More info in the documentation: https://huggingface.co/docs/lerobot/streaming_video_encoding
    streaming_encoding: bool = False
    # Maximum number of frames to buffer per camera when using streaming encoding.
    # ~1s buffer at 30fps. Provides backpressure if the encoder can't keep up.
    encoder_queue_maxsize: int = 30
    # Number of threads per encoder instance. None = auto (codec default).
    # Lower values reduce CPU usage, maps to 'lp' (via svtav1-params) for libsvtav1 and 'threads' for h264/hevc..
    encoder_threads: int | None = None
    # Rename map for the observation to override the image and state keys
    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    # Whether to control the robot with a teleoperator
    teleop: TeleoperatorConfig | None = None
    # Whether to control the robot with a policy
    policy: PreTrainedConfig | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Display data on a remote Rerun server
    display_ip: str | None = None
    # Port of the remote Rerun server
    display_port: int | None = None
    # Whether to  display compressed images in Rerun
    display_compressed_images: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs after teleop
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs before robot
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],  # runs after robot
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
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

        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    no_action_count = 0
    timestamp = 0
    start_episode_t = time.perf_counter()
    # --- [debug] realtime fps init ---
    dbg_prev_time = time.time()
    dbg_frame_count = 0
    dbg_update_interval = 2 # 10 프레임마다 한 번씩 갱신
    # ----------------------------------------------------

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Get robot observation
        obs = robot.get_observation()

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        obs_processed = robot_observation_processor(obs)

        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        # Get action from either policy or teleop
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

            act_processed_policy: RobotAction = make_robot_action(action_values, dataset.features)

        elif policy is None and isinstance(teleop, Teleoperator):
            act = teleop.get_action()

            # Applies a pipeline to the raw teleop action, default is IdentityProcessor
            act_processed_teleop = teleop_action_processor((act, obs))

        elif policy is None and isinstance(teleop, list):
            arm_action = teleop_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)
            act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
            act_processed_teleop = teleop_action_processor((act, obs))
        else:
            no_action_count += 1
            if no_action_count == 1 or no_action_count % 10 == 0:
                logging.warning(
                    "No policy or teleoperator provided, skipping action generation. "
                    "This is likely to happen when resetting the environment without a teleop device. "
                    "The robot won't be at its rest position at the start of the next episode."
                )
            continue

        # Applies a pipeline to the action, default is IdentityProcessor
        if policy is not None and act_processed_policy is not None:
            action_values = act_processed_policy
            robot_action_to_send = robot_action_processor((act_processed_policy, obs))
        else:
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        # Send action to robot
        _sent_action = robot.send_action(robot_action_to_send)

        if display_data:
            log_rerun_data(
                observation=obs_processed, action=action_values, compress_images=display_compressed_images
            )

        dt_s = time.perf_counter() - start_loop_t

        sleep_time_s: float = 1 / fps - dt_s
        if sleep_time_s < 0:
            print()
            logging.warning(
                f"Record loop is running slower ({1 / dt_s:.1f} Hz) than the target FPS ({fps} Hz). "
                "Robot control might be unstable."
            )

        precise_sleep(max(sleep_time_s, 0.0))

        timestamp = time.perf_counter() - start_episode_t

        # --- [debug] realtime fps ---
        dbg_frame_count += 1
        if dbg_frame_count % dbg_update_interval == 0:
            dbg_current_time = time.time()
            elapsed = dbg_current_time - dbg_prev_time
            current_fps = dbg_update_interval / elapsed

            # if current_fps < 25.0:
            #     print(f"\r⚠️ [FPS Monitor] Current FPS: {current_fps:.2f} Hz\033[K", end="", flush=True)
            # else:
            #     print(f"\r✅ [FPS Monitor] Current FPS: {current_fps:.2f} Hz\033[K", end="", flush=True)

            dbg_prev_time = dbg_current_time
        # -------------------------------------------------------



def record():
    global bin_weights  # 🔴 추가: global선언
    init_logging()

    # 캐시 정리
    print("\n[캐시 정리]")
    clean_eval_cache()

    # 하드코딩된 Config 생성
    robot_cfg = build_robot_config()

    # Teleop 설정
    # 시연 중 리셋이 필요하면 teleop 사용
    teleop_cfg = build_teleop_config() # teleop 사용 시 이 줄만 활성화, 아랫줄 주석처리
    # teleop_cfg = None  # teleop 사용 안 할 시 이 줄만 활성화하고, 윗줄을 주석처리

    # 공통 설정
    fps = 30
    display_data = True
    display_compressed_images = False
    play_sounds = False

    if display_data:
        init_rerun(session_name="lerobot-project")

    robot = make_robot_from_config(robot_cfg)
    teleop = make_teleoperator_from_config(teleop_cfg) if teleop_cfg is not None else None

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    dataset = None
    listener = None

    def load_policy_set(policy_path):    # 🔴 추가: Policy를 따로따로 만들어주는 함수
        policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
        policy_cfg.pretrained_path = policy_path
        policy_cfg.device = "cuda"

        policy_obj = make_policy(policy_cfg, ds_meta=dataset.meta)

        preprocessor_obj, postprocessor_obj = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=policy_cfg.pretrained_path,
            dataset_stats=dataset.meta.stats,
            preprocessor_overrides={
                "device_processor": {"device": policy_cfg.device},
            },
        )

        return policy_obj, preprocessor_obj, postprocessor_obj

    try:
        # Dataset 생성 (feature 정의용 — 저장은 하지 않음)
        dataset = LeRobotDataset.create(
            MODE3_REPO_ID,  # 초기값, 모드 선택 시 재생성
            fps,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=4 * len(robot.cameras),
        )

        robot.connect()
        if teleop is not None:
            teleop.connect()

        listener, events = init_keyboard_listener()

        # ============================================================
        # 🔴 모든 Policy 미리 로드 (모드 선택 시 즉시 실행 가능)
        # ============================================================
        print("\n[Begin policy loading sequence...]")

        # Mode 1: Organize (경로 준비되면 주석 해제)
        print("  Loading Mode 1 policies...")
        policy_sets_mode1 = {
            0: load_policy_set(POLICY_PATHS_ORGANIZE[0]),
            1: load_policy_set(POLICY_PATHS_ORGANIZE[1]),
            2: load_policy_set(POLICY_PATHS_ORGANIZE[2]),
            3: load_policy_set(POLICY_PATHS_ORGANIZE[3]),
        }

        # Mode 2: Packing
        print("  Loading Mode 2 policy (Packing)...")
        policy_mode2, pre_mode2, post_mode2 = load_policy_set(POLICY_PATH_PACKING)

        # Mode 3: Setting (1st + 2nd)
        print("  Loading Mode 3 policies (Setting 1st, 2nd)...")
        policy_mode3_1, pre_mode3_1, post_mode3_1 = load_policy_set(POLICY_PATH_SETTING_1ST)
        policy_mode3_2, pre_mode3_2, post_mode3_2 = load_policy_set(POLICY_PATH_SETTING_2ND)

        print("[Policy pre-loading complete!]\n")

        # ============================================================
        # 🔴 모드 선택 루프 (모드 끝나면 다시 돌아옴)
        # ============================================================
        while True:
            print("\n" + "=" * 50)
            print("  모드를 선택하세요:")
            print("    1: Organize Mode (정리 모드)")
            print("    2: Packing Mode (패킹 모드)")
            print("    3: Setting Mode (세팅 모드)")
            print("    0: 종료")
            print("=" * 50)

            try:
                mode_select = int(input("\n  Select Mode (0/1/2/3): "))
            except (ValueError, EOFError):
                print("  잘못된 입력입니다.")
                continue

            if mode_select == 0:
                play_tts("프로그램을 종료합니다")
                break

            # 모드 전환 시 캐시 정리
            clean_eval_cache()

            # 모드별 dataset 재생성
            if mode_select == 1:
                repo_id = MODE1_REPO_ID
            elif mode_select == 2:
                repo_id = MODE2_REPO_ID
            elif mode_select == 3:
                repo_id = MODE3_REPO_ID
            else:
                play_tts("존재하지 않는 모드입니다")
                continue

            dataset = LeRobotDataset.create(
                repo_id,
                fps,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=4 * len(robot.cameras),
            )

            # 🔴 MODE 1: Organize
            if mode_select == 1:
                print("\n  [Mode 1] Organize Mode")
                play_tts("정리 모드를 선택했습니다")

                while not events["stop_recording"]:
                    for _ in range(2):
                        time.sleep(0.5)
                        obs = robot.get_observation()
                        
                    cam = robot.cameras["top"]
                    img = cam.read()
                    debug_path = "debug_top.png"
                    from PIL import Image
                    Image.fromarray(img, mode="RGB").save(debug_path)
                    detected_objects = detect_object(debug_path)
                    rl_obs = detected_to_obs(detected_objects, bin_weights)
                    decision = decide_action(rl_obs)
                    if decision["success"]:
                        print("\n===== RL Decision =====")
                        print(decision["script"])
                        bin_weights = update_bin_weights(bin_weights, decision)
                        print("Updated bin weights:", bin_weights)
                    else:
                        print("정리가 완료되었으므로 종료합니다.")
                        play_tts("정리가 완료되었습니다")
                        break

                    object_id = decision["target_object_id"]
                    current_policy, current_pre, current_post = policy_sets_mode1[object_id]

                    record_loop(
                        robot=robot,
                        events=events,
                        fps=fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        policy=current_policy,
                        preprocessor=current_pre,
                        postprocessor=current_post,
                        dataset=dataset,
                        control_time_s=MODE1_EPISODE_TIME_S,
                        single_task=decision["script"],
                        display_data=display_data,
                        display_compressed_images=display_compressed_images,
                    )

                    events["stop_recording"] = False
                    print("\n  [Mode 1] 완료.")

            # 🔴 MODE 2: Packing
            elif mode_select == 2:
                print("\n  [Mode 2] Packing Mode")
                play_tts("패킹 모드를 선택했습니다")

                record_loop(
                    robot=robot,
                    events=events,
                    fps=fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    policy=policy_mode2,
                    preprocessor=pre_mode2,
                    postprocessor=post_mode2,
                    dataset=dataset,
                    control_time_s=MODE2_EPISODE_TIME_S,
                    single_task="Place box and put all objects into box.",
                    display_data=display_data,
                    display_compressed_images=display_compressed_images,
                )

                print("\n  [Mode 2] 완료.")
                play_tts("패킹 모드가 완료되었습니다")

            # 🔴 MODE 3: Setting
            elif mode_select == 3:
                print("\n  [Mode 3] Setting Mode")
                play_tts("세팅 모드를 선택했습니다")

                # setting_1st
                # print("  [Stage 1] Place objects to specific location")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    policy=policy_mode3_1,
                    preprocessor=pre_mode3_1,
                    postprocessor=post_mode3_1,
                    dataset=dataset,
                    control_time_s=MODE3_1ST_EPISODE_TIME_S,
                    single_task="Place objects to specific location",
                    display_data=display_data,
                    display_compressed_images=display_compressed_images,
                )

                # setting_2nd
                # print("\n  [Stage 2] Open drawer, grip objects and place them.")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    policy=policy_mode3_2,
                    preprocessor=pre_mode3_2,
                    postprocessor=post_mode3_2,
                    dataset=dataset,
                    control_time_s=MODE3_2ND_EPISODE_TIME_S,
                    single_task="Open drawer, grip objects and place them.",
                    display_data=display_data,
                    display_compressed_images=display_compressed_images,
                )

                print("\n  [Mode 3] 완료.")
                play_tts("세팅 모드가 완료되었습니다")

    finally:
        log_say("Stop recording", play_sounds, blocking=True)

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()

        if not is_headless() and listener:
            listener.stop()

        log_say("Exiting", play_sounds)


def main():
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()
