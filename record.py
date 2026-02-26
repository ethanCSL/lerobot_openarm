from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import login
import os


# Change this to your dataset name
REPO_ID = "ethanCSL/" + "0225-test"

hf_token = os.environ.get("HF_TOKEN") 

import shutil
shutil.rmtree(f'/home/csl/.cache/huggingface/lerobot/{REPO_ID}', ignore_errors=True)

if hf_token:
    login(token=hf_token)
    print("Logged in successfully!")
else:
    print("HF_TOKEN environment variable not set. Cannot log in.")

import time

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from teleoperators.openarm_leader import OpenArmConfig, OpenArmLeader
from robots.openarm_follower import OpenArmFollowerConfig, OpenArmFollower
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors


# change this to your parameters
NUM_EPISODES = 10
FPS = 30
EPISODE_TIME_SEC = 999
RESET_TIME_SEC = 5
TASK_DESCRIPTION = "tttttt"

# make sure the camera ports is correct for your setup
camera_config = {
    "right_camera": OpenCVCameraConfig(index_or_path=4, width=640, height=480, fps=FPS),
    "left_camera": OpenCVCameraConfig(index_or_path=16, width=640, height=480, fps=FPS), 
    "body_camera": OpenCVCameraConfig(index_or_path=10, width=640, height=480, fps=FPS),
}

robot_config = OpenArmFollowerConfig(
    right_port = 'can2',
    left_port  = 'can3',
    
    enable_fd = True,
    
    model_path='/home/csl/lerobot_openarm/model/openarm_description_leader.urdf',
    
    cameras=camera_config 
)

teleop_config = OpenArmConfig(
    right_port = 'can0',
    left_port  = 'can1',
    
    enable_fd = True,
    
    model_path='/home/csl/lerobot_openarm/model/openarm_description.urdf',
)

robot = OpenArmFollower(robot_config)
teleop = OpenArmLeader(teleop_config)


robot.connect()
teleop.connect()

time.sleep(1.0)  

action_features = hw_to_dataset_features(robot.action_features, "action") # type: ignore
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

dataset = LeRobotDataset.create(
    repo_id=REPO_ID,
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
    video_backend='torchcodec'
)

_, events = init_keyboard_listener()
init_rerun(session_name="recording")

teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

episode_idx = 0
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    print(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        teleop=teleop,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
            log_say("Reset the environment")
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                control_time_s=RESET_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
            )

    if events["rerecord_episode"]:
        log_say("Re-recording episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    episode_idx += 1
    
log_say("Stop recording")
dataset.finalize()
robot.disconnect()
teleop.disconnect()
# dataset.push_to_hub()
