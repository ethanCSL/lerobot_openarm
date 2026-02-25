import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from robots.openarm_follower import OpenArmFollowerConfig, OpenArmFollower
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata



MAX_EPISODES = 30
MAX_STEPS_PER_EPISODE = 999999999


def main():
    device = torch.device("cuda")  # or "cuda" or "cpu"
    pretrained_model_path = "/home/csl/lerobot_openarm/outputs/train/svla_multiblock/checkpoints/last/pretrained_model"
    model = SmolVLAPolicy.from_pretrained(pretrained_model_path)

    dataset_id = "ethanCSL/20260120-must-success"
    # This only downloads the metadata for the dataset, ~10s of MB even for large-scale datasets
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    preprocess, postprocess = make_pre_post_processors(
        model.config, 
        pretrained_model_path,
        dataset_stats=dataset_metadata.stats,
        # preprocessor_overrides={"device_processor": {"device": str(device)}}
    )                                                 



    # # find ports using lerobot-find-port
    # # something like "/dev/tty.usbmodem58760431631"

    # # the robot ids are used the load the right calibration files
    # something like "follower_so100"

    # Robot and environment configuration
    # Camera keys must match the name and resolutions of the ones used for training!
    # You can check the camera keys expected by a model in the info.json card on the model card on the Hub
    
    camera_config = {
    "camera1": OpenCVCameraConfig(index_or_path=4, width=640, height=480, fps=30),#right_camera
    "camera2": OpenCVCameraConfig(index_or_path=16, width=640, height=480, fps=30), #left_camera
    "camera3": OpenCVCameraConfig(index_or_path=10, width=640, height=480, fps=30),# body_camera
}
    
    robot_cfg = OpenArmFollowerConfig(
    right_port = 'can2',
    left_port  = 'can3',
    
    enable_fd = True,
    
    model_path='/home/csl/lerobot_openarm/model/openarm_description.urdf',
    
    cameras=camera_config # type: ignore
)
    robot = OpenArmFollower(robot_cfg)
    robot.connect()

    task = "fold the green towel in half"  # something like "pick the red block"
    # task = "say hi to human"  # something like "pick the red block"
    robot_type = "openarm_follower"  # something like "so100_follower" for multi-embodiment datasets

    # This is used to match the raw observation keys to the keys expected by the policy
    print(f"action_features_before: {robot.action_features}")
    action_features = hw_to_dataset_features(robot.action_features, "action")
    print(f"action_features_after: {action_features}")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}
    print(f"action_shape: {action_features['action']['shape']}")

    for _ in range(MAX_EPISODES):
        for _ in range(MAX_STEPS_PER_EPISODE):
            obs = robot.get_observation()
            obs_frame = build_inference_frame(
                observation=obs, ds_features=dataset_features, device=device, task=task, robot_type=robot_type
            )
            # print(obs_frame.keys())
            obs = preprocess(obs_frame)

            action = model.select_action(obs)
            action = postprocess(action)
            action = make_robot_action(action, dataset_features)
            robot.send_action(action)

        print("Episode finished! Starting new episode...")


if __name__ == "__main__":
    main()