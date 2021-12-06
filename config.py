from agents.sac_agent import SACAgent


class SubmissionConfig:
    agent = SACAgent
    pre_eval_time = 100
    eval_episodes = 10


class SACConfig:
    experiment_name = "SAC"
    make_random_actions = 0
    inference_only = False
    load_checkpoint = False
    record_experience = False
    encoder_switch = 1
    use_encoder_type = "vae_small"  # ['vae', 'vae_small', 'resent']

    vae_small = {
        "vae_chkpt_statedict": "./common/models/vae_144w_42h_32latent.pth",
        "latent_dims": 32,
        "hiddens": [32, 64, 64, 32, 32],
        "speed_hiddens": [8, 8],
        "actor_hiddens": [64, 64, 32],
        "im_c": 3,
        "im_w": 144,
        "im_h": 42,
        "ac_input_dims": 32,
    }

    seed = 0
    gamma = 0.99
    polyak = 0.995
    lr = 0.003
    alpha = 0.2
    num_test_episodes = 1
    safety_margin = 4.2
    save_episodes = 1
    save_freq = 1
    total_steps = 250_000
    replay_size = 250_000
    batch_size = 256
    start_steps = 2000
    update_after = 2000
    update_every = 1
    eval_every = 5000
    max_ep_len = 50000
    im_w = 144
    im_h = 144
    checkpoint = "${PREFIX}/l2r/checkpoints/agent/thruxton/sac/sac_episode_1000.pt"
    save_path = "${PREFIX}/l2r/results/${DIRHASH}workspaces/${USER}/results"
    track_name = "Thruxton"
    safety_data = "${PREFIX}/l2r/datasets/l2r/datasets/safety_sets"
    record_dir = "${PREFIX}/l2r/datasets/l2r/datasets/safety_records_dataset/"


class EnvConfig:
    multimodal = True
    max_timesteps = 2000
    obs_delay = 0.1
    not_moving_timeout = 50
    reward_pol = "default"
    reward_kwargs = {
        "oob_penalty": 5.0,
        "min_oob_penalty": 25.0,
    }
    controller_kwargs = {
        "sim_version": "ArrivalSim-linux-0.7.0-cmu4",
        "quiet": False,
        "user": "ubuntu",
        "start_container": False,
        "sim_path": None,
    }
    action_if_kwargs = {
        "max_accel": 4,
        "min_accel": -1,
        "max_steer": 0.3,
        "min_steer": -0.3,
        "ip": "0.0.0.0",
        "port": 7077,
    }
    pose_if_kwargs = {
        "ip": "0.0.0.0",
        "port": 7078,
    }
    camera_if_kwargs = {
        "ip": "tcp://127.0.0.1",
        "port": 8008,
    }


class SimulatorConfig:
    racetrack = ["Thruxton"]
    active_sensors = ["CameraFrontRGB", "ImuOxtsSensor"]
    driver_params = {
        "DriverAPIClass": "VApiUdp",
        "DriverAPI_UDP_SendAddress": "0.0.0.0",
    }
    camera_params = {
        "Format": "ColorBGR8",
        "FOVAngle": 90,
        "Width": 192,
        "Height": 144,
        "bAutoAdvertise": True,
    }
