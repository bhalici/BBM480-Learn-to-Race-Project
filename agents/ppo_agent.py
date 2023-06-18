"""This is OpenAI' Spinning Up PyTorch implementation of Soft-Actor-Critic with
minor adjustments.
For the official documentation, see below:
https://spinningup.openai.com/en/latest/algorithms/sac.html#documentation-pytorch-version
Source:
https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py
"""
import itertools
import queue, threading
from copy import deepcopy

import torch
import numpy as np
import cv2
from gym.spaces import Box
from torch.optim import Adam

from agents.base import BaseAgent
from agents.ppo_core import ActorCritic
from l2r.common.models.vae import VAE
from l2r.common.utils import RecordExperience
from l2r.common.utils import setup_logging

from ruamel.yaml import YAML

from agents.replay_buffer import ReplayBuffer
from agents.ppo_buffer import PPOBuffer

DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# seed = np.random.randint(255)
# torch.manual_seed(seed)
# np.random.seed(seed)


class PPOAgent(BaseAgent):

    def __init__(self):
        super(PPOAgent, self).__init__()

        self.cfg = self.load_model_config("models/ppo/params-ppo.yaml")
        self.file_logger, self.tb_logger = self.setup_loggers()

        if self.cfg["record_experience"]:
            self.setup_experience_recorder()

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        # self.act_limit = self.action_space.high[0]

        self.setup_vision_encoder()
        self.set_params()

    def select_action(self, obs, encode=True):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if encode:
            obs = self._encode(obs)

        a, v, logp = self.actor_critic.step(obs.to(DEVICE))
        a = np.clip(a, -1., 1.)

        self.t = self.t + 1
        return a, v, logp

    def register_reset(self, obs) -> np.array:
        """
        Same input/output as select_action, except this method is called at episodal reset.
        """
        # camera, features, state = obs
        self.t = 1e6

    def load_model(self, path):
        self.actor_critic.load_state_dict(torch.load(path))

    def save_model(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def setup_experience_recorder(self):
        self.save_queue = queue.Queue()
        self.save_batch_size = 256
        self.record_experience = RecordExperience(
            self.cfg["record_dir"],
            self.cfg["track_name"],
            self.cfg["experiment_name"],
            self.file_logger,
            self,
        )
        self.save_thread = threading.Thread(target=self.record_experience.save_thread)
        self.save_thread.start()

    def setup_vision_encoder(self):
        assert self.cfg["use_encoder_type"] in [
            "vae"
        ], "Specified encoder type must be in ['vae']"
        speed_hiddens = self.cfg[self.cfg["use_encoder_type"]]["speed_hiddens"]
        self.feat_dim = self.cfg[self.cfg["use_encoder_type"]]["latent_dims"] + 1
        self.obs_dim = (
            self.cfg[self.cfg["use_encoder_type"]]["latent_dims"] + speed_hiddens[-1]
            if self.cfg["encoder_switch"]
            else None
        )

        if self.cfg["use_encoder_type"] == "vae":
            self.backbone = VAE(
                im_c=self.cfg["vae"]["im_c"],
                im_h=self.cfg["vae"]["im_h"],
                im_w=self.cfg["vae"]["im_w"],
                z_dim=self.cfg["vae"]["latent_dims"],
            )
            self.backbone.load_state_dict(
                torch.load(self.cfg["vae"]["vae_chkpt_statedict"], map_location=DEVICE)
            )
        else:
            raise NotImplementedError

        self.backbone.to(DEVICE)

    def set_params(self):
        self.save_episodes = True
        self.episode_num = 0
        self.best_ret = 0
        self.t = 0
        self.atol = 1e-3
        self.store_from_safe = False
        self.pi_scheduler = None
        self.best_pct = 0

        self.local_steps_per_epoch = int(self.cfg["steps_per_epoch"])

        # This is important: it allows child classes (that extend this one) to "push up" information
        # that this parent class should log
        self.metadata = {}

        self.action_space = Box(-1, 1, (2,))
        self.act_dim = self.action_space.shape[0]

        # Experience buffer
        self.ppo_buffer = PPOBuffer(
            obs_dim=self.feat_dim, act_dim=self.act_dim, size=self.local_steps_per_epoch
        )

        self.actor_critic = ActorCritic(
            self.cfg,
            device=DEVICE,
        )

        if self.cfg["checkpoint"] and self.cfg["load_checkpoint"]:
            self.load_model(self.cfg["checkpoint"])


    @staticmethod
    def load_model_config(path):
        yaml = YAML()
        params = yaml.load(open(path))
        sac_kwargs = params["agent_kwargs"]
        return sac_kwargs

    def setup_loggers(self):
        save_path = self.cfg["model_save_path"]
        loggers = setup_logging(save_path, self.cfg["experiment_name"], True)
        loggers[0]("Using random seed: {}".format(0))
        return loggers


    def compute_loss_pi(self, data):
        clip_ratio = 0.2

        obs, act, adv, logp_old = data['obs'].to(DEVICE), data['act'].to(DEVICE), data['adv'].to(DEVICE), data['logp'].to(DEVICE)

        # Policy loss
        pi, logp = self.actor_critic.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info


    def compute_loss_v(self, data):
        obs, ret = data['obs'].to(DEVICE), data['ret'].to(DEVICE)
        return ((self.actor_critic.v(obs) - ret)**2).mean()
        
    def update(self):
            train_pi_iters=80
            target_kl=0.01
            train_v_iters=80

            data = self.ppo_buffer.get()

            # Train policy with multiple steps of gradient descent
            for i in range(train_pi_iters):
                self.pi_optimizer.zero_grad()
                loss_pi, pi_info = self.compute_loss_pi(data)
                kl = pi_info['kl']
                if kl > 1.5 * target_kl:
                    self.file_logger('Early stopping at step %d due to reaching max kl.'%i)
                    break
                loss_pi.backward()
                self.pi_optimizer.step()


            # Value function learning
            for i in range(train_v_iters):
                self.vf_optimizer.zero_grad()
                loss_v = self.compute_loss_v(data)
                loss_v.backward()
                self.vf_optimizer.step()
    
    def _step(self, env, action):
        obs, reward, done, info = env.step(action)
        return obs[1], self._encode(obs), obs[0], reward, done, info

    def _reset(self, env, random_pos=False):
        camera = 0
        while (np.mean(camera) == 0) | (np.mean(camera) == 255):
            obs = env.reset(random_pos=random_pos)
            (state, camera), _ = obs
        return camera, self._encode((state, camera)), state

    def _encode(self, o):
        state, img = o

        if self.cfg["use_encoder_type"] == "vae":
            img_embed = self.backbone.encode_raw(np.array(img), DEVICE)[0][0]
            speed = (
                torch.tensor((state[4] ** 2 + state[3] ** 2 + state[5] ** 2) ** 0.5)
                .float()
                .reshape(1, -1)
                .to(DEVICE)
            )
            out = torch.cat([img_embed.unsqueeze(0), speed], dim=-1).squeeze(
                0
            )  # torch.Size([33])
            self.using_speed = 1
        else:
            raise NotImplementedError

        assert not torch.sum(torch.isnan(out)), "found a nan value"
        out[torch.isnan(out)] = 0

        return out

    def checkpoint_model(self, ep_ret, n_eps):
        # Save if best (or periodically)
        if ep_ret > self.best_ret:  # and ep_ret > 100):
            path_name = f"{self.cfg['model_save_path']}/best_{self.cfg['experiment_name']}_episode_{n_eps}.statedict"
            self.file_logger(
                f"New best episode reward of {round(ep_ret, 1)}! Saving: {path_name}"
            )
            self.best_ret = ep_ret
            torch.save(self.actor_critic.state_dict(), path_name)

        elif self.save_episodes and ((n_eps + 1) % self.cfg["save_freq"] == 0):
            path_name = f"{self.cfg['model_save_path']}/{self.cfg['experiment_name']}_episode_{n_eps}.statedict"
            self.file_logger(
                f"Periodic save (save_freq of {self.cfg['save_freq']}) to {path_name}"
            )
            torch.save(self.actor_critic.state_dict(), path_name)

    def training(self, env):
        self.pi_optimizer = Adam(self.actor_critic.pi.parameters(), lr=self.cfg["lr"])
        self.vf_optimizer = Adam(self.actor_critic.v.parameters(), lr=self.cfg["lr"])

        # Prepare for interaction with environment
        # start_time = time.time()
        best_ret, ep_ret, ep_len = 0, 0, 0

        self._reset(env, random_pos=True)
        camera, feat, state, r, d, info = self._step(env, [0, 1])

        experience = []
        speed_dim = 1 if self.using_speed else 0
        assert (
            len(feat)
            == self.cfg[self.cfg["use_encoder_type"]]["latent_dims"] + speed_dim
        ), "'o' has unexpected dimension or is a tuple"

        t_start = 0
        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(self.cfg["epochs"]):
            for t in range(self.local_steps_per_epoch):
                a, v, logp = self.select_action(feat, encode=False)

                # Step the env
                camera2, feat2, state2, r, d, info = self._step(env, a)
                

                # Check that the camera is turned on
                assert (np.mean(camera2) > 0) & (np.mean(camera2) < 255)

                # Prevents the agent from getting stuck by sampling random actions
                # self.atol for SafeRandom and SPAR are set to -1 so that this condition does not activate
                if np.allclose(state2[15:16], state[15:16], atol=self.atol, rtol=0):
                    # self.file_logger("Sampling random action to get unstuck")
                    a = env.action_space.sample()

                    # Step the env
                    camera2, feat2, state2, r, d, info = self._step(env, a)
                    ep_len += 1

                state = state2
                ep_ret += r
                ep_len += 1
                
                self.ppo_buffer.store(feat, a, r, v, logp)


                if self.cfg["record_experience"]:
                    recording = self.add_experience(
                        action=a,
                        camera=camera,
                        next_camera=camera2,
                        done=d,
                        env=env,
                        feature=feat,
                        next_feature=feat2,
                        info=info,
                        reward=r,
                        state=state,
                        next_state=state2,
                        step=t,
                    )
                    experience.append(recording)

                    # quickly pass data to save thread
                    # if len(experience) == self.save_batch_size:
                    #    self.save_queue.put(experience)
                    #    experience = []

                # Super critical, easy to overlook step: make sure to update
                # most recent observation!
                feat = feat2
                state = state2  # in case we, later, wish to store the state in the replay as well
                camera = camera2  # in case we, later, wish to store the state in the replay as well

                timeout = ep_len == self.cfg["max_ep_len"]
                terminal = d or timeout
                epoch_ended = t == self.local_steps_per_epoch - 1
                if terminal or epoch_ended:
                    if epoch_ended and not(terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v, _ = self.actor_critic.step(feat.to(DEVICE))

                    else:
                        v = 0
                        
                    self.ppo_buffer.finish_path(v)

                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        self.metadata["info"] = info
                        self.episode_num += 1
                        msg = f"[Ep {self.episode_num }] {self.metadata}"
                        self.file_logger(msg)
                        self.log_train_metrics_to_tensorboard(ep_ret, t, t_start)

                        # Quickly dump recently-completed episode's experience to the multithread queue,
                        # as long as the episode resulted in "success"
                        if self.cfg[
                            "record_experience"
                        ]:  # and self.metadata['info']['success']:
                            self.file_logger("Writing experience")
                            self.save_queue.put(experience)
                    	
                    	

                    (
                        camera,
                        ep_len,
                        ep_ret,
                        experience,
                        feat,
                        state,
                        t_start,
                    ) = self.reset_episode(env, t)

            # Save model
            if (epoch % self.cfg["save_freq"] == 0) or (epoch == self.cfg["epochs"]-1):
                self.checkpoint_model(ep_ret, epoch)
                
            self.update()

    def reset_episode(self, env, t):
        camera, feat, state = self._reset(env, random_pos=True)
        ep_ret, ep_len, self.metadata, experience = 0, 0, {}, []
        t_start = t + 1
        camera, feat, state2, r, d, info = self._step(env, [0, 1])
        return camera, ep_len, ep_ret, experience, feat, state, t_start

    def add_experience(
        self,
        action,
        camera,
        next_camera,
        done,
        env,
        feature,
        next_feature,
        info,
        reward,
        state,
        next_state,
        step,
    ):
        self.recording = {
            "step": step,
            "nearest_idx": env.nearest_idx,
            "camera": camera,
            "feature": feature.detach().cpu().numpy(),
            "state": state,
            "action_taken": action,
            "next_camera": next_camera,
            "next_feature": next_feature.detach().cpu().numpy(),
            "next_state": next_state,
            "reward": reward,
            "episode": self.episode_num,
            "stage": "training",
            "done": done,
            "metadata": info,
        }
        return self.recording

    def log_val_metrics_to_tensorboard(self, info, ep_ret, n_eps, n_val_steps):
        self.tb_logger.add_scalar("val/episodic_return", ep_ret, n_eps)
        self.tb_logger.add_scalar("val/ep_n_steps", n_val_steps, n_eps)

        try:
            self.tb_logger.add_scalar(
                "val/ep_pct_complete", info["metrics"]["pct_complete"], n_eps
            )
            self.tb_logger.add_scalar(
                "val/ep_total_time", info["metrics"]["total_time"], n_eps
            )
            self.tb_logger.add_scalar(
                "val/ep_total_distance", info["metrics"]["total_distance"], n_eps
            )
            self.tb_logger.add_scalar(
                "val/ep_avg_speed", info["metrics"]["average_speed_kph"], n_eps
            )
            self.tb_logger.add_scalar(
                "val/ep_avg_disp_err",
                info["metrics"]["average_displacement_error"],
                n_eps,
            )
            self.tb_logger.add_scalar(
                "val/ep_traj_efficiency",
                info["metrics"]["trajectory_efficiency"],
                n_eps,
            )
            self.tb_logger.add_scalar(
                "val/ep_traj_admissibility",
                info["metrics"]["trajectory_admissibility"],
                n_eps,
            )
            self.tb_logger.add_scalar(
                "val/movement_smoothness",
                info["metrics"]["movement_smoothness"],
                n_eps,
            )
        except:
            pass

        # TODO: Find a better way: requires knowledge of child class API :(
        if "safety_info" in self.metadata:
            self.tb_logger.add_scalar(
                "val/ep_interventions",
                self.metadata["safety_info"]["ep_interventions"],
                n_eps,
            )

    def log_train_metrics_to_tensorboard(self, ep_ret, t, t_start):
        self.tb_logger.add_scalar("train/episodic_return", ep_ret, self.episode_num)
        self.tb_logger.add_scalar(
            "train/ep_total_time",
            self.metadata["info"]["metrics"]["total_time"],
            self.episode_num,
        )
        self.tb_logger.add_scalar(
            "train/ep_total_distance",
            self.metadata["info"]["metrics"]["total_distance"],
            self.episode_num,
        )
        self.tb_logger.add_scalar(
            "train/ep_avg_speed",
            self.metadata["info"]["metrics"]["average_speed_kph"],
            self.episode_num,
        )
        self.tb_logger.add_scalar(
            "train/ep_avg_disp_err",
            self.metadata["info"]["metrics"]["average_displacement_error"],
            self.episode_num,
        )
        self.tb_logger.add_scalar(
            "train/ep_traj_efficiency",
            self.metadata["info"]["metrics"]["trajectory_efficiency"],
            self.episode_num,
        )
        self.tb_logger.add_scalar(
            "train/ep_traj_admissibility",
            self.metadata["info"]["metrics"]["trajectory_admissibility"],
            self.episode_num,
        )
        self.tb_logger.add_scalar(
            "train/movement_smoothness",
            self.metadata["info"]["metrics"]["movement_smoothness"],
            self.episode_num,
        )
        self.tb_logger.add_scalar("train/ep_n_steps", t - t_start, self.episode_num)

