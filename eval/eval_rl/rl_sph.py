# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

base_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
)
sys.path.append(base_dir)

import warp as wp

wp.config.verify_cuda = True

import argparse
import torch
import yaml
from rl_games.torch_runner import Runner

from envs.rlgames_env_wrapper import register_env, RLGPUAlgoObserver
from envs.neural_environment import NeuralEnvironment
from utils.python_utils import set_random_seed, get_time_stamp
from envs.warp_sim_envs import RenderMode

import numpy as np

# Loads from Pytorch 2.2.2 .pth so trust old np types
torch.serialization.add_safe_globals(
    [np.dtypes.Float32DType, np.core.multiarray.scalar, np.dtype]
)

import sph 


class FluidEnvironment(NeuralEnvironment):
    def __init__(
        self,
        env_name,
        num_envs,
        warp_env_cfg=None,
        neural_integrator_cfg=None,
        neural_model=None,
        default_env_mode="neural",
        device="cuda:0",
        render=False,
    ):
        super().__init__(
            env_name,
            num_envs,
            warp_env_cfg,
            neural_integrator_cfg,
            neural_model,
            default_env_mode,
            device,
            render,
        )

        builder = wp.sim.ModelBuilder()
        urdf_path = "../../envs/warp_sim_envs/assets/anymal/urdf/anymal_joint_limits.urdf"
        wp.sim.parse_urdf(urdf_path, 
            builder, 
            floating=True,
            stiffness=85.0,
            damping=2.0,
            armature=0.06,
            contact_ke=2.0e3,
            contact_kd=5.0e2,
            contact_kf=1.0e2,
            contact_mu=0.75,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False
        )
        self.robot_model = builder.finalize()
        self.scale_robot_model(2.0)
        self.robot_state = self.robot_model.state()

        # From example_sph.py
        self.verbose = False
        self.sim_time = 0.0

        # simulation params
        self.smoothing_length = 0.1  # NOTE change this to adjust number of particles
        self.width = 1.0  # x
        self.height = 1.0  # y
        self.length = 1.0  # z
        self.isotropic_exp = 20
        self.base_density = 1.0
        self.particle_mass = (
            0.01 * self.smoothing_length**3
        )  # reduce according to smoothing length
        self.fluid_dt = (
            0.01 * self.smoothing_length
        )  # decrease sim dt by smoothing length
        self.dynamic_visc = 0.025
        self.damping_coef = -0.95
        self.gravity = -9.81
        self.n = int(
            self.height
            * (self.width / 4.0)
            * (self.height / 4.0)
            / (self.smoothing_length**3)
        )  # number particles (small box in corner)
        self.sim_step_to_frame_ratio = int(32 / self.smoothing_length)

        # constants
        self.density_normalization = (315.0 * self.particle_mass) / (
            64.0 * np.pi * self.smoothing_length**9
        )  # integrate density kernel
        self.pressure_normalization = -(45.0 * self.particle_mass) / (
            np.pi * self.smoothing_length**6
        )
        self.viscous_normalization = (45.0 * self.dynamic_visc * self.particle_mass) / (
            np.pi * self.smoothing_length**6
        )

        # allocate arrays
        self.x = wp.empty(self.n, dtype=wp.vec3, device=self.device)
        self.v = wp.zeros(self.n, dtype=wp.vec3, device=self.device)
        self.rho = wp.zeros(self.n, dtype=float, device=self.device)
        self.a = wp.zeros(self.n, dtype=wp.vec3, device=self.device)

        # 4. set random positions
        wp.launch(
            kernel=sph.initialize_particles,
            dim=self.n,
            inputs=[
                self.x,
                self.smoothing_length,
                self.width,
                self.height,
                self.length,
            ],
        )  # initialize in small area

        # create hash array
        grid_size = int(self.height / (4.0 * self.smoothing_length))
        self.grid = wp.HashGrid(grid_size, grid_size, grid_size, device=self.device)

    def step(self, actions: torch.Tensor, env_mode=None) -> torch.Tensor:
        """
        Step robot, step fluid (one-way coupling)
        """
        states = super().step(actions, env_mode)
        self._step_fluid() # fluids are not part of states to keep it pure
        return states

    def step_with_joint_act(
        self, joint_acts: torch.Tensor, env_mode=None
    ) -> torch.Tensor:
        """
        Step robot, step fluid (one-way coupling)
        """
        states = super().step_with_joint_act(joint_acts, env_mode)
        self._step_fluid() # fluids are not part of states to keep it pure
        return states

    def _step_fluid(self):
        # from example_sph.py
        # TODO: sync with robot
        for _ in range(self.sim_step_to_frame_ratio):
            # build grid
            self.grid.build(self.x, self.smoothing_length)

            # compute density of points
            wp.launch(
                kernel=sph.compute_density,
                dim=self.n,
                inputs=[
                    self.grid.id,
                    self.x,
                    self.rho,
                    self.density_normalization,
                    self.smoothing_length,
                ],
                device=self.device,
            )

            # get new acceleration
            wp.launch(
                kernel=sph.get_acceleration,
                dim=self.n,
                inputs=[
                    self.grid.id,
                    self.x,
                    self.v,
                    self.rho,
                    self.a,
                    self.isotropic_exp,
                    self.base_density,
                    self.gravity,
                    self.pressure_normalization,
                    self.viscous_normalization,
                    self.smoothing_length,
                ],
                device=self.device,
            )

            # apply bounds
            wp.launch(
                kernel=sph.apply_bounds,
                dim=self.n,
                inputs=[
                    self.x,
                    self.v,
                    self.damping_coef,
                    self.width,
                    self.height,
                    self.length,
                ],
                device=self.device,
            )

            # TODO: collision kernel
            wp.launch(
                kernel=sph.collide_particles_with_robot,
                dim=self.n,
                inputs=[
                    self.x, self.v,
                    self.robot_state.body_q,
                    self.robot_model.shape_transform,
                    self.robot_model.shape_body,
                    self.robot_model.shape_geo,
                    self.robot_model.shape_count,
                    0.5,
                    self.smoothing_length * 0.5
                ]
            )

            # kick
            wp.launch(
                kernel=sph.kick,
                dim=self.n,
                inputs=[self.v, self.a, self.fluid_dt],
                device=self.device,
            )

            #drift
            wp.launch(
                kernel=sph.drift,
                dim=self.n,
                inputs=[self.x, self.v, self.fluid_dt],
                device=self.device,
            )

        self.sim_time += self.frame_dt

    def render(self):
        super().render()
        if hasattr(self.env, "renderer") and self.env.renderer:
            self.env.renderer.begin_frame(self.sim_time)
            self.env.renderer.render_points(
                points=self.x, radius=self.smoothing_length * 0.5, name="fluid"
            )
            self.env.renderer.end_frame()

    def scale_robot_model(self, scale_factor):
        joint_X_np = self.robot_model.joint_X_p.numpy()
        joint_X_np[:, :3] *= scale_factor
        self.robot_model.joint_X_p = wp.from_numpy(joint_X_np, dtype=wp.transform)
        shape_scale_np = self.robot_model.shape_geo.scale.numpy()
        shape_scale_np *= scale_factor
        self.robot_model.shape_geo.scale = wp.from_numpy(shape_scale_np, dtype=wp.vec3)
        shape_X_np = self.robot_model.shape_transform.numpy()
        shape_X_np[:, :3] *= scale_factor
        self.robot_model.shape_transform = wp.from_numpy(shape_X_np, dtype=wp.transform)
        


def get_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument(
        "--rl-cfg",
        default="./cfg/Cartpole/cartpole.yaml",
        type=str,
        help="Path to rl config file.",
    )
    # Some command-line overriding parameters to provide a flexible experiment experience.
    parser.add_argument(
        "--exp-name", default=None, type=str, help="Name for rl experiment."
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Whether not to include timestamp in exp folder name.",
    )
    parser.add_argument(
        "--env-mode",
        default=None,
        type=str,
        choices=["neural", "ground-truth"],
        help="Environment mode: in neural the policy is trained "
        "or played back in the neural simulation, in ground-truth "
        "the Warp simulation is used.",
    )
    parser.add_argument(
        "--nerd-model-path",
        default=None,
        type=str,
        help="Path to the neural sim model *.pt file. "
        "Required for neural environment mode.",
    )
    parser.add_argument(
        "--num-envs",
        default=None,
        type=int,
        help="Number of environments to run in parallel.",
    )
    parser.add_argument(
        "--max-episode-length",
        default=None,
        type=int,
        help="Maximum number of time steps to use for a policy "
        "roll-out before the environment is reset.",
    )
    parser.add_argument(
        "--max-epochs", default=None, type=int, help="Number of PPO epochs."
    )
    parser.add_argument(
        "--horizon-length", default=None, type=int, help="Horizon length for PPO"
    )
    parser.add_argument(
        "--num-games", default=None, type=int, help="Number of games to play."
    )

    # other params
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--render-mode",
        default="human",
        type=str,
        choices=["none", "human", "rgb_array"],
        help="Render mode set to 'none' will not have any rendering, "
        "'human' will render the policy roll-out in an OpenGL-based "
        "renderer to observe the training interactively, "
        "'rgb_array' will render the training environments to separate "
        "tiles and use these tile RGB images as observation space "
        "for the policy.",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--playback",
        default=None,
        type=str,
        help="Path to the pretrained policy which rl_games saves "
        "typically to a *.pth file. If provided, the policy "
        "will be played back in the environment for testing only.",
    )
    parser.add_argument("--export-video", action="store_true")
    parser.add_argument("--export-video-path", type=str, default="video.gif")
    parser.add_argument("--export-usd", action="store_true")

    args = parser.parse_args()

    return args


def load_rl_config(args):
    if args.playback is not None:
        policy_dir = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(args.playback)), "../")
        )
        rl_cfg_path = os.path.join(policy_dir, "rl_cfg.yaml")
    else:
        rl_cfg_path = args.rl_cfg

    with open(rl_cfg_path, "r") as f:
        rl_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # overrides command-line arguments
    if args.exp_name is not None:
        if args.no_timestamp:
            rl_cfg["rl"]["config"]["full_experiment_name"] = args.exp_name
        else:
            full_experiment_name = "{}/{}".format(args.exp_name, get_time_stamp())
            rl_cfg["rl"]["config"]["full_experiment_name"] = full_experiment_name
    else:
        rl_cfg["rl"]["config"]["full_experiment_name"] = (
            f"{rl_cfg['env']['env_name']}PPO"
        )

    if args.env_mode is not None:
        rl_cfg["env"]["env_mode"] = args.env_mode
    if args.nerd_model_path is not None:
        rl_cfg["env"]["model_path"] = args.nerd_model_path
    if args.num_envs is not None:
        rl_cfg["env"]["num_envs"] = args.num_envs
    if args.max_episode_length is not None:
        rl_cfg["env"]["max_episode_length"] = args.max_episode_length
    if args.max_epochs is not None:
        rl_cfg["rl"]["config"]["max_epochs"] = args.max_epochs
    if args.horizon_length is not None:
        rl_cfg["rl"]["config"]["horizon_length"] = args.horizon_length
    if args.num_games is not None:
        rl_cfg["rl"]["config"]["player"]["games_num"] = args.num_games
    rl_cfg["seed"] = args.seed

    return rl_cfg


"""
Load neural_integrator_cfg and set neural model.
Construct the neural env.
"""


def construct_env(env_specs, device, args):
    # copy seed into warp_env_cfg
    if "warp_env_cfg" not in env_specs:
        env_specs["warp_env_cfg"] = {}
    env_specs["warp_env_cfg"]["seed"] = args.seed
    env_specs["warp_env_cfg"]["setup_renderer"] = False
    if args.export_usd:
        args.render = True
        env_specs["warp_env_cfg"]["render_mode"] = RenderMode.USD

    # Load neural model and neural_integrator_cfg if env_mode is "neural"
    if env_specs["env_mode"] == "neural":
        neural_model, robot_name = torch.load(
            env_specs["model_path"], map_location=device, weights_only=False
        )
        neural_model.to(device)

        train_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(env_specs["model_path"])), "../"
            )
        )
        cfg_path = os.path.join(train_dir, "cfg.yaml")
        with open(cfg_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
        neural_integrator_cfg = cfg["env"]["neural_integrator_cfg"]
    else:
        neural_model = None
        neural_integrator_cfg = None

    env = FluidEnvironment(
        env_name=env_specs["env_name"],
        num_envs=env_specs["num_envs"],
        warp_env_cfg=env_specs["warp_env_cfg"],
        neural_integrator_cfg=neural_integrator_cfg,
        neural_model=neural_model,
        default_env_mode=env_specs["env_mode"],
        render=args.render,
    )

    if neural_model is not None:
        assert env.robot_name == robot_name, (
            "env.robot_name is not equal to neural_model's robot_name."
        )

    register_env(
        env,
        render_mode=args.render_mode,
        max_episode_length=env_specs["max_episode_length"],
        reward_bias=env_specs["reward_bias"],
        control_steps=env_specs.get("control_steps", 1),
        image_width=64,
        image_height=64,
    )

    return env


def construct_rlg_config(rl_cfg):
    rlg_config_dict = {}
    rlg_config_dict["params"] = {
        "seed": rl_cfg["seed"],
        "algo": {"name": "a2c_continuous"},
        "model": {"name": "continuous_a2c_logstd"},
        "network": {
            "name": "actor_critic",
            "separate": False,
            "space": {
                "continuous": {
                    "mu_activation": "None",
                    "sigma_activation": "None",
                    "mu_init": {"name": "default"},
                    "sigma_init": {"name": "const_initializer", "val": 0},
                    "fixed_sigma": rl_cfg["rl"]
                    .get("other_params", {})
                    .get("fixed_sigma", True),
                }
            },
            **rl_cfg["rl"]["network"],
        },
        "load_checkpoint": False,
        "load_path": "",
        "config": {
            **rl_cfg["rl"]["config"],
            "env_name": "warp",
            "multi_gpu": False,
            "ppo": True,
            "mixed_precision": True,
            "value_bootstrap": True,
            "num_actors": rl_cfg["env"]["num_envs"],
        },
    }
    return rlg_config_dict


def train_policy(runner):
    runner.run(
        {
            "train": True,
        }
    )


def evaluate_policy(runner, policy_path):
    results = runner.run(
        {
            "train": False,
            "play": True,
            "checkpoint": policy_path,
        }
    )
    return results


if __name__ == "__main__":
    args = get_args()

    device = "cuda:0"

    set_random_seed(args.seed)

    rl_config = load_rl_config(args)

    env = construct_env(rl_config["env"], device, args)

    rlg_config_dict = construct_rlg_config(rl_config)

    # save rl config
    if not args.playback:
        train_dir = os.path.join(
            "runs", rl_config["rl"]["config"]["full_experiment_name"]
        )
        os.makedirs(train_dir, exist_ok=False)
        yaml.dump(rl_config, open(os.path.join(train_dir, "rl_cfg.yaml"), "w"))

    runner = Runner(RLGPUAlgoObserver())
    runner.load(rlg_config_dict)
    runner.reset()

    if args.playback is not None and args.export_video:
        env.start_video_export(args.export_video_path)

    if args.playback is None:
        train_policy(runner)
    else:
        evaluate_policy(runner, args.playback)

    if args.playback is not None and args.export_video:
        env.end_video_export()
    if args.playback is not None and args.export_usd:
        env.save_usd()

    print("visited states range:")
    for i in range(len(env.visited_state_min)):
        print(
            "State {}: [{}, {}]".format(
                i, env.visited_state_min[i], env.visited_state_max[i]
            )
        )
    torch.save((env.visited_state_min, env.visited_state_max), "visited_state_range.pt")

    env.close()
