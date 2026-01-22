from dataclasses import dataclass, field
from typing import Sequence, Union

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import Actor, ContinuousCritic, SACPolicy


@dataclass
class SACAgentSB3Config:
    """Configuration knobs that mirror the SAC JAX implementation."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    buffer_size: int = 3_000_000
    batch_size: int = 256
    learning_starts: int = 5_000
    train_freq: Union[int, tuple[int, str]] = 1
    gradient_steps: int = 1
    ent_coef: Union[str, float] = "auto"
    target_entropy: Union[str, float] = "auto"
    log_std_init: float = -3.0
    actor_hidden: Sequence[int] = field(default_factory=lambda: [256, 256])
    critic_hidden: Sequence[int] = field(default_factory=lambda: [512, 512, 512])
    tensorboard_log: str | None = None
    verbose: int = 1
    device: str = "auto"
    policy_kwargs: dict | None = None
    seed: int = 0

    def build_policy_kwargs(self) -> dict:
        if self.policy_kwargs is not None:
            return dict(self.policy_kwargs)
        return {
            "log_std_init": self.log_std_init,
            "actor_hidden": list(self.actor_hidden),
            "critic_hidden": list(self.critic_hidden),
        }


class AsymSACPolicy(SACPolicy):
    """Custom SAC policy with separate actor/critic hidden sizes."""

    def __init__(self, *args, actor_hidden=None, critic_hidden=None, **kwargs):
        self._actor_hidden = actor_hidden or [256, 256]
        self._critic_hidden = critic_hidden or [512, 512, 512]
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor: BaseFeaturesExtractor | None = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_kwargs.update(net_arch=self._actor_hidden)
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: BaseFeaturesExtractor | None = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        critic_kwargs.update(net_arch=self._critic_hidden)
        return ContinuousCritic(**critic_kwargs).to(self.device)


def _wrap_env(env: gym.Env | VecEnv) -> gym.Env | VecEnv:
    if isinstance(env, gym.Env) and not isinstance(env, VecEnv):
        return env
    return env


class SACAgentSB3:
    """Lightweight wrapper around Stable-Baselines3 SAC models."""

    def __init__(self, env: gym.Env | VecEnv, config: SACAgentSB3Config):
        self.env = env
        self.config = config
        policy_kwargs = self.config.build_policy_kwargs()
        self.model = SAC(
            policy=AsymSACPolicy,
            env=_wrap_env(env),
            gamma=config.gamma,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            learning_starts=config.learning_starts,
            train_freq=config.train_freq,
            gradient_steps=config.gradient_steps,
            tau=config.tau,
            ent_coef=config.ent_coef,
            target_entropy=config.target_entropy,
            policy_kwargs=policy_kwargs,
            tensorboard_log=config.tensorboard_log,
            verbose=config.verbose,
            device=config.device,
            seed=config.seed,
        )

    def learn(self, total_timesteps: int, **learn_kwargs):
        return self.model.learn(total_timesteps=total_timesteps, **learn_kwargs)

    def predict(self, observation, deterministic: bool = False):
        return self.model.predict(observation, deterministic=deterministic)

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = SAC.load(path, env=self.env)
