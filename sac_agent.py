import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------
# Replay Buffer
# --------------------------------------------------------
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, task_dim=0, size=int(1e6)):
        self.obs_buf = np.zeros((size, obs_dim + task_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim + task_dim), dtype=np.float32)
        self.acts_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)
        self.max_size = size
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=256):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )
        return {k: torch.as_tensor(v, device=device) for k, v in batch.items()}


# --------------------------------------------------------
# Networks
# --------------------------------------------------------
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit,
                 hidden_sizes=None, log_std_min=-20, log_std_max=2):
        if hidden_sizes is None:
            hidden_sizes = (256, 256)
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), nn.ReLU, nn.ReLU)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.act_limit = act_limit

    def forward(self, obs):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs):
        mu, std = self.forward(obs)
        normal = torch.distributions.Normal(mu, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = self.act_limit * y_t

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.act_limit * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)

        mu_action = self.act_limit * torch.tanh(mu)
        return action, log_prob, mu_action

    def act(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mu, _ = self.forward(obs_t)
                action = self.act_limit * torch.tanh(mu)
            else:
                action, _, _ = self.sample(obs_t)
        return action.cpu().numpy()[0]


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = (1024, 1024, 1024)
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], nn.ReLU, nn.Identity)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)


# --------------------------------------------------------
# SAC Agent
# --------------------------------------------------------
class SACAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        act_limit,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        lr=3e-4,
        hidden_actor=None,
        hidden_critic=None,
        target_entropy=None,
        automatic_entropy_tuning=True,
    ):
        if hidden_actor is None:
            hidden_actor = (256, 256)
        if hidden_critic is None:
            hidden_critic = (1024, 1024, 1024)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.act_limit = act_limit
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.actor = GaussianPolicy(obs_dim, act_dim, act_limit, hidden_actor).to(device)

        self.q1 = QNetwork(obs_dim, act_dim, hidden_critic).to(device)
        self.q2 = QNetwork(obs_dim, act_dim, hidden_critic).to(device)
        self.q1_target = QNetwork(obs_dim, act_dim, hidden_critic).to(device)
        self.q2_target = QNetwork(obs_dim, act_dim, hidden_critic).to(device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr)

        if target_entropy is None:
            target_entropy = -act_dim
        self.target_entropy = target_entropy

        if automatic_entropy_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        else:
            self.log_alpha = None

    def update(self, replay_buffer, batch_size=256):
        batch = replay_buffer.sample_batch(batch_size)
        obs, next_obs = batch["obs"], batch["obs2"]
        acts, rews, done = batch["acts"], batch["rews"], batch["done"]

        # --------------------
        # Critic Update
        # --------------------
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_obs)
            q1_next = self.q1_target(next_obs, next_action)
            q2_next = self.q2_target(next_obs, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob.squeeze(-1)
            target_q = rews.squeeze(-1) + (1 - done.squeeze(-1)) * self.gamma * q_next

        q1_pred = self.q1(obs, acts)
        q2_pred = self.q2(obs, acts)

        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # --------------------
        # Actor Update
        # --------------------
        new_actions, log_prob, _ = self.actor.sample(obs)
        q1_new = self.q1(obs, new_actions)
        q2_new = self.q2(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_prob - q_new.unsqueeze(-1)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --------------------
        # Entropy Tuning
        # --------------------
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # --------------------
        # W&B Logging
        # --------------------
        wandb.log({
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
        })

    def act(self, obs, deterministic=False):
        return self.actor.act(obs, deterministic)
