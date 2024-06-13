import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import OrderedDict
import robomimic.models.obs_nets as ObsNets
import robomimic.models.value_nets as ValueNets
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.algo import register_algo_factory_func, PolicyAlgo, ValueAlgo

# Replay buffer class for experience replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Register PPO algorithm
@register_algo_factory_func("ppo")
def algo_config_to_class(algo_config):
    return PPO, {}

class PPO(PolicyAlgo, ValueAlgo):
    def __init__(self, **kwargs):
        PolicyAlgo.__init__(self, **kwargs)
        self._create_networks()

        # Set learning rates from config
        self.optimizer_actor = optim.Adam(self.nets['actor'].parameters(),
                                          lr=self.algo_config.optim_params.actor.learning_rate.initial)
        self.optimizer_critic = optim.Adam(self.nets['critic'].parameters(),
                                           lr=self.algo_config.optim_params.critic.learning_rate.initial)

        # Set hyperparameters from config
        self.eps_clip = self.algo_config.eps_clip
        self.gamma = self.algo_config.discount
        self.lamda = self.algo_config.lamda
        self.ppo_update_steps = self.algo_config.ppo_update_steps
        self.bc_loss_weight = self.algo_config.get("bc_loss_weight", 2.5)  # Default to 2.5 if not specified

    def _create_networks(self):
        self.nets = nn.ModuleDict()
        self._create_actor()
        self._create_critic()
        self.nets = self.nets.float().to(self.device)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.nets.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def _create_actor(self):
        actor_class = ActorNetwork
        actor_args = {
            'obs_shapes': self.obs_shapes,
            'goal_shapes': self.goal_shapes,
            'ac_dim': self.ac_dim,
            'mlp_layer_dims': self.algo_config.actor.layer_dims,
            'encoder_kwargs': self._get_encoder_kwargs()
        }
        self.nets['actor'] = actor_class(**actor_args)

    def _create_ritic(self):
        critic_class = ValueNets.ValueNetwork
        critic_args = {
            'obs_shapes': self.obs_shapes,
            'goal_shapes': self.goal_shapes,
            'mlp_layer_dims': self.algo_config.critic.layer_dims,
            'encoder_kwargs': self._get_encoder_kwargs()
        }
        self.nets['critic'] = critic_class(**critic_args)

    def _get_encoder_kwargs(self):
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)
        if 'obs' not in encoder_kwargs:
            encoder_kwargs['obs'] = {}
        encoder_kwargs['obs']['feature_activation'] = nn.ReLU
        return encoder_kwargs

    def process_batch_for_training(self, batch):
        input_batch = {
            'obs': {k: batch['obs'][k].squeeze(1) for k in batch['obs'] if batch['obs'][k] is not None},
            'next_obs': {k: batch['next_obs'][k].squeeze(1) for k in batch['next_obs'] if batch['next_obs'][k] is not None},
            'goal_obs': batch.get('goal_obs', None),
            'actions': batch['actions'].squeeze(1) if batch['actions'] is not None else None,
            'rewards': batch['rewards'].squeeze() if batch['rewards'] is not None else None,
            'dones': batch['dones'].squeeze() if batch['dones'] is not None else None,
        }
        self._check_for_nan_in_data(input_batch)
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def pretrain_bc(self, expert_data, num_pretrain_epochs):
        for epoch in range(num_pretrain_epochs):
            bc_loss = self._compute_bc_loss(expert_data['obs'], expert_data['actions'])
            self.optimizer_actor.zero_grad()
            bc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.nets['actor'].parameters(), max_norm=self.algo_config.max_grad_norm)
            self.optimizer_actor.step()

    def train_on_batch(self, batch, epoch, validate=False):
        processed_batch = self.process_batch_for_training(batch)
        obs, actions, rewards, dones, next_obs = processed_batch['obs'], processed_batch['actions'], processed_batch['rewards'], processed_batch['dones'], processed_batch['next_obs']
        old_log_probs = self._get_log_prob(obs, actions).detach()

        for _ in range(self.ppo_update_steps):
            log_probs = self._get_log_prob(obs, actions)
            ratios = torch.exp(log_probs - old_log_probs)
            advantages = self._compute_advantages(rewards, dones, obs, next_obs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Compute BC loss and combine with PPO loss
            bc_loss = self._compute_bc_loss(obs, actions)
            dynamic_bc_loss_weight = self.bc_loss_weight * (1 - epoch / self.algo_config.num_epochs)  # 逐步减少BC损失权重
            combined_actor_loss = actor_loss + dynamic_bc_loss_weight * bc_loss

            critic_loss = self._compute_critic_loss(obs, rewards, dones, next_obs)

            if not validate:
                self.optimizer_actor.zero_grad()
                combined_actor_loss.backward()
                self._check_for_nan(self.nets['actor'], name="actor")
                torch.nn.utils.clip_grad_norm_(self.nets['actor'].parameters(), max_norm=self.algo_config.max_grad_norm)
                self.optimizer_actor.step()

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                self._check_for_nan(self.nets['critic'], name="critic")
                torch.nn.utils.clip_grad_norm_(self.nets['critic'].parameters(), max_norm=self.algo_config.max_grad_norm)
                self.optimizer_critic.step()

        return {'actor_loss': combined_actor_loss.item(), 'critic_loss': critic_loss.item()}

    def _get_log_prob(self, obs, actions):
        mu, log_std = self.nets['actor'](obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        log_probs = dist.log_prob(actions).sum(-1, keepdim=True)
        return log_probs

    def _compute_advantages(self, rewards, dones, obs, next_obs):
        values = self.nets['critic'](obs).detach().squeeze()
        next_values = self.nets['critic'](next_obs).detach().squeeze()
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = torch.zeros_like(rewards).to(self.device)
        advantage = 0
        for t in reversed(range(len(rewards))):
            advantage = deltas[t] + self.gamma * self.lamda * (1 - dones[t]) * advantage
            advantages[t] = advantage
        return advantages

    def _compute_critic_loss(self, obs, rewards, dones, next_obs):
        values = self.nets['critic'](obs).squeeze()
        targets = rewards + self.gamma * self.nets['critic'](next_obs).squeeze() * (1 - dones)
        return nn.MSELoss()(values, targets)

    def _compute_bc_loss(self, obs, actions):
        # Compute behavior cloning (BC) loss
        mu, _ = self.nets['actor'](obs)
        bc_loss = nn.MSELoss()(mu, actions)
        return bc_loss

    def _check_for_nan(self, module, name):
        for param_name, param in module.named_parameters():
            if param.grad is None:
                print(f"Warning: Gradient for {name}.{param_name} is None")
            elif torch.isnan(param.grad).any():
                print(f"NaN detected in gradients of {name}.{param_name}")
                print(f"grad: {param.grad}")
                raise ValueError(f"NaN detected in gradients of {name}.{param_name}")

    def _check_for_nan_in_data(self, data):
        if isinstance(data, dict):
            for key, value in data.items():
                if value is None:
                    print(f"Warning: {key} is None")
                elif isinstance(value, dict):
                    self._check_for_nan_in_data(value)
                elif torch.isnan(value).any() or torch.isinf(value).any():
                    print(f"NaN or Inf detected in {key}")
                    print(f"value: {value}")
                    raise ValueError(f"NaN or Inf detected in {key}")
        elif isinstance(data, torch.Tensor):
            if torch.isnan(data).any() or torch.isinf(data).any():
                print("NaN or Inf detected in tensor")
                raise ValueError("NaN or Inf detected in tensor")

    def log_info(self, info):
        loss_log = OrderedDict()
        loss_log["Actor/Loss"] = info["actor_loss"]
        loss_log["Critic/Loss"] = info["critic_loss"]
        return loss_log

    def set_train(self):
        self.nets.train()

    def set_eval(self):
        self.nets.eval()

    def on_epoch_end(self, epoch):
        if self.lr_schedulers["critic"] is not None:
            for lr_sc in self.lr_schedulers["critic"]:
                if lr_sc is not None:
                    lr_sc.step()

        if self.lr_schedulers["actor"] is not None:
            self.lr_schedulers["actor"].step()

    def get_action(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        obs = TensorUtils.to_tensor(obs_dict)
        with torch.no_grad():
            mu, log_std = self.nets['actor'](obs)
            std = log_std.exp()
            dist = Normal(mu, std)
            action = dist.sample()
        return action

    def get_state_value(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        obs = TensorUtils.to_tensor(obs_dict)
        with torch.no_grad():
            value = self.nets['critic'](obs)
        return value

class ActorNetwork(nn.Module):
    def __init__(self, obs_shapes, goal_shapes, ac_dim, mlp_layer_dims, encoder_kwargs):
        super(ActorNetwork, self).__init__()
        self.encoder = ObsNets.obs_encoder_factory(obs_shapes, encoder_kwargs=encoder_kwargs)
        output_shape = self.encoder.output_shape()
        self.mlp = nn.Sequential(
            nn.Linear(output_shape[0], mlp_layer_dims[0]),
            nn.ReLU(),
            nn.Linear(mlp_layer_dims[0], mlp_layer_dims[1]),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(mlp_layer_dims[1], ac_dim)
        self.log_std_layer = nn.Linear(mlp_layer_dims[1], ac_dim)

    def forward(self, obs_dict, goal_dict=None):
        h = self.encoder(obs_dict)
        h = self.mlp(h)
        mu = self.mu_layer(h)
        log_std = self.log_std_layer(h)
        return mu, log_std
