import copy
import torch
from torch import nn

import networks
import tools

to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )
        self.heads = nn.ModuleDict()
        # Latent state features (z, h) used for all heads. 
        if config.dyn_discrete: # If using discrete latent variables
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else: # If using continuous latent variables
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        ) # Decoder head that reconstructs the input observation from the latent state.
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        ) # Reward head that predicts the reward based on the latent state.
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        ) # Continuation head that predicts whether the episode continues based on the latent state.
        for name in config.grad_heads:
            assert name in self.heads, name
        # Optimiser for the world model, which optimizes the parameters of the encoder, dynamics, and heads based on the combined loss from all heads and the KL divergence loss from the dynamics model.
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        with tools.RequiresGrad(self): # Training
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                """
                Observe the latent state by passing the embedded observation, action, and is_first flag to the dynamics model.
                 - embed: the embedded observation from the encoder.
                 - data["action"]: the action taken at each time step.
                 - data["is_first"]: a binary flag indicating whether the current time step is the first step of an episode. This is used to reset the hidden state of the dynamics model at the beginning of each episode.
                 - post: the posterior distribution of the latent state after observing the current time step.
                 - prior: the prior distribution of the latent state before observing the current time step. This is typically obtained from the previous time step's posterior and the action taken.
                 The observe function returns both the posterior and prior distributions, which are then used to calculate the KL divergence loss and to make predictions for various heads (e.g., reward, continuation).
                """
                # self.dynamics.observe baiscally runs the RSSM model forward for one step, giving results from both posterior and prior. The posterior is used for training the model, while the prior is used for imagination during policy learning.
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                """
                kl_free is the free nats for KL divergence, which is a threshold below which the KL divergence loss will not be optimised. 
                This is a common technique to prevent the model from collapsing the latent space too early in training. By setting a free nats threshold, 
                you allow the model some flexibility in how much it needs to match the prior distribution, which can help with learning more useful representations in the latent space.
                """
                kl_free = self._config.kl_free
                """
                dunamic loss scale for KL divergence and representation loss.
                 - dyn_scale: the scale for the KL divergence loss from the dynamics model. 
                              This loss encourages the posterior distribution to be close to the prior distribution, which helps to regularize the latent space and prevent overfitting.
                 - rep_scale: the scale for the representation loss, which is typically the negative log-likelihood of the observed data under the model's predictions. 
                              This loss encourages the model to learn representations that can accurately reconstruct the input observations and predict rewards and continuation signals. By adjusting
                """
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                """
                Calculate the KL divergence between the posterior and prior distributions of the latent state.
                """
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                """
                Iterate over the heads defined in the model and make predictions based on the features extracted from the posterior distribution of the latent state.
                DreamerV3 has:
                - a decoder head that reconstructs the input observation (e.g., image) from the latent state.
                - a reward head that predicts the reward based on the latent state.
                - a continuation head that predicts whether the episode will continue or terminate based on the latent state.
                """
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                """
                losses is a dictionary that stores the negative log-likelihood loss for each head.
                It should normally inlude:
                - reward_loss: the negative log-likelihood of the observed rewards under the reward head's predictions.
                - cont_loss: the negative log-likelihood of the observed continuation signals under the continuation head's predictions.
                - image_loss: the negative log-likelihood of the observed images under the decoder head's predictions.
                """
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        """
        kl_free: the free nats for KL divergence.
        if kl_free is 0, then the KL divergence will be fully optimized.
        if kl_free is 1, then the KL divergence will not be optimized at all.
        who decides the value of kl_free? it's a hyperparameter that you can tune.
         - if kl_free is too high, then the model will not learn anything useful.
         - if kl_free is too low, then the model will learn to ignore the latent state
        """
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        """
        Preprocess the raw observation data, including:
            - Converting the observation data into PyTorch tensors and moving them to the appropriate device (e.g., GPU).
            - Normalizing the image data by scaling pixel values to the range [0, 1].
            - Handle the discount factor by scaling it with the configured discount.
            - Ensure 'is_first' and 'is_terminal' flags exist in the observation.
        """
        obs = {
            k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
            for k, v in obs.items()
        }
        obs["image"] = obs["image"] / 255.0
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = obs["discount"].unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1) # for metaworld multi camera setup, shape is (6, time, h, w, 3*num_cameras)
        # turn into (6, time, h, w*num_cameras, 3)
        b, t, h, w, c = model.shape
        num_cameras = c // 3
        model = model.reshape(b, t, h, w, num_cameras, 3).permute(0, 1, 2, 4, 3, 5).reshape(b, t, h, num_cameras * w, 3)
        truth = data["image"][:6] # shape is (6, time, h, w, 3*num_cameras)
        truth = truth.reshape(b, t, h, w, num_cameras, 3).permute(0, 1, 2, 4, 3, 5).reshape(b, t, h, num_cameras * w, 3)
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):
    """
    This is the actor-critic, but it never sees the real environment. It learns entirely from imagined trajectories produced by the world model.
    Contains:
        - actor: the policy network that takes in the latent state features and outputs a distribution over actions.
        - value: the value network that takes in the latent state features and outputs a distribution over returns (or values).
        - slow_value: a slowly updated copy of the value network used for stabilizing training (optional, based on config).
        - optimizers for both the actor and value networks.
    """
    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False # Whether to use automatic mixed precision for training, which can speed up training and reduce memory usage on compatible hardware.
        self._config = config
        self._world_model = world_model # The world model is used to generate imagined trajectories for training the actor and critic. It provides the dynamics and reward predictions.
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        """
        Actor network is an MLP that:
            - takes in the latent state features (z, h) as input.
            - outputs a distribution over actions, which can be either continuous (e.g., Gaussian) or discrete (e.g., categorical), depending on the configuration.
        The actor is trained to maximize the expected return of the imagined trajectories, 
        which is computed using the value network and reward predictions from the world model.
        """
        self.actor = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        )
        """
        Value network is an MLP that:
            - takes in the latent state features (z, h) as input.
            - outputs a distribution over returns (or values). The type of distribution can be configured (e.g., Gaussian, categorical, or symlog discrete).
        The value network is trained to predict the expected return of the imagined trajectories, 
        which is computed using the reward predictions from the world model.
        """
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer(
                "ema_vals", torch.zeros((2,), device=self._config.device)
            )
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start,
        objective,
    ):
        """
        Training the actor and critic using imagined trajectories generated by the world model. The training process involves:
            - Generating imagined trajectories by rolling out the world model starting from the given initial latent state (start) and using the current actor policy to select actions.
            - Computing the rewards for the imagined trajectories using the provided objective function.
            - Computing the target values for the critic using the rewards and the value predictions from the world model.
            - Updating the actor by maximizing the expected return of the imagined trajectories, which is computed using the rewards and the value predictions.
            - Updating the critic by minimizing the difference between the predicted values and the target values.
        """
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                # Run the world model forward in imagination mode to generate imagined trajectories.
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon
                )
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(imag_feat).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled by ema or sym_log.
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base,
                )
                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor["dist"] in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon):
        """
        Perform imagination by rolling out the world model with a given initial latent state (start), and using a policy, for a specified horizon.
        """
        dynamics = self._world_model.dynamics # World model dynamics model
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state) # Concatenate the stochastic and deterministic parts of the latent state to get the latent features
            inp = feat.detach()
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action) # World model imagination step: given current latent (z,h) and action, returns a distribution over next stochastic latent (z)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward):
        """
        
        """
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
    ):
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0]) # EMA_005 is the 5th percentile of the reward distribution, which can be used as a reference point for normalizing rewards and stabilizing training.
            metrics["EMA_095"] = to_np(self.ema_vals[1]) # EMA_095 is the 95th percentile of the reward distribution, which can be used as a reference point for normalizing rewards and stabilizing training.

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1