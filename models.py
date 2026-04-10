import copy
import torch
from torch import nn

import networks
import tools

to_np = lambda x: x.detach().cpu().numpy()


# ---------------------------------------------------------------------------
#  RewardEMA – running quantile tracker for reward normalisation
# ---------------------------------------------------------------------------
class RewardEMA:
    """Running mean and std based on exponential moving average of quantiles."""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # in-place update so the registered buffer is modified
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


# ===========================================================================
#  1. RSSMWorldModel – the shared backbone
# ===========================================================================
# Contains: encoder, RSSM dynamics, and decoder.
# These are task-agnostic and shared across all tasks in continual learning.
# The decoder lives here because it reconstructs observations (task-agnostic).
# ---------------------------------------------------------------------------
class RSSMWorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super().__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config

        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}

        # --- Encoder: raw observations -> embedding vector ---
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim

        # --- RSSM dynamics: (embed, action, is_first) -> posterior & prior ---
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

        # --- Decoder: latent features -> reconstructed observations ---
        # The decoder is part of the shared backbone because reconstructing
        # observations is task-agnostic (every task shares the same obs space).
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.feat_size = feat_size

        self.decoder = networks.MultiDecoder(feat_size, shapes, **config.decoder)

        # --- Optimizer for the shared RSSM + encoder + decoder ---
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
            f"Optimizer model_opt has "
            f"{sum(p.numel() for p in self.parameters())} variables."
        )

    # ---- helpers ----------------------------------------------------------

    def get_feat(self, state):
        """Convenience wrapper: latent state dict -> feature vector."""
        return self.dynamics.get_feat(state)

    def get_dist(self, state):
        """Convenience wrapper: latent state dict -> distribution."""
        return self.dynamics.get_dist(state)

    # ---- preprocessing (used both at training and rollout) ----------------

    def preprocess(self, obs):
        """Convert raw numpy observations to tensors, normalise images, and
        derive the continuation signal from is_terminal."""
        obs = {
            k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
            for k, v in obs.items()
        }
        obs["image"] = obs["image"] / 255.0
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            obs["discount"] = obs["discount"].unsqueeze(-1)
        assert "is_first" in obs   # needed to reset hidden state
        assert "is_terminal" in obs  # needed to derive continuation target
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        return obs

    # ---- training step (backbone only, called jointly with task heads) ----

    def observe(self, data):
        """Run encoder + RSSM observe on preprocessed data.

        Returns:
            embed:  (B, T, embed_dim)
            post:   posterior latent state dict
            prior:  prior latent state dict
        """
        embed = self.encoder(data)
        post, prior = self.dynamics.observe(
            embed, data["action"], data["is_first"]
        )
        return embed, post, prior

    def compute_kl_loss(self, post, prior):
        """Compute the KL divergence components between posterior and prior.

        Returns: kl_loss, kl_value, dyn_loss, rep_loss  (all tensors)."""
        return self.dynamics.kl_loss(
            post, prior,
            self._config.kl_free,
            self._config.dyn_scale,
            self._config.rep_scale,
        )

    def compute_decoder_loss(self, feat, data):
        """Compute the reconstruction loss from the decoder.

        Returns a dict  {obs_key: loss_tensor}  for every decoded modality."""
        preds = self.decoder(feat)
        # MultiDecoder returns a dict of distributions keyed by obs name
        if not isinstance(preds, dict):
            preds = {"image": preds}
        losses = {}
        for name, pred in preds.items():
            losses[name] = -pred.log_prob(data[name])
        return losses

    # ---- video prediction (for logging / visualisation) -------------------

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.decoder(self.dynamics.get_feat(states))["image"].mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.decoder(self.dynamics.get_feat(prior))["image"].mode()

        # observed image is given for the first 5 steps, then open-loop
        model = torch.cat([recon[:, :5], openl], 1)
        # handle multi-camera: (B,T,H,W,3*C) -> (B,T,H,C*W,3)
        b, t, h, w, c = model.shape
        num_cameras = c // 3
        model = (model.reshape(b, t, h, w, num_cameras, 3)
                 .permute(0, 1, 2, 4, 3, 5)
                 .reshape(b, t, h, num_cameras * w, 3))
        truth = data["image"][:6]
        truth = (truth.reshape(b, t, h, w, num_cameras, 3)
                 .permute(0, 1, 2, 4, 3, 5)
                 .reshape(b, t, h, num_cameras * w, 3))
        error = (model - truth + 1.0) / 2.0
        return torch.cat([truth, model, error], 2)


# ===========================================================================
#  2. TaskHeads – per-task reward & continuation heads
# ===========================================================================
# These are lightweight MLPs that predict reward and episode continuation
# from the latent features produced by the shared RSSM.
# A fresh TaskHeads is created for each new task.
# ---------------------------------------------------------------------------
class TaskHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config

        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.feat_size = feat_size

        # --- Reward head: latent features -> reward distribution ---
        self.reward = networks.MLP(
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
        )

        # --- Continuation head: latent features -> binary continue/stop ---
        self.cont = networks.MLP(
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
        )

        # Which heads propagate gradients back into the RSSM.
        # Typically ["decoder", "reward", "cont"] – but decoder now lives in
        # the RSSM, so here we only track reward / cont.
        self._grad_heads = set(config.grad_heads) - {"decoder"}

        # Loss scales (reward and cont may be weighted differently)
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )

        # --- Optimizer for the task heads only ---
        self._heads_opt = tools.Optimizer(
            "task_heads",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer task_heads_opt has "
            f"{sum(p.numel() for p in self.parameters())} variables."
        )

    def compute_losses(self, feat, data):
        """Predict reward and continuation from latent features and return
        per-head negative log-likelihood losses.

        Args:
            feat: (B, T, feat_size) latent features from the RSSM.
            data: preprocessed observation dict (must contain 'reward', 'cont').

        Returns:
            losses: dict  {"reward": tensor, "cont": tensor}
            preds:  dict  {"reward": dist, "cont": dist}
        """
        heads = {"reward": self.reward, "cont": self.cont}
        preds = {}
        losses = {}
        for name, head in heads.items():
            grad_head = name in self._grad_heads
            inp = feat if grad_head else feat.detach()
            pred = head(inp)
            preds[name] = pred
            losses[name] = -pred.log_prob(data[name])
        return losses, preds

    def scale_losses(self, losses):
        """Apply configured loss scales to each head's loss."""
        return {
            k: v * self._scales.get(k, 1.0) for k, v in losses.items()
        }


# ===========================================================================
#  3. ActorCritic – per-task policy and value function
# ===========================================================================
# Learns entirely from imagined trajectories produced by the RSSM.
# The RSSM is *not* stored as an attribute; it is passed into _train() and
# _imagine() so no parameter duplication occurs.
# A fresh ActorCritic is created for each new task.
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config

        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.feat_size = feat_size

        # --- Actor: latent features -> action distribution ---
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

        # --- Critic (value): latent features -> return distribution ---
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

        # --- Slow target network for stabilising critic training ---
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0

        # --- Optimizers (actor and critic are updated separately) ---
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
            f"Optimizer actor_opt has "
            f"{sum(p.numel() for p in self.actor.parameters())} variables."
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
            f"Optimizer value_opt has "
            f"{sum(p.numel() for p in self.value.parameters())} variables."
        )

        # --- Reward EMA for return normalisation ---
        if self._config.reward_EMA:
            self.register_buffer(
                "ema_vals", torch.zeros((2,), device=self._config.device)
            )
            self.reward_ema = RewardEMA(device=self._config.device)

    # ---- main training entry point ----------------------------------------

    def _train(self, start, objective, rssm, task_heads):
        """Train actor and critic on imagined trajectories.

        Args:
            start:      initial latent state dict (from RSSM posterior on real data).
            objective:  callable(feat, state, action) -> reward tensor.
                        Typically the reward head: task_heads.reward(feat).mode().
            rssm:       the shared RSSMWorldModel (passed in, NOT stored).
            task_heads: the TaskHeads for the current task (passed in, NOT stored).
                        Used to get continuation predictions for discount computation.

        Returns:
            imag_feat, imag_state, imag_action, weights, metrics
        """
        self._update_slow_target()
        metrics = {}

        # ---- Actor loss: maximise expected imagined return ----
        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon, rssm
                )
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(imag_feat).entropy()
                state_ent = rssm.get_dist(imag_state).entropy()

                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward, rssm, task_heads
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat, imag_action, target, weights, base,
                )
                # Entropy bonus encourages exploration
                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat

        # ---- Critic loss: predict lambda-returns ----
        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        # ---- Logging ----
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

        # ---- Parameter updates ----
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))

        return imag_feat, imag_state, imag_action, weights, metrics

    # ---- imagination rollout ----------------------------------------------

    def _imagine(self, start, policy, horizon, rssm):
        """Roll out the RSSM in imagination using the given policy.

        Args:
            start:   initial latent state dict.
            policy:  actor network (feat -> action distribution).
            horizon: number of imagination steps.
            rssm:    the shared RSSMWorldModel (for dynamics).

        Returns:
            feats:   (horizon, batch, feat_dim)
            states:  latent state dict, each (horizon, batch, ...).
            actions: (horizon, batch, act_dim)
        """
        dynamics = rssm.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
        return feats, states, actions

    # ---- target computation -----------------------------------------------

    def _compute_target(self, imag_feat, imag_state, reward, rssm, task_heads):
        """Compute lambda-returns as training targets for the critic.

        Args:
            rssm:       shared RSSMWorldModel (for get_feat).
            task_heads: TaskHeads (for continuation predictions).
        """
        # Discount = config.discount * P(continue)
        if task_heads is not None:
            inp = rssm.get_feat(imag_state)
            discount = self._config.discount * task_heads.cont(inp).mean
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

    # ---- actor loss -------------------------------------------------------

    def _compute_actor_loss(self, imag_feat, imag_action, target, weights, base):
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        target = torch.stack(target, dim=1)

        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

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

    # ---- slow target update -----------------------------------------------

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(
                    self.value.parameters(), self._slow_value.parameters()
                ):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1


# ===========================================================================
#  4. Joint training step – ties the three components together
# ===========================================================================
# This function orchestrates a single training step. It is intentionally a
# free function (not a method) so that no component needs to own the others.
# ---------------------------------------------------------------------------

def train_world_model_step(rssm, task_heads, data):
    """One training step for the shared RSSM backbone + current task heads.

    The RSSM (encoder, dynamics, decoder) and the task heads (reward, cont)
    are optimised jointly so that gradients from the task heads (if they are
    in grad_heads) flow back into the RSSM.

    Args:
        rssm:       RSSMWorldModel instance (shared across tasks).
        task_heads: TaskHeads instance (specific to the current task).
        data:       raw observation dict from the replay buffer.

    Returns:
        post:    posterior latent state dict (detached).
        context: dict with embed, feat, kl, postent (for actor-critic).
        metrics: dict of scalar metrics for logging.
    """
    config = rssm._config
    use_amp = rssm._use_amp
    data = rssm.preprocess(data)

    # We need gradients through both rssm and task_heads
    with tools.RequiresGrad(rssm):
        with tools.RequiresGrad(task_heads):
            with torch.cuda.amp.autocast(use_amp):
                # --- Forward through RSSM ---
                embed, post, prior = rssm.observe(data)

                # --- KL divergence loss ---
                kl_loss, kl_value, dyn_loss, rep_loss = rssm.compute_kl_loss(
                    post, prior
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape

                # --- Decoder (reconstruction) loss ---
                feat = rssm.get_feat(post)
                decoder_losses = rssm.compute_decoder_loss(feat, data)

                # --- Task head losses (reward, cont) ---
                # Use feat directly for grad_heads, detached otherwise
                head_losses, _ = task_heads.compute_losses(feat, data)

                # --- Combine all losses ---
                all_losses = {}
                all_losses.update(decoder_losses)
                all_losses.update(head_losses)

                # Apply loss scales (reward and cont may be weighted)
                scaled = {}
                for k, v in all_losses.items():
                    if k in task_heads._scales:
                        scaled[k] = v * task_heads._scales[k]
                    else:
                        scaled[k] = v * 1.0  # decoder losses unscaled

                model_loss = sum(scaled.values()) + kl_loss
                base_loss = torch.mean(model_loss)

                # EWC penalty (no-op when ewc_manager is not set on the RSSM)
                ewc_loss = torch.tensor(0.0, device=config.device)
                if getattr(rssm, "ewc_manager", None) is not None:
                    ewc_loss = rssm.ewc_manager.penalty(rssm)
                total_loss = base_loss + ewc_loss

            # --- Single backward pass, then step both optimizers ---
            # We must NOT call backward twice (via two Optimizer.__call__),
            # because the first optimizer's step() modifies parameters
            # in-place, which invalidates the graph for a second backward.
            # Instead: one backward, then unscale+clip+step each optimizer.
            # Use the RSSM scaler for the shared backward pass — both
            # optimizers must use the same scaler to keep gradients
            # correctly scaled.
            scaler = rssm._model_opt._scaler

            rssm._model_opt._opt.zero_grad()
            task_heads._heads_opt._opt.zero_grad()

            scaler.scale(total_loss).backward()

            # Unscale both optimizers before clipping
            scaler.unscale_(rssm._model_opt._opt)
            scaler.unscale_(task_heads._heads_opt._opt)

            # PackNet gradient masking (no-op when packnet_manager is not set)
            if getattr(rssm, "packnet_manager", None) is not None:
                rssm.packnet_manager.apply_gradient_mask(rssm)

            # Clip and step RSSM optimizer
            rssm_norm = torch.nn.utils.clip_grad_norm_(
                list(rssm.parameters()), rssm._model_opt._clip
            )
            if rssm._model_opt._wd:
                rssm._model_opt._apply_weight_decay(list(rssm.parameters()))
            scaler.step(rssm._model_opt._opt)

            # PackNet weight re-zeroing after step (prevents momentum revival)
            if getattr(rssm, "packnet_manager", None) is not None:
                rssm.packnet_manager.apply_weight_mask(rssm)

            # Clip and step task heads optimizer
            heads_norm = torch.nn.utils.clip_grad_norm_(
                list(task_heads.parameters()), task_heads._heads_opt._clip
            )
            if task_heads._heads_opt._wd:
                task_heads._heads_opt._apply_weight_decay(
                    list(task_heads.parameters())
                )
            scaler.step(task_heads._heads_opt._opt)

            # Update scaler once after all optimizer steps
            scaler.update()

            metrics = {}
            metrics[f"{rssm._model_opt._name}_loss"] = to_np(total_loss)
            metrics[f"{rssm._model_opt._name}_grad_norm"] = to_np(rssm_norm)
            metrics[f"{task_heads._heads_opt._name}_loss"] = to_np(total_loss)
            metrics[f"{task_heads._heads_opt._name}_grad_norm"] = to_np(heads_norm)

    # --- Metrics ---
    metrics.update(
        {f"{name}_loss": to_np(loss) for name, loss in all_losses.items()}
    )
    metrics["kl_free"] = config.kl_free
    metrics["dyn_scale"] = config.dyn_scale
    metrics["rep_scale"] = config.rep_scale
    metrics["dyn_loss"] = to_np(dyn_loss)
    metrics["rep_loss"] = to_np(rep_loss)
    metrics["kl"] = to_np(torch.mean(kl_value))
    metrics["ewc_loss"] = float(ewc_loss.detach())
    metrics["model_loss_base"] = float(base_loss.detach())

    with torch.cuda.amp.autocast(use_amp):
        metrics["prior_ent"] = to_np(
            torch.mean(rssm.get_dist(prior).entropy())
        )
        metrics["post_ent"] = to_np(
            torch.mean(rssm.get_dist(post).entropy())
        )
        context = dict(
            embed=embed,
            feat=rssm.get_feat(post),
            kl=kl_value,
            postent=rssm.get_dist(post).entropy(),
        )

    post = {k: v.detach() for k, v in post.items()}
    return post, context, metrics
