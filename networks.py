import math
import numpy as np
import re

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

import tools


class RSSM(nn.Module):
    """
    The Recurrent State-Space Model (RSSM) is the core world model used in DreamerV3.
    It learns a compact latent representation of the environment and its dynamics,
    enabling prediction, imagination, and planning in latent space.

    The RSSM factorizes the latent state into:
    - a deterministic recurrent state (h_t, called `deter`), and
    - a stochastic latent state (z_t, called `stoch`).

    It consists of three components:

    1) Sequence model (deterministic)
    A GRU-based recurrent model that updates the deterministic state h_t based on
    the previous stochastic state z_{t-1} and the previous action a_{t-1}.
    Its role is to provide stable memory and summarize the interaction history.
    Keeping this component deterministic improves gradient flow, long-horizon
    credit assignment, and stability during imagination rollouts.

    2) Stochastic dynamics model (prior)
    A probabilistic model p(z_t | h_t) that predicts a distribution over the next
    stochastic latent state given the deterministic state. This component models
    uncertainty and stochasticity in the environment that cannot be captured by
    a deterministic recurrence alone. It defines the prior used during imagination
    and latent trajectory rollouts.

    3) Encoder (posterior)
    A probabilistic encoder q(z_t | h_t, x_t) that infers the stochastic latent
    state from the current observation x_t and the deterministic state. During
    training, this posterior corrects the dynamics model using real observations.

    During imagination and policy learning, the encoder is not used; the model
    rolls out future latent trajectories using only the sequence model and the
    stochastic dynamics prior. This separation allows Dreamer to learn stable,
    predictable latent dynamics while retaining expressive stochasticity, following
    the principles of variational state-space models.

    Rollout process:
        state₀ = posterior(real observation)
        for t in 0..H-1:
            a_t ~ actor(state_t)
            h_{t+1} = GRU(h_t, z_t, a_t)
            z_{t+1} ~ p(z | h_{t+1})
            state_{t+1} = (h_{t+1}, z_{t+1})
    """
    def __init__(
        self,
        stoch=30,
        deter=200,
        hidden=200,
        rec_depth=1,
        discrete=False,
        act="SiLU",
        norm=True,
        mean_act="none",
        std_act="softplus",
        min_std=0.1,
        unimix_ratio=0.01,
        initial="learned",
        num_actions=None,
        embed=None,
        device=None,
    ):
        super(RSSM, self).__init__()
        self._stoch = stoch # Dimension of the stochastic latent state z_t (Given by either prior p(z_t | h_t) or posterior q(z_t | h_t, x_t))
        self._deter = deter # Dimension of the deterministic recurrent state h_t (Given by the GRU sequence model, serves as memory and temporal context)
        self._hidden = hidden  # hidden size used throughout RSSM MLPs, including:
                       # - imagination input projection (z_{t-1}, a_{t-1}) → hidden
                       # - imagination output projection h_t → hidden (prior path)
                       # - observation output projection (h_t, embed_t) → hidden (posterior path)
                       # - shared intermediate representation before parameterizing z distributions
        self._min_std = min_std # minimum standard deviation for numerical stability
        self._rec_depth = rec_depth # number of recurrent updates per time step (not correctly implemented in the current version, always 1)
        self._discrete = discrete # whether z is discrete or continuous
        act = getattr(torch.nn, act) # activation function used in RSSM MLPs, default being SiLU
        self._mean_act = mean_act # activation applied to mean of continuous z
        self._std_act = std_act # activation applied to std of continuous z
        self._unimix_ratio = unimix_ratio # for discrete z, the ratio of unimix used to prevent category collapse
        self._initial = initial # how to initialize the initial deterministic state
        self._num_actions = num_actions # # dimensionality of action space
        self._embed = embed # dimensionality of encoder output
        self._device = device
        
        """
        Imagination input model, maps (z_{t-1}, a_{t-1}) -> hidden features. This is the input preprocessing for the sequence model.
        This is the input to the sequence model (GRU), to form the sequence model h_t = f(h_t-1, z_t-1, a_t-1).
        """
        inp_layers = [] # list of layers for processing the input to the GRU. The input is the concatenation of the previous stochastic state z_{t-1} and the previous action a_{t-1}.
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions # if z is discrete, the dimension is dimension of stochastic state (stoch * discrete) + dimension of action
        else:
            inp_dim = self._stoch + num_actions # if z is continuous, the dimension is dimension of stochastic state (stoch) + dimension of action
        inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False)) # linear layer to project the input to the hidden dimension
        if norm:
            inp_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        inp_layers.append(act())
        self._img_in_layers = nn.Sequential(*inp_layers) # the sequential model for processing the input to the GRU
        self._img_in_layers.apply(tools.weight_init)
        """
        Sequence model: a GRU cell that takes the processed input and the previous deterministic state to produce the next deterministic state.
        """
        self._cell = GRUCell(self._hidden, self._deter, norm=norm) # GRU cell initialisation, which takes the processed input and the previous deterministic state to produce the next deterministic state
        self._cell.apply(tools.weight_init)

        """
        Imagination output model, maps h_t -> hidden features used to parameterize prior p(z_t | h_t).
        """
        img_out_layers = []
        inp_dim = self._deter
        img_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            img_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        img_out_layers.append(act())
        self._img_out_layers = nn.Sequential(*img_out_layers)
        self._img_out_layers.apply(tools.weight_init)

        """
        Observation output model, maps (h_t, embed_t) -> hidden features used to parameterize posterior q(z_t | h_t, x_t)
        """
        obs_out_layers = []
        inp_dim = self._deter + self._embed
        obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            obs_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        obs_out_layers.append(act())
        self._obs_out_layers = nn.Sequential(*obs_out_layers)
        self._obs_out_layers.apply(tools.weight_init)

        """
        Stochastic state parameter heads
        Separate heads for:
        - prior  p(z_t | h_t)         -> imgs_stat_layer
        - posterior q(z_t | h_t, x_t) -> obs_stat_layer
        Same shape, separate parameters
        """
        if self._discrete:
            self._imgs_stat_layer = nn.Linear(
                self._hidden, self._stoch * self._discrete
            )
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))
        else:
            self._imgs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))

        """
        Learned initial deterministic state h_0, used at episode start when no previous state exists
        """
        if self._initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )

    def initial(self, batch_size):
        """
        The initial function initializes the latent state of the RSSM at the beginning of an episode or when a reset signal is received.
        """
        """
        Initialises the deterministic part (sequence model) of the latent state to zeros.
        Initialises the stochastic part (dynamics model) of the latent state to either:
        - discrete: logits and samples of a one-hot distribution. 
                    The dimension is (self._stoch, self._discrete) where self._stoch is the number of discrete latent variables and 
                    self._discrete is the number of categories for each variable.
        - continuous: mean and std of a normal distribution. The dimension is (self._stoch,) where self._stoch is the number of continuous latent variables.
        """
        deter = torch.zeros(batch_size, self._deter, device=self._device)
        if self._discrete:
            state = dict(
                logit=torch.zeros(
                    [batch_size, self._stoch, self._discrete], device=self._device
                ),
                stoch=torch.zeros(
                    [batch_size, self._stoch, self._discrete], device=self._device
                ),
                deter=deter,
            )
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch], device=self._device),
                std=torch.zeros([batch_size, self._stoch], device=self._device),
                stoch=torch.zeros([batch_size, self._stoch], device=self._device),
                deter=deter,
            )
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
        """
        The observe function processes a sequence of observations and actions to update the latent state of the RSSM.

        Args:
        - embed: the embedded observations given by the encoder network, with shape (batch, time, embed_dim).
        - action: the sequence of actions taken by the agent, with shape (batch, time, action_dim).
        - is_first: a binary tensor indicating the first time step of each episode in the batch, with shape (batch, time).
        - state: the initial latent state of the RSSM, which can be None or a dictionary containing the deterministic and stochastic components of the state.
        """
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        # (batch, time, ch) -> (time, batch, ch)
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        # prev_state[0] means selecting posterior of return(posterior, prior) from obs_step
        """
        static_scan is a function that iteratively applies the fc (in this case, obs_step) to the input sequences.
        High level speaking, this function will loop over the time dimension of the inputs (encoder embeddings, actions, and is_first flags) 
        and apply the obs_step function at each time step, passing the previous state and the current inputs to it.
        """
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(
                prev_state[0], prev_act, embed, is_first
            ),
            (action, embed, is_first), # inputs, the function will iterate over the first dimension of these inputs and pass the corresponding slices to obs_step
            (state, state), # start, the initial state that will be passed to `fn` on the first iteration
        )

        # (batch, time, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine_with_action(self, action, state):
        """
        This function performs open-loop imagination using the dynamics model p(z_t | h_t) only.
        It dreams future trajectories by rolling the RSSM forward given a sequence of actions and a starting latent state.
        In DreamerV3, this is used for:
        1. actor learning: the actor generates a sequence of future actions, and RSSM imagines the resulting latent trajectory, 
                           which is then used to compute the actor loss based on predicted rewards and values along the imagined trajectory.
        2. value prediction: the value function is trained to predict the expected return along imagined trajectories, 
                             so the RSSM imagines future latent states given the current state and a sequence of actions, 
                             and the value function learns to predict the returns from those imagined states.

        Args:
        - action: a sequence of future actions, typically generated by the actor network, with shape (batch, time, action_dim).
        - state: the starting latent state of the RSSM, which is a dictionary containing the deterministic (h_t) and stochastic (z_t) components of the state.
        """
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        assert isinstance(state, dict), state
        action = swap(action)
        prior = tools.static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        """
        Concatenates the stochastic (z_t) and deterministic (h_t) to form the complete latent state representation used for downstream heads,
        e.g. actor, value, reward model, continuation model.
        z contains uncertainty and environment variation.
        h contains memory and temporal context
        """
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)

    def get_dist(self, state, dtype=None):
        """
        Converts raw latent parameters into a proper probability distribution.
        - For discrete latents:
            Creates a stable one-hot categorical distribution.
            Uses unimix to prevent category collapse.
        - For continuous latents:
            Creates a Normal distribution wrapped for numerical stability.
            Used for reparameterized sampling and KL computation.
        """
        if self._discrete:
            """create a stable, one-hot action distribution from logits, with correct probability math."""
            logit = state["logit"]
            dist = torchd.independent.Independent(
                tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
            )
        else:
            """create a stable, squashed normal distribution from mean and std, with correct probability math."""
            mean, std = state["mean"], state["std"]
            dist = tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)
            )
        return dist

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        """
        This is the core posterior update step.
        It combines: previous latent state (h_t-1, z_t-1), previous action (a_t-1), and current observation embedding.
        Gives: a prior (what the dynamics model predicts p(z_t | h_t)) and a posterior (what the observation corrects it to q(z_t | h_t, x_t))
        In Dreamer, this is used only during training. The posterior never exists during imagination.
        
        Args:
        - prev_state: the previous latent state of the RSSM, which is a dictionary containing the deterministic (h_t-1) and stochastic (z_t-1) components of the state.
        - prev_action: the action taken at the previous time step, with shape (batch, action_dim).
        - embed: the embedded observation at the current time step, with shape (batch, embed_dim).
        - is_first: a binary tensor indicating whether the current time step is the first step of an episode, with shape (batch,).
        """
        # initialize all prev_state
        if prev_state == None or torch.sum(is_first) == len(is_first):
            prev_state = self.initial(len(is_first))
            prev_action = torch.zeros(
                (len(is_first), self._num_actions), device=self._device
            )
        # overwrite the prev_state only where is_first=True
        elif torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action *= 1.0 - is_first
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                prev_state[key] = (
                    val * (1.0 - is_first_r) + init_state[key] * is_first_r
                )

        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior["deter"], embed], -1)
        # (batch_size, prior_deter + embed) -> (batch_size, hidden)
        x = self._obs_out_layers(x)
        # (batch_size, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("obs", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    def img_step(self, prev_state, prev_action, sample=True):
        """
        This is the pure dynamics rollout step.
        Given: previous stochastic state z_t-1, previous deterministic state h_t-1, and previous action a_t-1,
        Updates h_t using the GRU (sequence model)
        Predicts a prior distribution over z_t (dynamics model)
        Samples z_t

        This function defines:

        p(h_t, z_t | h_t-1, z_t-1, a_t-1)

        Every imagined trajectory in Dreamer is built by repeatedly calling this.
        """
        # (batch, stoch, discrete_num)
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
            prev_stoch = prev_stoch.reshape(shape)
        # (batch, stoch * discrete_num) -> (batch, stoch * discrete_num + action)
        x = torch.cat([prev_stoch, prev_action], -1)
        # (batch, stoch * discrete_num + action, embed) -> (batch, hidden)
        x = self._img_in_layers(x)
        for _ in range(self._rec_depth):  # rec depth is not correctly implemented
            deter = prev_state["deter"]
            # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
            x, deter = self._cell(x, [deter])
            deter = deter[0]  # Keras wraps the state in a list.
        # (batch, deter) -> (batch, hidden)
        x = self._img_out_layers(x)
        # (batch, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("ims", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def get_stoch(self, deter):
        """
        This function generates a default stochastic state from a deterministic state alone.

        It is used primarily during initialization:
        When no observation is available

        When starting imagination
        It passes h through the same dynamics head used for priors.

        Conceptually:
        “If I only know my memory h_t, what latent state z_t is most likely?”
        """
        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()

    def _suff_stats_layer(self, name, x):
        """
        This function maps hidden features to distribution parameters.
        Depending on context:
        "ims": imagination step, uses the dynamics model p(z_t | h_t)
        "obs": observation step, uses the encoder q(z_t | h_t, x_t)

        For continuous latents:
        Produces mean and std with carefully chosen nonlinearities.
        Ensures std is positive and bounded away from zero.
        For discrete latents:
        Produces logits for categorical distributions.

        This separation is what allows:
        One shared latent space
        Two different inference paths (prior vs posterior)

        Args:
            - name: a string indicating whether we are in the imagination step ("ims") or the observation step ("obs"), which determines which set of layers to use for computing the sufficient statistics.
            - x: the input features from which to compute the distribution parameters, typically the output of a feedforward layer applied to the deterministic state (and possibly the observation embedding for the posterior).
        """
        if self._discrete:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = torch.split(x, [self._stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        """
        Calculates the KL divergence loss between the posterior and prior distributions of the latent state.

        In dreamerv3, the posterior is the latent state inferred from observation (encoder), 
        and the prior is the latent state predicted from the previous latent state and action (dynamics model).

        Args:
            post: The posterior distribution of the latent state. This is typically obtained from the observation step of the RSSM.
        """
        kld = torchd.kl.kl_divergence
        """function that yields the distribution object for the given state (state["mean"], state["std"] for continuous, state["logit"] for discrete)"""
        dist = lambda x: self.get_dist(x)
        """function that detaches the parameters of the distribution from the computational graph, preventing gradients from flowing back through the prior when calculating the KL divergence loss."""
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        """
        rep_loss encourages the posterior (encoder) to be close to the prior (dynamics model)
        dyn_loss encourages the prior (dynamics model) to be close to the posterior (encoder)

        Why not just one KL?
        Because a single KL would let:
        - the encoder chase reconstruction
        - the dynamics chase the encoder
        leading to both collapsing or oscillating.
        """
        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )
        # this is implemented using maximum at the original repo as the gradients are not backpropagated for the out of limits.
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss


class MultiEncoder(nn.Module):
    def __init__(
        self,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        symlog_inputs,
    ):
        super(MultiEncoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.outdim = 0
        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self._cnn = ConvEncoder(
                input_shape, cnn_depth, act, norm, kernel_size, minres
            )
            self.outdim += self._cnn.outdim
        if self.mlp_shapes:
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            self._mlp = MLP(
                input_size,
                None,
                mlp_layers,
                mlp_units,
                act,
                norm,
                symlog_inputs=symlog_inputs,
                name="Encoder",
            )
            self.outdim += mlp_units

    def forward(self, obs):
        outputs = []
        if self.cnn_shapes:
            inputs = torch.cat([obs[k] for k in self.cnn_shapes], -1)
            outputs.append(self._cnn(inputs))
        if self.mlp_shapes:
            inputs = torch.cat([obs[k] for k in self.mlp_shapes], -1)
            outputs.append(self._mlp(inputs))
        outputs = torch.cat(outputs, -1)
        return outputs


class MultiDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        cnn_sigmoid,
        image_dist,
        vector_dist,
        outscale,
    ):
        super(MultiDecoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)

        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self._cnn = ConvDecoder(
                feat_size,
                shape,
                cnn_depth,
                act,
                norm,
                kernel_size,
                minres,
                outscale=outscale,
                cnn_sigmoid=cnn_sigmoid,
            )
        if self.mlp_shapes:
            self._mlp = MLP(
                feat_size,
                self.mlp_shapes,
                mlp_layers,
                mlp_units,
                act,
                norm,
                vector_dist,
                outscale=outscale,
                name="Decoder",
            )
        self._image_dist = image_dist

    def forward(self, features):
        dists = {}
        if self.cnn_shapes:
            feat = features
            outputs = self._cnn(feat)
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update(
                {
                    key: self._make_image_dist(output)
                    for key, output in zip(self.cnn_shapes.keys(), outputs)
                }
            )
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        return dists

    def _make_image_dist(self, mean):
        if self._image_dist == "normal":
            return tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, 1), 3)
            )
        if self._image_dist == "mse":
            return tools.MSEDist(mean)
        raise NotImplementedError(self._image_dist)


class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        depth=32,
        act="SiLU",
        norm=True,
        kernel_size=4,
        minres=4,
    ):
        super(ConvEncoder, self).__init__()
        act = getattr(torch.nn, act)
        h, w, input_ch = input_shape
        stages = int(np.log2(h) - np.log2(minres))
        in_dim = input_ch
        out_dim = depth
        layers = []
        for i in range(stages):
            layers.append(
                Conv2dSamePad(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=False,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            layers.append(act())
            in_dim = out_dim
            out_dim *= 2
            h, w = h // 2, w // 2

        self.outdim = out_dim // 2 * h * w
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

    def forward(self, obs):
        obs -= 0.5
        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch * time, ...) -> (batch * time, -1)
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        # (batch * time, -1) -> (batch, time, -1)
        return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])


class ConvDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        shape=(3, 64, 64),
        depth=32,
        act=nn.ELU,
        norm=True,
        kernel_size=4,
        minres=4,
        outscale=1.0,
        cnn_sigmoid=False,
    ):
        super(ConvDecoder, self).__init__()
        act = getattr(torch.nn, act)
        self._shape = shape
        self._cnn_sigmoid = cnn_sigmoid
        layer_num = int(np.log2(shape[1]) - np.log2(minres))
        self._minres = minres
        out_ch = minres**2 * depth * 2 ** (layer_num - 1)
        self._embed_size = out_ch

        self._linear_layer = nn.Linear(feat_size, out_ch)
        self._linear_layer.apply(tools.uniform_weight_init(outscale))
        in_dim = out_ch // (minres**2)
        out_dim = in_dim // 2

        layers = []
        h, w = minres, minres
        for i in range(layer_num):
            bias = False
            if i == layer_num - 1:
                out_dim = self._shape[0]
                act = False
                bias = True
                norm = False

            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth
            pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
            pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)
            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            if act:
                layers.append(act())
            in_dim = out_dim
            out_dim //= 2
            h, w = h * 2, w * 2
        [m.apply(tools.weight_init) for m in layers[:-1]]
        layers[-1].apply(tools.uniform_weight_init(outscale))
        self.layers = nn.Sequential(*layers)

    def calc_same_pad(self, k, s, d):
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def forward(self, features, dtype=None):
        x = self._linear_layer(features)
        # (batch, time, -1) -> (batch * time, h, w, ch)
        x = x.reshape(
            [-1, self._minres, self._minres, self._embed_size // self._minres**2]
        )
        # (batch, time, -1) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch, time, -1) -> (batch, time, ch, h, w)
        mean = x.reshape(features.shape[:-1] + self._shape)
        # (batch, time, ch, h, w) -> (batch, time, h, w, ch)
        mean = mean.permute(0, 1, 3, 4, 2)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean)
        else:
            mean += 0.5
        return mean


class MLP(nn.Module):
    def __init__(
        self,
        inp_dim,
        shape,
        layers,
        units,
        act="SiLU",
        norm=True,
        dist="normal",
        std=1.0,
        min_std=0.1,
        max_std=1.0,
        absmax=None,
        temp=0.1,
        unimix_ratio=0.01,
        outscale=1.0,
        symlog_inputs=False,
        device="cuda",
        name="NoName",
    ):
        super(MLP, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)
        act = getattr(torch.nn, act)
        self._dist = dist
        self._std = std if isinstance(std, str) else torch.tensor((std,), device=device)
        self._min_std = min_std
        self._max_std = max_std
        self._absmax = absmax
        self._temp = temp
        self._unimix_ratio = unimix_ratio
        self._symlog_inputs = symlog_inputs
        self._device = device

        self.layers = nn.Sequential()
        for i in range(layers):
            self.layers.add_module(
                f"{name}_linear{i}", nn.Linear(inp_dim, units, bias=False)
            )
            if norm:
                self.layers.add_module(
                    f"{name}_norm{i}", nn.LayerNorm(units, eps=1e-03)
                )
            self.layers.add_module(f"{name}_act{i}", act())
            if i == 0:
                inp_dim = units
        self.layers.apply(tools.weight_init)

        if isinstance(self._shape, dict):
            self.mean_layer = nn.ModuleDict()
            for name, shape in self._shape.items():
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.ModuleDict()
                for name, shape in self._shape.items():
                    self.std_layer[name] = nn.Linear(inp_dim, np.prod(shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))
        elif self._shape is not None:
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.Linear(units, np.prod(self._shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        if self._symlog_inputs:
            x = tools.symlog(x)
        out = self.layers(x)
        # Used for encoder output
        if self._shape is None:
            return out
        if isinstance(self._shape, dict):
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)
                if self._std == "learned":
                    std = self.std_layer[name](out)
                else:
                    std = self._std
                dists.update({name: self.dist(self._dist, mean, std, shape)})
            return dists
        else:
            mean = self.mean_layer(out)
            if self._std == "learned":
                std = self.std_layer(out)
            else:
                std = self._std
            return self.dist(self._dist, mean, std, self._shape)

    def dist(self, dist, mean, std, shape):
        if dist == "tanh_normal":
            mean = torch.tanh(mean)
            std = F.softplus(std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif dist == "normal":
            std = (self._max_std - self._min_std) * torch.sigmoid(
                std + 2.0
            ) + self._min_std
            dist = torchd.normal.Normal(torch.tanh(mean), std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif dist == "normal_std_fixed":
            dist = torchd.normal.Normal(mean, self._std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif dist == "trunc_normal":
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif dist == "onehot":
            dist = tools.OneHotDist(mean, unimix_ratio=self._unimix_ratio)
        elif dist == "onehot_gumble":
            dist = tools.ContDist(
                torchd.gumbel.Gumbel(mean, 1 / self._temp), absmax=self._absmax
            )
        elif dist == "huber":
            dist = tools.ContDist(
                torchd.independent.Independent(
                    tools.UnnormalizedHuber(mean, std, 1.0),
                    len(shape),
                    absmax=self._absmax,
                )
            )
        elif dist == "binary":
            dist = tools.Bernoulli(
                torchd.independent.Independent(
                    torchd.bernoulli.Bernoulli(logits=mean), len(shape)
                )
            )
        elif dist == "symlog_disc":
            dist = tools.DiscDist(logits=mean, device=self._device)
        elif dist == "symlog_mse":
            dist = tools.SymlogDist(mean)
        else:
            raise NotImplementedError(dist)
        return dist


class GRUCell(nn.Module):
    """
    Standard GRU cell model:
    A GRU maintains one state vector h_t (hidden state)
    At each time step, it decides:
        - how much of the old state h_t-1 to keep (update gate)
        - how much of the new candidate state to use (reset gate)
    Reset gate r_t: consider it as a soft switch that controls how much of the previous hidden state h_t-1 should influence the candidate hidden state.
    Update gate z_t: determines how much of the candidate hidden state should be used to update the hidden state, and how much of the previous hidden state should be retained.

    For input x_t and previous hidden state h_t-1, the GRU computes:
    r_t = sigmoid(W_r * [x_t, h_t-1] + b_r)
    z_t = sigmoid(W_z * [x_t, h_t-1] + b_z)
    h~_t = tanh(W_h * [x_t, r_t * h_t-1] + b_h) # candidate hidden state, where the reset gate r_t modulates the influence of the previous hidden state
    h_t = (1 - z_t) * h_t-1 + z_t * h~_t # the new hidden state is a combination of the previous hidden state and the candidate hidden state, weighted by the update gate z_t
    
    GRU strength vs LSTM:
        - Fewer parameters than LSTM
        - Faster
        - Often just as expressive
        - Easier to train

    

    """
    def __init__(self, inp_size, size, norm=True, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._update_bias = update_bias
        self.layers = nn.Sequential()
        self.layers.add_module(
            "GRU_linear", nn.Linear(inp_size + size, 3 * size, bias=False)
        )
        if norm:
            self.layers.add_module("GRU_norm", nn.LayerNorm(3 * size, eps=1e-03))

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self.layers(torch.cat([inputs, state], -1))
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class Conv2dSamePad(torch.nn.Conv2d):
    def calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret


class ImgChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super(ImgChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x
