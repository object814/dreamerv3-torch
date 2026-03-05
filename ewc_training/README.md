# Adapting EWC to DreamerV3 RSSM: A Complete Roadmap

## 1. How EWC Works (High-Level)

Elastic Weight Consolidation (EWC), from the Kirkpatrick et al. 2017 paper "Overcoming Catastrophic Forgetting in Neural Networks," takes a Bayesian perspective on sequential learning. The core insight is:

> After learning task A, the posterior distribution over parameters becomes the prior when learning task B.

### The Key Idea

When you train a network on task A, you find optimal parameters **θ\*_A**. When you then train on task B, naïve SGD will move parameters far away from θ\*_A, destroying task A performance. EWC prevents this by asking: **which parameters were important for task A?**

The answer comes from the **Fisher Information Matrix (FIM)**. The diagonal of the FIM, **F_i**, measures how sensitive the loss is to each parameter θ_i. Parameters with high Fisher values were critical for task A; parameters with low Fisher values can be freely changed.

### The EWC Loss

```
L_total = L_task_B(θ) + (λ/2) · Σ_i F_i · (θ_i − θ*_A,i)²
```

Where:
- `L_task_B(θ)` — standard loss on the new task
- `θ*_A` — optimal parameters after training task A (a snapshot)
- `F_i` — diagonal Fisher Information for parameter i (computed after task A)
- `λ` — hyperparameter controlling how much to protect old knowledge

This is essentially a weighted L2 regularization: instead of penalizing all parameter changes equally, it penalizes changes proportional to how important each parameter was for the previous task.

### Computing the Fisher Diagonal

The Fisher diagonal is estimated empirically from task A's data:

```
F_i = (1/N) · Σ_n (∂ log p(x_n | θ) / ∂θ_i)²
```

In practice, you sample mini-batches from task A's data, compute the loss, backpropagate, and accumulate the squared gradients for each parameter. This is computationally cheap — just the cost of a few extra forward-backward passes.

### Online EWC (Multi-Task Extension)

For 3+ tasks, standard EWC stores a separate Fisher + θ* for every previous task, which scales linearly. **Online EWC** (Schwarz et al. 2018) solves this with a running average:

```
F_running = γ · F_running + F_new_task
θ*_running = updated after each task
```

This keeps memory constant regardless of the number of tasks. **I strongly recommend Online EWC for your setup** since you'll be running multiple MetaWorld tasks sequentially.

---

## 2. How EWC Maps to Your DreamerV3 Architecture

Looking at your code, the Dreamer agent has these component groups:

| Component | Key Prefixes | Role | Apply EWC? |
|-----------|-------------|------|------------|
| **Encoder** | `_wm.encoder.*` | Image → embedding | **Yes** (shared across tasks) |
| **RSSM Dynamics** | `_wm.dynamics.*` | Latent transition model | **Yes** (shared across tasks) |
| **Decoder** | `_wm.heads.decoder.*` | Reconstruction head | **Yes** (shared across tasks) |
| **Reward Head** | `_wm.heads.reward.*` | Per-task reward prediction | **No** (reset per task) |
| **Continue Head** | `_wm.heads.cont.*` | Per-task termination | **No** (reset per task) |
| **Actor** | `_task_behavior.actor.*` | Per-task policy | **No** (reset per task) |
| **Critic** | `_task_behavior.value.*` | Per-task value function | **No** (reset per task) |

**The critical decision**: EWC should regularize only the **RSSM components** (encoder, dynamics, decoder) — these are the shared world model parameters that persist across tasks. The reward head, continue head, actor, and critic are reset per task in your ER script (lines 894–901), so regularizing them makes no sense.

This aligns perfectly with your existing `RSSM_PREFIXES` separation in `dreamer_sequential_er.py` (line 184).

### Where the Fisher Is Computed

The Fisher should be computed from the **world model loss** (not the actor-critic loss), because we want to preserve the world model's knowledge. In your `models.py`, the world model loss (line 147) includes:

```python
model_loss = sum(scaled_losses) + kl_loss  # reconstruction + reward + cont + KL
```

The gradients of this loss with respect to RSSM parameters give you the Fisher diagonal.

---

## 3. Step-by-Step Implementation Roadmap

### Step 1: Create an `EWCManager` Class

This is the core data structure that stores Fisher information and parameter snapshots. Create it as a standalone module (e.g., `ewc.py`):

```python
class EWCManager:
    """Manages Fisher diagonals and parameter snapshots for EWC regularization."""
    
    def __init__(self, lambda_ewc=5000.0, gamma=0.95, online=True):
        self.lambda_ewc = lambda_ewc    # regularization strength
        self.gamma = gamma               # decay for online EWC
        self.online = online
        
        # Running Fisher + params (for online EWC)
        self.fisher_running = {}         # param_name -> Fisher diagonal tensor
        self.params_running = {}         # param_name -> parameter snapshot tensor
        
        # Per-task storage (for standard EWC, if needed)
        self.fisher_per_task = []
        self.params_per_task = []
        
        self.num_tasks_consolidated = 0
```

### Step 2: Implement Fisher Computation

After training on task N, compute the Fisher diagonal from that task's data by sampling mini-batches and accumulating squared gradients:

```python
def compute_fisher(self, world_model, dataset, num_batches=100, device='cuda'):
    """Compute diagonal Fisher Information Matrix for RSSM parameters."""
    fisher = {}
    for name, param in world_model.named_parameters():
        if self._is_rssm_param(name):
            fisher[name] = torch.zeros_like(param.data)
    
    world_model.train()
    for i in range(num_batches):
        data = next(dataset)
        world_model.zero_grad()
        
        # Forward pass through world model (same as WorldModel._train)
        data = world_model.preprocess(data)
        embed = world_model.encoder(data)
        post, prior = world_model.dynamics.observe(embed, data["action"], data["is_first"])
        
        # Compute world model loss
        kl_loss, _, _, _ = world_model.dynamics.kl_loss(post, prior, ...)
        feat = world_model.dynamics.get_feat(post)
        losses = {}
        for name, head in world_model.heads.items():
            pred = head(feat if name in config.grad_heads else feat.detach())
            losses[name] = -pred.log_prob(data[name])
        model_loss = torch.mean(sum(losses.values()) + kl_loss)
        model_loss.backward()
        
        # Accumulate squared gradients
        for name, param in world_model.named_parameters():
            if name in fisher and param.grad is not None:
                fisher[name] += param.grad.data.clone() ** 2
    
    # Average over batches
    for name in fisher:
        fisher[name] /= num_batches
    
    return fisher
```

### Step 3: Implement Consolidation (After Each Task)

```python
def consolidate(self, world_model, fisher_new):
    """Update running Fisher and params after completing a task."""
    if self.online:
        for name, param in world_model.named_parameters():
            if self._is_rssm_param(name):
                if name in self.fisher_running:
                    # Running average: γ * F_old + F_new
                    self.fisher_running[name] = (
                        self.gamma * self.fisher_running[name] + fisher_new[name]
                    )
                else:
                    self.fisher_running[name] = fisher_new[name].clone()
                self.params_running[name] = param.data.clone()
    else:
        # Standard EWC: store per-task
        params_snapshot = {}
        for name, param in world_model.named_parameters():
            if self._is_rssm_param(name):
                params_snapshot[name] = param.data.clone()
        self.fisher_per_task.append(fisher_new)
        self.params_per_task.append(params_snapshot)
    
    self.num_tasks_consolidated += 1
```

### Step 4: Implement the EWC Penalty

```python
def penalty(self, world_model):
    """Compute EWC regularization loss for current parameters."""
    loss = 0.0
    if self.num_tasks_consolidated == 0:
        return torch.tensor(0.0, device=next(world_model.parameters()).device)
    
    if self.online:
        for name, param in world_model.named_parameters():
            if name in self.fisher_running:
                loss += (self.fisher_running[name] * 
                        (param - self.params_running[name]) ** 2).sum()
    else:
        for fisher, params in zip(self.fisher_per_task, self.params_per_task):
            for name, param in world_model.named_parameters():
                if name in fisher:
                    loss += (fisher[name] * 
                            (param - params[name]) ** 2).sum()
    
    return (self.lambda_ewc / 2.0) * loss
```

### Step 5: Modify `WorldModel._train()` to Include EWC Loss

This is the key integration point. In `models.py`, line 147–148:

```python
# BEFORE (vanilla):
model_loss = sum(scaled.values()) + kl_loss
metrics = self._model_opt(torch.mean(model_loss), self.parameters())

# AFTER (with EWC):
model_loss = sum(scaled.values()) + kl_loss
ewc_loss = self.ewc_manager.penalty(self) if hasattr(self, 'ewc_manager') else 0.0
total_loss = torch.mean(model_loss) + ewc_loss
metrics = self._model_opt(total_loss, self.parameters())
metrics["ewc_loss"] = float(ewc_loss) if isinstance(ewc_loss, torch.Tensor) else 0.0
```

### Step 6: Modify the Sequential Training Script

In your `dreamer_sequential_er.py`, the task transition logic (around lines 1035–1067) needs to:

1. **After completing task N**: Compute Fisher, call `consolidate()`
2. **During task N+1**: The modified `WorldModel._train()` automatically adds the penalty

```python
# After task training completes (around line 1035):
if task_idx < num_tasks - 1:  # No need after last task
    print(f">>> EWC: Computing Fisher for task {task_idx+1}...")
    fisher = ewc_manager.compute_fisher(
        agent._wm, train_dataset, 
        num_batches=args.ewc_fisher_batches,
        device=config.device,
    )
    ewc_manager.consolidate(agent._wm, fisher)
    print(f">>> EWC: Consolidated {len(fisher)} parameter groups")
    
    # Save EWC state
    torch.save({
        'fisher_running': ewc_manager.fisher_running,
        'params_running': ewc_manager.params_running,
        'num_tasks': ewc_manager.num_tasks_consolidated,
    }, task_logdir / f"ewc_state_task{task_idx+1}.pt")
```

### Step 7: Add CLI Arguments

```python
parser.add_argument("--ewc-lambda", type=float, default=5000.0,
                    help="EWC regularization strength")
parser.add_argument("--ewc-gamma", type=float, default=0.95,
                    help="Online EWC decay factor")
parser.add_argument("--ewc-fisher-batches", type=int, default=100,
                    help="Number of batches for Fisher estimation")
parser.add_argument("--ewc-online", action="store_true", default=True,
                    help="Use Online EWC (recommended)")
```

---

## 4. File Structure for the Implementation

```
continual_dreamer/
├── dreamer_sequential_ewc.py   # Main training script (fork from dreamer_sequential_er.py)
├── ewc.py                       # EWCManager class (standalone)
├── models.py                    # Modified WorldModel._train() with EWC penalty
├── networks.py                  # Unchanged
└── dreamer.py                   # Unchanged (single-task reference)
```

The key changes touch only **two files**:
- `ewc.py` — new file, ~150 lines
- `models.py` — modify `WorldModel._train()` to add ~5 lines for EWC penalty
- `dreamer_sequential_ewc.py` — fork from ER script, add Fisher computation + consolidation at task boundaries

---

## 5. Relevant GitHub Repos

### Directly Relevant
- **`skezle/continual-dreamer`** — Continual RL with DreamerV2, uses experience replay (their "Continual-Dreamer" configuration). Built on DreamerV2, not V3, but the closest existing work to what you're doing. They don't implement EWC on the world model, but the codebase shows how to handle task transitions with Dreamer.

- **`skezle/owl`** — From the same research group (Sam Kessler, Oxford). This repo explicitly uses **EWC on a shared feature extractor** for continual RL, combined with multi-head architecture. Very relevant architecture-wise: they apply EWC to the shared backbone and use separate heads per task — exactly what you'd do.

### Clean EWC Implementations (for reference)
- **`moskomule/ewc.pytorch`** — Minimal, clean PyTorch EWC implementation with a demo notebook. Good for understanding the core algorithm (~100 lines of actual EWC code).

- **`GMvandeVen/continual-learning`** — Comprehensive PyTorch library implementing EWC, SI, LwF, and many other CL methods. Well-tested and has an ICLR 2025 blog post on Fisher computation nuances. Good for verifying your Fisher computation is correct.

- **`Yuxing-Wang-THU/Elastic-Weights-Consolidation`** — Implements both standard and Online EWC with visualization. Good for comparing standard vs. online variants.

- **`ContinualAI/continual-learning-baselines`** — Uses the Avalanche library; has reproducible EWC benchmarks. More heavyweight but useful if you want a standardized evaluation framework.

### Key Recommendation
I'd recommend using **`skezle/owl`** as your primary architectural reference (EWC on shared features + multi-head), and **`moskomule/ewc.pytorch`** as your EWC algorithm reference for a clean, minimal implementation.

---

## 6. Important Design Decisions and Pitfalls

### λ (lambda) Tuning
This is the single most important hyperparameter. Too high → the world model can't learn new tasks (underfitting). Too low → catastrophic forgetting persists. Start with λ = 5000 and sweep over [100, 1000, 5000, 10000, 50000]. Log the EWC penalty magnitude alongside the world model loss to make sure they're in a comparable range.

### Fisher Computation Considerations
- **Use the full world model loss** (reconstruction + KL + reward + continue), not just reconstruction. This captures the full sensitivity of the RSSM.
- **Number of batches**: 50–200 mini-batches is typically sufficient. More is better but has diminishing returns.
- **AMP (mixed precision)**: Your code uses AMP (line 116 of models.py). Make sure Fisher computation uses full precision — squared gradients in FP16 can overflow or underflow.

### Online EWC vs Standard EWC
For your MetaWorld sequential setup, Online EWC is almost certainly the right choice. Standard EWC stores O(tasks × params) Fisher values, which becomes expensive. Online EWC with γ = 0.95 effectively gives an exponentially-decaying window of importance.

### Interaction with Experience Replay
EWC and ER are complementary and can be combined. EWC protects parameter space; ER protects data distribution. You could create a `dreamer_sequential_ewc_er.py` that uses both. Research suggests the combination often outperforms either alone.

### What NOT to Regularize
- Don't regularize batch norm statistics (if any)
- Don't regularize the reward/continue/actor/critic heads (they're reset per task)
- Don't regularize optimizer state (Adam momentum/variance) — only model parameters

---

## 7. Evaluation Strategy

For each task boundary, track:
1. **Forward transfer**: Performance on task N+1 at start of training (does the RSSM help?)
2. **Backward transfer / Forgetting**: Performance on tasks 1..N after training on task N+1
3. **EWC penalty magnitude**: Should start at 0 and grow, but stay comparable to the main loss
4. **Fisher statistics**: Log mean/std/max of Fisher diagonal to ensure it's well-behaved

Your existing eval loop (lines 970–1003) already evaluates all tasks seen so far, which is perfect for measuring forgetting.