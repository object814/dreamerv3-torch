"""Wall-clock timing profiler for sequential DreamerV3 training.

This script reuses dreamer_sequential.py but adds detailed timing
instrumentation for:
- environment initialization
- environment reset/step inside simulation
- agent call and policy forward
- dataset fetch and model updates
- logger calls
- episode/cache bookkeeping
- checkpoint load/save

It writes a JSON report with totals, percentages, counters, and derived rates.
"""

import argparse
import json
import pathlib
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np

import dreamer_sequential as seq


class TimingProfiler:
    def __init__(self):
        self.seconds = defaultdict(float)
        self.counts = defaultdict(int)
        self.counters = defaultdict(int)

    @contextmanager
    def section(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.seconds[name] += elapsed
            self.counts[name] += 1

    def inc(self, key, value=1):
        self.counters[key] += int(value)

    def build_report(self, total_wall_time):
        items = []
        for name, sec in sorted(self.seconds.items(), key=lambda x: x[1], reverse=True):
            pct = 100.0 * sec / max(total_wall_time, 1e-12)
            items.append(
                {
                    "name": name,
                    "seconds": sec,
                    "percent_total_wall": pct,
                    "calls": self.counts.get(name, 0),
                    "avg_ms_per_call": 1000.0 * sec / max(self.counts.get(name, 1), 1),
                }
            )

        env_steps = self.counters.get("env_steps", 0)
        agent_calls = self.counters.get("agent_calls", 0)
        episodes_done = self.counters.get("episodes_done", 0)
        total_sim_time = sum(
            sec for name, sec in self.seconds.items() if name.startswith("simulate.")
        )

        derived = {
            "env_steps": env_steps,
            "agent_calls": agent_calls,
            "episodes_done": episodes_done,
            "env_steps_per_wall_sec": env_steps / max(total_wall_time, 1e-12),
            "env_steps_per_sim_sec": env_steps / max(total_sim_time, 1e-12),
            "agent_calls_per_wall_sec": agent_calls / max(total_wall_time, 1e-12),
        }

        return {
            "total_wall_seconds": total_wall_time,
            "sections": items,
            "counters": dict(self.counters),
            "derived": derived,
        }


class TimedLogger:
    def __init__(self, base_logger, profiler):
        self._base = base_logger
        self._prof = profiler

    def scalar(self, name, value):
        with self._prof.section("logger.scalar"):
            return self._base.scalar(name, value)

    def image(self, name, value):
        with self._prof.section("logger.image"):
            return self._base.image(name, value)

    def video(self, name, value):
        with self._prof.section("logger.video"):
            return self._base.video(name, value)

    def write(self, *args, **kwargs):
        with self._prof.section("logger.write"):
            return self._base.write(*args, **kwargs)

    def finish(self):
        if hasattr(self._base, "finish"):
            with self._prof.section("logger.finish"):
                return self._base.finish()
        return None

    @property
    def step(self):
        return self._base.step

    @step.setter
    def step(self, value):
        self._base.step = value

    def __getattr__(self, name):
        return getattr(self._base, name)


def install_instrumentation(profiler):
    tools_mod = seq.tools

    # Time logger construction and every logger operation.
    orig_logger_cls = tools_mod.Logger
    orig_wandb_logger_cls = tools_mod.WandBLogger

    def timed_logger_ctor(*args, **kwargs):
        with profiler.section("logger.init.tensorboard"):
            base = orig_logger_cls(*args, **kwargs)
        return TimedLogger(base, profiler)

    def timed_wandb_logger_ctor(*args, **kwargs):
        with profiler.section("logger.init.wandb"):
            base = orig_wandb_logger_cls(*args, **kwargs)
        return TimedLogger(base, profiler)

    tools_mod.Logger = timed_logger_ctor
    tools_mod.WandBLogger = timed_wandb_logger_ctor

    # Time checkpoint I/O.
    orig_torch_save = seq.torch.save
    orig_torch_load = seq.torch.load

    def timed_torch_save(*args, **kwargs):
        with profiler.section("checkpoint.save"):
            return orig_torch_save(*args, **kwargs)

    def timed_torch_load(*args, **kwargs):
        with profiler.section("checkpoint.load"):
            return orig_torch_load(*args, **kwargs)

    seq.torch.save = timed_torch_save
    seq.torch.load = timed_torch_load

    # Time dataset loading and creation.
    orig_load_episodes = tools_mod.load_episodes
    orig_make_dataset = seq.make_dataset

    def timed_load_episodes(*args, **kwargs):
        with profiler.section("dataset.load_episodes"):
            return orig_load_episodes(*args, **kwargs)

    def timed_make_dataset(*args, **kwargs):
        with profiler.section("dataset.make_dataset"):
            return orig_make_dataset(*args, **kwargs)

    tools_mod.load_episodes = timed_load_episodes
    seq.make_dataset = timed_make_dataset

    # Time environment creation.
    orig_make_env = seq.make_env

    def timed_make_env(task_name, config, mode, env_id):
        with profiler.section(f"env_init.{mode}"):
            return orig_make_env(task_name, config, mode, env_id)

    seq.make_env = timed_make_env

    # Time Dreamer call internals: dataset fetch, model update, policy forward.
    def profiled_dreamer_call(self, obs, reset, state=None, training=True):
        with profiler.section(f"dreamer.call.{ 'train' if training else 'eval' }"):
            step = self._step
            if training:
                with profiler.section("dreamer.train.schedule"):
                    steps = (
                        self._config.pretrain
                        if self._should_pretrain()
                        else self._should_train(step)
                    )
                for _ in range(steps):
                    with profiler.section("dreamer.train.dataset_next"):
                        batch = next(self._dataset)
                    with profiler.section("dreamer.train.update"):
                        self._train(batch)
                    self._update_count += 1
                    self._metrics["update_count"] = self._update_count
                if self._should_log(step):
                    with profiler.section("dreamer.train.log.aggregate"):
                        for name, values in self._metrics.items():
                            self._logger.scalar(name, float(np.mean(values)))
                            self._metrics[name] = []
                    if self._config.video_pred_log:
                        with profiler.section("dreamer.train.log.video_pred"):
                            openl = self._wm.video_pred(next(self._dataset))
                            self._logger.video("train_openl", seq.to_np(openl))
                    with profiler.section("dreamer.train.log.write"):
                        self._logger.write(fps=True)

            with profiler.section("dreamer.policy"):
                policy_output, state = self._policy(obs, state, training)

            if training:
                with profiler.section("dreamer.train.step_bookkeeping"):
                    self._step += len(reset)
                    self._logger.step = self._config.action_repeat * self._step
            return policy_output, state

    seq.Dreamer.__call__ = profiled_dreamer_call

    # Time simulation internals to split env and non-env costs.
    add_to_cache = tools_mod.add_to_cache
    save_episodes = tools_mod.save_episodes
    erase_over_episodes = tools_mod.erase_over_episodes
    erase_over_episode_files = tools_mod.erase_over_episode_files
    convert = tools_mod.convert

    def _simulate_phase(is_eval, agent):
        if is_eval:
            return "eval"
        if isinstance(agent, seq.Dreamer):
            return "train"
        return "prefill"

    def profiled_simulate(
        agent,
        envs,
        cache,
        directory,
        logger,
        is_eval=False,
        limit=None,
        steps=0,
        episodes=0,
        state=None,
    ):
        phase = _simulate_phase(is_eval, agent)
        with profiler.section(f"simulate.{phase}.total"):
            if state is None:
                step, episode = 0, 0
                done = np.ones(len(envs), bool)
                length = np.zeros(len(envs), np.int32)
                obs = [None] * len(envs)
                agent_state = None
                reward = [0] * len(envs)
            else:
                step, episode, done, length, obs, agent_state, reward = state

            while (steps and step < steps) or (episodes and episode < episodes):
                if done.any():
                    indices = [index for index, d in enumerate(done) if d]
                    with profiler.section(f"simulate.{phase}.env_reset"):
                        results = [envs[i].reset() for i in indices]
                        results = [r() for r in results]
                    with profiler.section(f"simulate.{phase}.cache_reset_add"):
                        for index, result in zip(indices, results):
                            t = result.copy()
                            t = {k: convert(v) for k, v in t.items()}
                            t["reward"] = 0.0
                            t["discount"] = 1.0
                            add_to_cache(cache, envs[index].id, t)
                            obs[index] = result

                with profiler.section(f"simulate.{phase}.agent_call"):
                    obs = {
                        k: np.stack([o[k] for o in obs])
                        for k in obs[0]
                        if "log_" not in k
                    }
                    action, agent_state = agent(obs, done, agent_state)
                profiler.inc("agent_calls", 1)

                with profiler.section(f"simulate.{phase}.action_pack"):
                    if isinstance(action, dict):
                        action = [
                            {k: np.array(action[k][i].detach().cpu()) for k in action}
                            for i in range(len(envs))
                        ]
                    else:
                        action = np.array(action)
                    assert len(action) == len(envs)

                with profiler.section(f"simulate.{phase}.env_step"):
                    results = [e.step(a) for e, a in zip(envs, action)]
                    results = [r() for r in results]

                with profiler.section(f"simulate.{phase}.post_step_unpack"):
                    obs, reward, done = zip(*[p[:3] for p in results])
                    obs = list(obs)
                    reward = list(reward)
                    done = np.stack(done)
                    episode += int(done.sum())
                    length += 1
                    step += len(envs)
                    length *= 1 - done
                    profiler.inc("env_steps", len(envs))
                    profiler.inc("episodes_done", int(done.sum()))

                with profiler.section(f"simulate.{phase}.cache_step_add"):
                    for a, result, env in zip(action, results, envs):
                        o, r, d, info = result
                        o = {k: convert(v) for k, v in o.items()}
                        transition = o.copy()
                        if isinstance(a, dict):
                            transition.update(a)
                        else:
                            transition["action"] = a
                        transition["reward"] = r
                        transition["discount"] = info.get(
                            "discount", np.array(1 - float(d))
                        )
                        add_to_cache(cache, env.id, transition)

                if done.any():
                    indices = [index for index, d in enumerate(done) if d]
                    for i in indices:
                        with profiler.section(f"simulate.{phase}.episode_save"):
                            save_episodes(directory, {envs[i].id: cache[envs[i].id]})

                        with profiler.section(f"simulate.{phase}.episode_stats"):
                            length_v = len(cache[envs[i].id]["reward"]) - 1
                            score = float(np.array(cache[envs[i].id]["reward"]).sum())
                            video = cache[envs[i].id]["image"]

                        with profiler.section(f"simulate.{phase}.episode_log_items"):
                            for key in list(cache[envs[i].id].keys()):
                                if "log_" in key:
                                    logger.scalar(
                                        key, float(np.array(cache[envs[i].id][key]).sum())
                                    )
                                    cache[envs[i].id].pop(key)

                        if not is_eval:
                            with profiler.section(f"simulate.{phase}.dataset_maint"):
                                step_in_dataset = erase_over_episodes(cache, limit)
                                if limit:
                                    erase_over_episode_files(directory, cache)
                            with profiler.section(f"simulate.{phase}.logger_train"):
                                logger.scalar("dataset_size", step_in_dataset)
                                logger.scalar("train_return", score)
                                logger.scalar("train_length", length_v)
                                logger.scalar("train_episodes", len(cache))
                                logger.write(step=logger.step)
                        else:
                            if "eval_lengths" not in locals():
                                eval_lengths = []
                                eval_scores = []
                                eval_done = False

                            with profiler.section(f"simulate.{phase}.eval_accumulate"):
                                eval_scores.append(score)
                                eval_lengths.append(length_v)

                                mean_score = sum(eval_scores) / len(eval_scores)
                                mean_length = sum(eval_lengths) / len(eval_lengths)
                                logger.video("eval_policy", np.array(video)[None])

                                if len(eval_scores) >= episodes and not eval_done:
                                    logger.scalar("eval_return", mean_score)
                                    logger.scalar("eval_return_min", min(eval_scores))
                                    logger.scalar("eval_return_max", max(eval_scores))
                                    logger.scalar(
                                        "eval_return_std", float(np.std(eval_scores))
                                    )
                                    logger.scalar("eval_length", mean_length)
                                    logger.scalar("eval_length_min", min(eval_lengths))
                                    logger.scalar("eval_length_max", max(eval_lengths))
                                    logger.scalar(
                                        "eval_length_std", float(np.std(eval_lengths))
                                    )
                                    logger.scalar("eval_episodes", len(eval_scores))
                                    logger.write(step=logger.step)
                                    eval_done = True

            if is_eval:
                with profiler.section("simulate.eval.cache_trim"):
                    while len(cache) > 1:
                        cache.popitem(last=False)

            return (step - steps, episode - episodes, done, length, obs, agent_state, reward)

    tools_mod.simulate = profiled_simulate


def build_parser():
    parser = argparse.ArgumentParser(
        description="Sequential DreamerV3 wall-clock timing profiler",
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        required=True,
        help="List of task names (e.g., metaworld_drawer-open-v3 metaworld_pick-place-v3)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Config profile per task (e.g., debug metaworld_default_light)",
    )
    parser.add_argument(
        "--task-steps",
        nargs="+",
        type=int,
        required=True,
        help="Training steps per task in env steps",
    )

    parser.add_argument("--logdir", type=str, required=True, help="Base log directory")
    parser.add_argument("--logger", type=str, default="tensorboard", help="wandb or tensorboard")
    parser.add_argument("--wandb-entity", type=str, default="haoyu-a2i")
    parser.add_argument("--wandb-project", type=str, default="CCLB_Dreamerv3_Sequential")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    parser.add_argument(
        "--from-checkpoint",
        type=str,
        default=None,
        help="Path to initial checkpoint for the first task",
    )
    parser.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="Skip pretraining on the first task",
    )
    parser.add_argument(
        "--skip-config-check",
        action="store_true",
        help="Skip interactive confirmation before training",
    )

    parser.add_argument(
        "--dataset-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Override dataset_size per task",
    )

    parser.add_argument(
        "--eval-prev-video",
        action="store_true",
        default=True,
        help="Record eval videos for previous tasks (default: True)",
    )
    parser.add_argument(
        "--no-eval-prev-video",
        dest="eval_prev_video",
        action="store_false",
        help="Disable eval videos for previous tasks",
    )

    parser.add_argument(
        "--time-report",
        type=str,
        default=None,
        help="Path to write timing JSON report (default: <logdir>/time_profile.json)",
    )
    parser.add_argument(
        "--time-topk",
        type=int,
        default=30,
        help="Number of top timing sections to print",
    )

    return parser


def print_top_sections(report, topk):
    print("=" * 80)
    print(">>> TIMING PROFILE SUMMARY (wall clock)")
    print("=" * 80)
    print(f"Total wall time: {report['total_wall_seconds']:.2f} s")
    print(
        f"Env steps: {report['derived']['env_steps']}, "
        f"episodes done: {report['derived']['episodes_done']}, "
        f"env steps/sec: {report['derived']['env_steps_per_wall_sec']:.2f}"
    )
    print("-" * 80)
    for i, item in enumerate(report["sections"][:topk], start=1):
        print(
            f"{i:02d}. {item['name']}: "
            f"{item['seconds']:.3f}s "
            f"({item['percent_total_wall']:.2f}%), "
            f"calls={item['calls']}, "
            f"avg={item['avg_ms_per_call']:.3f}ms"
        )
    print("=" * 80)


def main():
    parser = build_parser()
    args, remaining = parser.parse_known_args()

    profiler = TimingProfiler()
    install_instrumentation(profiler)

    start = time.perf_counter()
    exc = None
    try:
        seq.main(args, remaining)
    except Exception as e:
        exc = e
    finally:
        total_wall_time = time.perf_counter() - start
        report = profiler.build_report(total_wall_time)

        logdir = pathlib.Path(args.logdir).expanduser()
        logdir.mkdir(parents=True, exist_ok=True)
        report_path = pathlib.Path(args.time_report).expanduser() if args.time_report else (logdir / "time_profile.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w") as f:
            json.dump(report, f, indent=2)

        print_top_sections(report, args.time_topk)
        print(f">>> Full timing report written to: {report_path}")

    if exc is not None:
        raise exc


if __name__ == "__main__":
    main()
