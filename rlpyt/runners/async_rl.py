
import time
import multiprocessing as mp
import psutil
import torch
from collections import deque

from rlpyt.runners.base import BaseRunner
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.utils.collections import AttrDict
from rlpyt.utils.seed import set_seed
from rlpyt.utils.prog_bar import ProgBarCounter


class AsyncRl(BaseRunner):

    def __init__(
            self,
            algo,
            agent,
            sampler,
            n_steps,
            affinity,
            updates_per_sync=1,
            seed=None,
            log_interval_steps=1e5,
            evaluate_agent=False,
            ):
        n_steps = int(n_steps)
        log_interval_steps = int(log_interval_steps)
        save__init__args(locals())

    def train(self):
        double_buffer, examples = self.sampler.master_runner_initialize(
            agent=self.agent,
            bootstrap_value=getattr(self.algo, "bootstrap_value", False),
            traj_info_kwargs=self.get_traj_info_kwargs(),
        )
        replay_buffer = self.algo.initialize_replay_buffer(
            batch_spec=self.sampler.batch_spec,
            examples=examples,
            mid_batch_reset=self.sampler.mid_batch_reset,
            async=True,
        )
        n_itr = self.get_n_itr()
        self.ctrl = self.build_ctrl(len(self.affinity.optimizer))
        self.launch_sampler(n_itr)
        self.launch_memcpy(double_buffer, replay_buffer)
        self.launch_optimizer()

    def get_n_itr(self):
        log_interval_itrs = max(self.log_interval_steps //
            self.sampler.batch_spec.size, 1)
        n_itr = math.ceil(self.n_steps / self.log_interval_steps) * log_interval_itrs
        self.log_interval_itrs = log_interval_itrs
        self.n_itr = n_itr
        logger.log(f"Running {n_itr} iterations of sampler.")
        return n_itr

    def build_ctrl(self, n_optim_runner):
        opt_throttle = (mp.Barrier(n_optim_runner) if n_optim_runner > 1 else
            None)
        return AttrDict(
            quit=mp.Value('b', lock=True),
            sample_ready=[mp.Semaphore(0) for _ in range(2)],  # Double buffer.
            sample_copied=[mp.Semaphore(1) for _ in range(2)],
            sample_itr=mp.Value('l', lock=True),
            opt_throttle=opt_throttle,
            eval_time=mp.Value('d', lock=True),
            )

    def launch_optimizer(self):
        affinities = self.affinity.optimizer
        MasterOpt = AsyncOptRlEval if self.evaluate_agent else AsyncOptRl
        runners = [MasterOpt(
            algo=self.algo,
            agent=self.agent,
            updates_per_sync=self.updates_per_sync,
            affinity=affinities[0],
            seed=self.seed,
            log_interval_itr=self.log_interval_itr,
            ddp=len(affinities) > 1,
            ctrl=self.ctrl,
        )]
        runners += [AsyncOptWorker(
            algo=self.algo,
            agent=self.agent,
            updates_per_sync=self.updates_per_sync,
            affinity=affinities[i],
            seed=self.seed + 100,
            ctrl=self.ctrl,
            ) for i in range(1, len(affinities))]
        procs = [mp.Process(target=r.optimize, args=()) for r in runners]
        for p in procs:
            p.start()
        self.optimizer_procs = procs

    def launch_memcpy(self, sample_buffers, replay_buffer):
        args_list = list()
        for i in range(len(sample_buffers)):
            ctrl = AttrDict(
                quit=self.ctrl.quit,
                sample_ready=self.ctrl.sample_ready[i],
                sample_copied=self.ctrl.sample_copied[i],
            )
            args_list.append((sample_buffers[i], replay_buffer, ctrl))
        procs = [mp.Process(target=memory_copier, args=a) for a in args_list]
        for p in procs:
            p.start()
        self.memcpy_procs = procs

    def launch_sampler(self, n_itr):
        proc = mp.Process(target=run_async_sampler,
            kwargs=dict(
                sampler=self.sampler,
                affinity=self.affinity.sampler,
                ctrl=self.ctrl,
                seed=self.seed,
                traj_infos_queue=self.traj_infos_queue,
                eval_itrs=self.log_interval_itrs if self.evaluate_agent else 0,
                n_itr=n_itr,
                ),
        )
        proc.start()
        self.sample_proc = proc


###############################################################################
# Optimizing Runners
###############################################################################

THROTTLE_WAIT = 0.05


class AsyncOptMaster(object):

    def __init__(
            self,
            algo,
            agent,
            itr_batch_size,
            updates_per_sync,
            affinity,
            seed,
            log_interval_itr,
            n_runner,
            ctrl,
            ):
        save__init__args(locals())

    def optimize(self):
        throttle_itr, delta_throttle_itr = self.startup()
        throttle_time = 0.
        itr = 0
        if self._log_itr0:
            while self.ctrl.sample_itr.value < 1:
                time.sleep(THROTTLE_WAIT)
            while self.traj_infos_queue.qsize():
                traj_infos = self.traj_infos.queue.get()
            self.store_diagnostics(0, 0, traj_infos, ())
            self.log_diagnostics(0, 0, 0)
        log_counter = 0
        while True:
            if self.ctrl.quit.value:
                break
            with logger.prefix(f"opt_itr #{itr} "):
                while self.ctrl.sample_itr.value < throttle_itr:
                    time.sleep(THROTTLE_WAIT)
                    throttle_time += THROTTLE_WAIT
                if self.ctrl.opt_throttle is not None:
                    self.ctrl.opt_throttle.wait()
                throttle_itr += delta_throttle_itr
                opt_info = self.algo.optimize_agent(itr)
                self.agent.send_shared_memory()
                traj_infos = list()
                sample_itr = self.ctrl.sample_itr.value  # Check before queue.
                while self.traj_infos_queue.qsize():
                    traj_infos = self.traj_infos.queue.get()
                self.store_diagnostics(itr, sample_itr, traj_infos, opt_info)
                if (sample_itr // self.log_interval_itrs > log_counter):
                    self.log_diagnostics(itr, sample_itr, throttle_time)
                    log_counter += 1
                    throttle_time = 0.
            itr += 1
        self.shutdown()

    def startup(self):
        p = psutil.Process()
        p.cpu_affinity(self.affinity["cpus"])
        logger.log("Optimizer master CPU affinity: {p.cpu_affinity()}.")
        torch.set_num_threads(self.affinity["torch_threads"])
        logger.log("Optimizer master Torch threads: {torch.get_num_threads()}.")
        set_seed(self.seed)
        self.agent.initialize_cuda(
            cuda_idx=self.affinity.get("cuda_idx", None),
            dpp=self.n_runner > 1,
        )
        self.algo.initialize_async(agent=self.agent,
            updates_per_sync=self.updates_per_sync)
        throttle_itr = 1 + self.algo.min_steps_learn // self.itr_batch_size
        delta_throttle_itr = (self.algo.batch_size * self.n_runner *
            self.algo.updates_per_optimize /  # (is updates_per_sync)
            (self.itr_batch_size * self.training_ratio))
        self.initilaize_logging()
        return throttle_itr, delta_throttle_itr

    def initialize_logging(self):
        self._opt_infos = {k: list() for k in self.algo.opt_info_fields}
        self._start_time = self._last_time = time.time()
        self._last_itr = 0

    def shutdown(self):
        logger.log("Training complete.")
        self.pbar.stop()

    def get_itr_snapshot(self, itr, sample_itr):
        return dict(
            itr=itr,
            sample_itr=sample_itr,
            cum_steps=sample_itr * self.itr_batch_size,
            agent_state_dict=self.agent.state_dict(),
            optimizer_state_dict=self.algo.optim_state_dict(),
        )

    def save_itr_snapshot(self, itr):
        logger.log("saving snapshot...")
        params = self.get_itr_snapshot(itr)
        logger.save_itr_params(itr, params)
        logger.log("saved")

    def store_diagnostics(self, itr, sample_itr, traj_infos, opt_info):
        self._traj_infos.extend(traj_infos)
        for k, v in self._opt_infos.items():
            new_v = getattr(opt_info, k, [])
            v.extend(new_v if isinstance(new_v, list) else [new_v])
        self.pbar.update((sample_itr + 1) % self.log_interval_itrs)

    def log_diagnostics(self, itr, sample_itr, throttle_time):
        self.pbar.stop()
        self.save_itr_snapshot(itr, sample_itr)
        new_time = time.time()
        time_elapsed = new_time - self._last_time
        samples_per_second = (float('nan') if itr == 0 else
            self.log_interval_itrs * self.itr_batch_size / time_elapsed)
        updates_per_second = (float('nan') if itr == 0 else
            self.algo.updates_per_optimize * (itr - self._last_itr) / time_elapsed)
        logger.record_tabular('Iteration', itr)
        logger.record_tabular('SamplerIteration', sample_itr)
        logger.record_tabular('CumTime (s)', new_time - self._start_time)
        logger.record_tabular('CumSteps', sample_itr * self.itr_batch_size)
        logger.record_tabular('CumUpdates', itr * self.algo.updates_per_optimize)
        logger.record_tabular('SamplesPerSecond', samples_per_second)
        logger.record_tabular('UpdatesPerSecond', updates_per_second)
        logger.record_tabular('OptThrottle', (time_elapsed - throttle_time) /
            time_elapsed)

        self._log_infos()
        self._last_time = new_time
        self._last_itr = itr
        logger.dump_tabular(with_prefix=False)
        logger.log(f"Optimizing over {self.log_interval_itrs} sampler "
            "iterations.")
        self.pbar = ProgBarCounter(self.log_interval_itrs)

    def _log_infos(self, traj_infos=None):
        if traj_infos is None:
            traj_infos = self._traj_infos
        if traj_infos:
            for k in traj_infos[0]:
                if not k.startswith("_"):
                    logger.record_tabular_misc_stat(k,
                        [info[k] for info in traj_infos])

        if self._opt_infos:
            for k, v in self._opt_infos.items():
                logger.record_tabular_misc_stat(k, v)
        self._opt_infos = {k: list() for k in self._opt_infos}  # (reset)


class AsyncOptRl(AsyncOptMaster):

    def initialize_logging(self):
        self._traj_infos = deque(maxlen=self.log_traj_window)
        self._cum_completed_trajs = 0
        self._new_completed_trajs = 0
        self._log_itr0 = False
        super().initialize_logging()
        logger.log(f"Optimizing over {self.log_interval_itrs} sampler "
            "iterations.")
        self.pbar = ProgBarCounter(self.log_interval_itrs)

    def store_diagnostics(self, itr, sample_itr, traj_infos, opt_info):
        self._cum_completed_trajs += len(traj_infos)
        self._new_completed_trajs += len(traj_infos)
        super().store_diagnostics(itr, sample_itr, traj_infos, opt_info)

    def log_diagnostics(self, itr, sample_itr, throttle_time):
        logger.record_tabular('CumCompletedTrajs', self._cum_completed_trajs)
        logger.record_tabular('NewCompletedTrajs', self._new_completed_trajs)
        logger.record_tabular('StepsInTrajWindow',
            sum(info["Length"] for info in self._traj_infos))
        super().log_diagnostics(itr, sample_itr, throttle_time)
        self._new_completed_trajs = 0


class AsyncOptRlEval(AsyncOptMaster):

    def initialize_logging(self):
        self._traj_infos = list()
        self._log_itr0 = True
        super().initialize_logging()

    def log_diagnostics(self, itr, sample_itr, throttle_time):
        if not self._traj_infos:
            logger.log("WARNING: had no complete trajectories in eval.")
        steps_in_eval = sum([info["Length"] for info in self._traj_infos])
        logger.record_tabular('StepsInEval', steps_in_eval)
        logger.record_tabular('TrajsInEval', len(self._traj_infos))
        logger.record_tabular('CumEvalTime', self.ctrl.eval_time.value)
        super().log_diagnostics(itr, sample_itr, throttle_time)
        self._traj_infos = list()  # Clear after each eval.


class AsyncOptWorker(object):

    def __init__(
            self,
            algo,
            agent,
            n_itr,
            affinity,
            seed,
            ctrl,
            rank,
            ):
        save__init__args(locals())

    def optimize(self):
        self.startup()
        for itr in range(self.n_itr):
            self.ctrl.opt_throttle.wait()
            _opt_info = self.algo.optimize_agent(None, itr)  # Leave un-logged.
        self.shutdown()

    def startup(self):
        p = psutil.Process()
        p.cpu_affinity(self.affinity["cpus"])
        logger.log(f"Optimizer rank {rank} CPU affinity: {p.cpu_affinity()}.")
        torch.set_num_threads(self.affinity["torch_threads"])
        logger.log(f"Optimizer rank {rank} Torch threads: {torch.get_num_threads()}.")
        logger.log(f"Optimizer rank {rank} CUDA index: "
            f"{self.affinity.get('cuda_idx', None)}.")
        set_seed(self.seed)
        self.agent.initialize_cuda(
            cuda_idx=self.affinity.get("cuda_idx", None),
            dpp=True,
        )
        self.algo.initialize_async(agent=self.agent,
            updates_per_sync=self.updates_per_sync)

    def shutdown(self):
        pass


###############################################################################
# Sampling Runner
###############################################################################


def run_async_sampler(sampler, affinity, ctrl, seed, traj_infos_queue,
        n_itr, eval_itrs):
    sampler.sample_runner_initialize(affinity, seed)
    j = 0
    for itr in range(n_itr):
        ctrl.sample_copied[j].acquire()
        traj_infos = sampler.obtain_samples(itr)
        ctrl.sample_ready[j].release()
        if eval_itrs > 0:  # Only send traj_infos from evaluation.
            traj_infos = []
            if itr % eval_itrs == 0:
                eval_time = -time.time()
                traj_infos = sampler.evaluate_agent(itr)
                eval_time += time.time()
                ctrl.eval.time.value += eval_time  # Not atomic but only writer.
        for traj_info in traj_infos:
            traj_infos_queue.put(traj_info)  # Into queue before increment itr.
        ctrl.sample_itr.value = itr
        j ^= 1  # Double buffer
    ctrl.quit.value = True  # This ends the experiment.
    sampler.shutdown()
    for s in ctrl.sample_ready:
        s.release()  # Let memcpy workers finish and quit.


def memory_copier(sample_buffer, replay_buffer, ctrl):
    while True:
        ctrl.sample_ready.acquire()
        if ctrl.quit.value:
            break
        replay_buffer.append_samples(sample_buffer)
        ctrl.sample_copied.release()
