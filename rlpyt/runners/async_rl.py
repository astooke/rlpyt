
import time
import multiprocessing as mp
import psutil
import torch
from collections import deque
import math

from rlpyt.runners.base import BaseRunner
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.utils.collections import AttrDict
from rlpyt.utils.seed import set_seed, make_seed
from rlpyt.utils.prog_bar import ProgBarCounter
from rlpyt.utils.synchronize import drain_queue, find_port


THROTTLE_WAIT = 0.05


class AsyncRlBase(BaseRunner):
    """
    Runs sampling and optimization asynchronously in separate Python
    processes.  May be useful to achieve higher hardware utilization, e.g.
    CPUs fully busy simulating the environment while GPU fully busy training
    the agent (there's no reason to use this CPU-only).  This setup is
    significantly more complicated than the synchronous (single- or multi-GPU)
    runners, requires use of the asynchronous sampler, and may require special
    methods in the algorithm.

    Further parallelization within the sampler and optimizer are independent.
    The asynchronous sampler can be serial, cpu-parallel, gpu-parallel, or
    multi-gpu-parallel.  The optimizer can be single- or multi-gpu.

    The algorithm must initialize a replay buffer on OS shared memory.  The
    asynchronous sampler will allocate minibatch buffers on OS shared memory,
    and yet another Python process is run to copy the completed minibatches
    over to the algorithm's replay buffer.  While that memory copy is
    underway, the sampler immediately begins gathering the next minibatch.

    Care should be taken to balance the rate at which the algorithm runs against
    the rate of the sampler, as this can affect learning performance.  In the existing
    implementations, the sampler runs at full speed, and the algorithm may be throttled
    not to exceed the specified relative rate.  This is set by the algorithm's ``replay_ratio``,
    which becomes the upper bound on the amount of training samples used in ratio with
    the amount of samples generated.  (In synchronous mode, the replay ratio is enforced 
    more precisely by running a fixed batch size and number of updates per iteration.)

    The master process runs the (first) training GPU and performs all logging.

    Within the optimizer, one agent exists.  If multi-GPU, the same parameter
    values are copied across all GPUs, and PyTorch's DistributedDataParallel
    is used to all-reduce gradients (as in the synchronous multi-GPU runners).
    Within the sampler, one agent exists.  If new agent parameters are
    available from the optimizer between sampler minibatches, then those
    values are copied into the sampler before gathering the next minibatch.

    Note: 
        The ``affinity`` argument should be a structure with ``sampler`` and
        ``optimizer`` attributes holding the respective hardware allocations.
        Optimizer and sampler parallelization is determined from this.
    """

    _eval = False

    def __init__(
            self,
            algo,
            agent,
            sampler,
            n_steps,
            affinity,
            seed=None,
            log_interval_steps=1e5,
            ):
        n_steps = int(n_steps)
        log_interval_steps = int(log_interval_steps)
        save__init__args(locals())

    def train(self):
        """
        Run the optimizer in a loop.  Check whether enough new samples have
        been generated, and throttle down if necessary at each iteration.  Log
        at an interval in the number of sampler iterations, not optimizer
        iterations.
        """
        throttle_itr, delta_throttle_itr = self.startup()
        throttle_time = 0.
        sampler_itr = itr = 0
        if self._eval:
            while self.ctrl.sampler_itr.value < 1:  # Sampler does eval first.
                time.sleep(THROTTLE_WAIT)
            traj_infos = drain_queue(self.traj_infos_queue, n_sentinel=1)
            self.store_diagnostics(0, 0, traj_infos, ())
            self.log_diagnostics(0, 0, 0)
        log_counter = 0
        while True:  # Run until sampler hits n_steps and sets ctrl.quit=True.
            logger.set_iteration(itr)
            with logger.prefix(f"opt_itr #{itr} "):
                while self.ctrl.sampler_itr.value < throttle_itr:
                    if self.ctrl.quit.value:
                        break
                    time.sleep(THROTTLE_WAIT)
                    throttle_time += THROTTLE_WAIT
                if self.ctrl.quit.value:
                    break
                if self.ctrl.opt_throttle is not None:
                    self.ctrl.opt_throttle.wait()
                throttle_itr += delta_throttle_itr
                opt_info = self.algo.optimize_agent(itr,
                    sampler_itr=self.ctrl.sampler_itr.value)
                self.agent.send_shared_memory()  # To sampler.
                sampler_itr = self.ctrl.sampler_itr.value
                traj_infos = (list() if self._eval else
                    drain_queue(self.traj_infos_queue))
                self.store_diagnostics(itr, sampler_itr, traj_infos, opt_info)
                if (sampler_itr // self.log_interval_itrs > log_counter):
                    if self._eval:
                        with self.ctrl.sampler_itr.get_lock():
                            traj_infos = drain_queue(self.traj_infos_queue, n_sentinel=1)
                        self.store_diagnostics(itr, sampler_itr, traj_infos, ())
                    self.log_diagnostics(itr, sampler_itr, throttle_time)
                    log_counter += 1
                    throttle_time = 0.
            itr += 1
        # Final log:
        sampler_itr = self.ctrl.sampler_itr.value
        traj_infos = drain_queue(self.traj_infos_queue)
        if traj_infos or not self._eval:
            self.store_diagnostics(itr, sampler_itr, traj_infos, ())
            self.log_diagnostics(itr, sampler_itr, throttle_time)
        self.shutdown()

    def startup(self):
        """
        Calls ``sampler.async_initialize()`` to get a double buffer for minibatches,
        followed by ``algo.async_initialize()`` to get a replay buffer on shared memory,
        then launches all workers (sampler, optimizer, memory copier).
        """
        if self.seed is None:
            self.seed = make_seed()
        set_seed(self.seed)
        double_buffer, examples = self.sampler.async_initialize(
            agent=self.agent,
            bootstrap_value=getattr(self.algo, "bootstrap_value", False),
            traj_info_kwargs=self.get_traj_info_kwargs(),
            seed=self.seed,
        )
        self.sampler_batch_size = self.sampler.batch_spec.size
        self.world_size = len(self.affinity.optimizer)
        n_itr = self.get_n_itr()  # Number of sampler iterations.
        replay_buffer = self.algo.async_initialize(
            agent=self.agent,
            sampler_n_itr=n_itr,
            batch_spec=self.sampler.batch_spec,
            mid_batch_reset=self.sampler.mid_batch_reset,
            examples=examples,
            world_size=self.world_size,
        )
        self.launch_workers(n_itr, double_buffer, replay_buffer)
        throttle_itr, delta_throttle_itr = self.optim_startup()
        return throttle_itr, delta_throttle_itr

    def optim_startup(self):
        """
        Sets the hardware affinity, moves the agent's model parameters onto
        device and initialize data-parallel agent, if applicable.  Computes
        optimizer throttling settings.
        """
        main_affinity = self.affinity.optimizer[0]
        p = psutil.Process()
        if main_affinity.get("set_affinity", True):
            p.cpu_affinity(main_affinity["cpus"])
        logger.log(f"Optimizer master CPU affinity: {p.cpu_affinity()}.")
        torch.set_num_threads(main_affinity["torch_threads"])
        logger.log(f"Optimizer master Torch threads: {torch.get_num_threads()}.")
        self.agent.to_device(main_affinity.get("cuda_idx", None))
        if self.world_size > 1:
            self.agent.data_parallel()
        self.algo.optim_initialize(rank=0)
        throttle_itr = 1 + getattr(self.algo,
            "min_steps_learn", 0) // self.sampler_batch_size
        delta_throttle_itr = (self.algo.batch_size * self.world_size *
            self.algo.updates_per_optimize /  # (is updates_per_sync)
            (self.sampler_batch_size * self.algo.replay_ratio))
        self.initialize_logging()
        return throttle_itr, delta_throttle_itr

    def launch_workers(self, n_itr, double_buffer, replay_buffer):
        self.traj_infos_queue = mp.Queue()
        self.ctrl = self.build_ctrl(self.world_size)
        self.launch_sampler(n_itr)
        self.launch_memcpy(double_buffer, replay_buffer)
        self.launch_optimizer_workers(n_itr)

    def get_n_itr(self):
        log_interval_itrs = max(self.log_interval_steps //
            self.sampler_batch_size, 1)
        n_itr = math.ceil(self.n_steps / self.log_interval_steps) * log_interval_itrs
        self.log_interval_itrs = log_interval_itrs
        self.n_itr = n_itr
        logger.log(f"Running {n_itr} sampler iterations.")
        return n_itr

    def build_ctrl(self, world_size):
        """
        Builds several parallel communication mechanisms for controlling the
        workflow across processes.
        """
        opt_throttle = (mp.Barrier(world_size) if world_size > 1 else
            None)
        return AttrDict(
            quit=mp.Value('b', lock=True),
            quit_opt=mp.RawValue('b'),
            sample_ready=[mp.Semaphore(0) for _ in range(2)],  # Double buffer.
            sample_copied=[mp.Semaphore(1) for _ in range(2)],
            sampler_itr=mp.Value('l', lock=True),
            opt_throttle=opt_throttle,
            eval_time=mp.Value('d', lock=True),
        )

    def launch_optimizer_workers(self, n_itr):
        """
        If multi-GPU optimization, launches an optimizer worker for each GPU
        and initializes ``torch.distributed.``
        """
        if self.world_size == 1:
            return
        offset = self.affinity.optimizer[0].get("master_cpus", [0])[0]
        port = find_port(offset=offset)
        affinities = self.affinity.optimizer
        runners = [AsyncOptWorker(
            rank=rank,
            world_size=self.world_size,
            algo=self.algo,
            agent=self.agent,
            n_itr=n_itr,
            affinity=affinities[rank],
            seed=self.seed + 100,
            ctrl=self.ctrl,
            port=port,
        ) for rank in range(1, len(affinities))]
        procs = [mp.Process(target=r.optimize, args=()) for r in runners]
        for p in procs:
            p.start()
        torch.distributed.init_process_group(
            backend="nccl",
            rank=0,
            world_size=self.world_size,
            init_method=f"tcp://127.0.0.1:{port}",
        )
        self.optimizer_procs = procs

    def launch_memcpy(self, sample_buffers, replay_buffer):
        """
        Fork a Python process for each of the sampler double buffers.  (It may
        be overkill to use two separate processes here, may be able to simplify
        to one and still get good performance.)
        """
        procs = list()
        for i in range(len(sample_buffers)):  # (2 for double-buffer.)
            ctrl = AttrDict(
                quit=self.ctrl.quit,
                sample_ready=self.ctrl.sample_ready[i],
                sample_copied=self.ctrl.sample_copied[i],
            )
            procs.append(mp.Process(target=memory_copier,
                args=(sample_buffers[i], self.algo.samples_to_buffer,
                replay_buffer, ctrl)))
        for p in procs:
            p.start()
        self.memcpy_procs = procs

    def launch_sampler(self, n_itr):
        target = run_async_sampler
        kwargs = dict(
            sampler=self.sampler,
            affinity=self.affinity.sampler,
            ctrl=self.ctrl,
            traj_infos_queue=self.traj_infos_queue,
            n_itr=n_itr,
        )
        if self._eval:
            target = run_async_sampler_eval
            kwargs["eval_itrs"] = self.log_interval_itrs
        self.sampler_proc = mp.Process(target=target, kwargs=kwargs)
        self.sampler_proc.start()

    def shutdown(self):
        self.pbar.stop()
        logger.log("Master optimizer shutting down, joining sampler process...")
        self.sampler_proc.join()
        logger.log("Joining memory copiers...")
        for p in self.memcpy_procs:
            p.join()
        if self.ctrl.opt_throttle is not None:
            logger.log("Joining optimizer processes...")
            self.ctrl.quit_opt.value = True
            self.ctrl.opt_throttle.wait()
            for p in self.optimizer_procs:
                p.join()
        logger.log("All processes shutdown.  Training complete.")

    def initialize_logging(self):
        self._opt_infos = {k: list() for k in self.algo.opt_info_fields}
        self._start_time = self._last_time = time.time()
        self._last_itr = 0
        self._last_sampler_itr = 0
        self._last_update_counter = 0

    def get_itr_snapshot(self, itr, sampler_itr):
        return dict(
            itr=itr,
            sampler_itr=sampler_itr,
            cum_steps=sampler_itr * self.sampler_batch_size,
            cum_updates=self.algo.update_counter,
            agent_state_dict=self.agent.state_dict(),
            optimizer_state_dict=self.algo.optim_state_dict(),
        )

    def save_itr_snapshot(self, itr, sample_itr):
        logger.log("saving snapshot...")
        params = self.get_itr_snapshot(itr, sample_itr)
        logger.save_itr_params(itr, params)
        logger.log("saved")

    def get_traj_info_kwargs(self):
        return dict(discount=getattr(self.algo, "discount", 1))

    def store_diagnostics(self, itr, sampler_itr, traj_infos, opt_info):
        self._traj_infos.extend(traj_infos)
        for k, v in self._opt_infos.items():
            new_v = getattr(opt_info, k, [])
            v.extend(new_v if isinstance(new_v, list) else [new_v])
        self.pbar.update((sampler_itr + 1) % self.log_interval_itrs)

    def log_diagnostics(self, itr, sampler_itr, throttle_time, prefix='Diagnostics/'):
        self.pbar.stop()
        self.save_itr_snapshot(itr, sampler_itr)
        new_time = time.time()
        time_elapsed = new_time - self._last_time
        new_updates = self.algo.update_counter - self._last_update_counter
        new_samples = self.sampler.batch_size * (sampler_itr - self._last_sampler_itr)
        updates_per_second = (float('nan') if itr == 0 else
            new_updates / time_elapsed)
        samples_per_second = (float('nan') if itr == 0 else
            new_samples / time_elapsed)
        if self._eval:
            new_eval_time = self.ctrl.eval_time.value
            eval_time_elapsed = new_eval_time - self._last_eval_time
            non_eval_time_elapsed = time_elapsed - eval_time_elapsed
            non_eval_samples_per_second = (float('nan') if itr == 0 else
                new_samples / non_eval_time_elapsed)
            self._last_eval_time = new_eval_time
        cum_steps = sampler_itr * self.sampler.batch_size  # No * world_size.
        replay_ratio = (new_updates * self.algo.batch_size * self.world_size /
            max(1, new_samples))
        cum_replay_ratio = (self.algo.update_counter * self.algo.batch_size *
            self.world_size / max(1, cum_steps))

        with logger.tabular_prefix(prefix):
            logger.record_tabular('Iteration', itr)
            logger.record_tabular('SamplerIteration', sampler_itr)
            logger.record_tabular('CumTime (s)', new_time - self._start_time)
            logger.record_tabular('CumSteps', cum_steps)
            logger.record_tabular('CumUpdates', self.algo.update_counter)
            logger.record_tabular('ReplayRatio', replay_ratio)
            logger.record_tabular('CumReplayRatio', cum_replay_ratio)
            logger.record_tabular('StepsPerSecond', samples_per_second)
            if self._eval:
                logger.record_tabular('NonEvalSamplesPerSecond', non_eval_samples_per_second)
            logger.record_tabular('UpdatesPerSecond', updates_per_second)
            logger.record_tabular('OptThrottle', (time_elapsed - throttle_time) /
                time_elapsed)

        self._log_infos()
        self._last_time = new_time
        self._last_itr = itr
        self._last_sampler_itr = sampler_itr
        self._last_update_counter = self.algo.update_counter
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


class AsyncRl(AsyncRlBase):
    """
    Asynchronous RL with online agent performance tracking.
    """

    def __init__(self, *args, log_traj_window=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_traj_window = int(log_traj_window)

    def initialize_logging(self):
        self._traj_infos = deque(maxlen=self.log_traj_window)
        self._cum_completed_trajs = 0
        self._new_completed_trajs = 0
        super().initialize_logging()
        logger.log(f"Optimizing over {self.log_interval_itrs} sampler "
            "iterations.")
        self.pbar = ProgBarCounter(self.log_interval_itrs)

    def store_diagnostics(self, itr, sampler_itr, traj_infos, opt_info):
        self._cum_completed_trajs += len(traj_infos)
        self._new_completed_trajs += len(traj_infos)
        super().store_diagnostics(itr, sampler_itr, traj_infos, opt_info)

    def log_diagnostics(self, itr, sampler_itr, throttle_time, prefix='Diagnostics/'):
        with logger.tabular_prefix(prefix):
            logger.record_tabular('CumCompletedTrajs', self._cum_completed_trajs)
            logger.record_tabular('NewCompletedTrajs', self._new_completed_trajs)
            logger.record_tabular('StepsInTrajWindow',
                sum(info["Length"] for info in self._traj_infos))
        super().log_diagnostics(itr, sampler_itr, throttle_time, prefix=prefix)
        self._new_completed_trajs = 0


class AsyncRlEval(AsyncRlBase):
    """
    Asynchronous RL with offline agent performance evaluation.
    """

    _eval = True

    def initialize_logging(self):
        self._traj_infos = list()
        self._last_eval_time = 0.
        super().initialize_logging()
        self.pbar = ProgBarCounter(self.log_interval_itrs)

    def log_diagnostics(self, itr, sampler_itr, throttle_time, prefix='Diagnostics/'):
        if not self._traj_infos:
            logger.log("WARNING: had no complete trajectories in eval.")
        steps_in_eval = sum([info["Length"] for info in self._traj_infos])
        with logger.tabular_prefix(prefix):
            logger.record_tabular('StepsInEval', steps_in_eval)
            logger.record_tabular('TrajsInEval', len(self._traj_infos))
            logger.record_tabular('CumEvalTime', self.ctrl.eval_time.value)
        super().log_diagnostics(itr, sampler_itr, throttle_time, prefix=prefix)
        self._traj_infos = list()  # Clear after each eval.


###############################################################################
# Worker processes.
###############################################################################


class AsyncOptWorker:

    def __init__(
            self,
            rank,
            world_size,
            algo,
            agent,
            n_itr,
            affinity,
            seed,
            ctrl,
            port
            ):
        save__init__args(locals())

    def optimize(self):
        self.startup()
        itr = 0
        while True:
            self.ctrl.opt_throttle.wait()
            if self.ctrl.quit_opt.value:
                break
            self.algo.optimize_agent(itr, sampler_itr=self.ctrl.sampler_itr.value)  # Leave un-logged.
            itr += 1
        self.shutdown()

    def startup(self):
        torch.distributed.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
            init_method=f"tcp://127.0.0.1:{self.port}",
        )
        p = psutil.Process()
        if self.affinity.get("set_affinity", True):
            p.cpu_affinity(self.affinity["cpus"])
        logger.log(f"Optimizer rank {self.rank} CPU affinity: {p.cpu_affinity()}.")
        torch.set_num_threads(self.affinity["torch_threads"])
        logger.log(f"Optimizer rank {self.rank} Torch threads: {torch.get_num_threads()}.")
        logger.log(f"Optimizer rank {self.rank} CUDA index: "
            f"{self.affinity.get('cuda_idx', None)}.")
        set_seed(self.seed)
        self.agent.to_device(cuda_idx=self.affinity.get("cuda_idx", None))
        self.agent.data_parallel()
        self.algo.optim_initialize(rank=self.rank)

    def shutdown(self):
        logger.log(f"Async optimization worker {self.rank} shutting down.")


def run_async_sampler(sampler, affinity, ctrl, traj_infos_queue, n_itr):
    """
    Target function for the process which will run the sampler, in the case of
    online performance logging.  Toggles the sampler's double-buffer for each
    iteration, waits for the memory copier to finish before writing into that
    buffer, and signals the memory copier when the sampler is done writing a
    minibatch.
    """
    sampler.initialize(affinity)
    db_idx = 0
    for itr in range(n_itr):
        ctrl.sample_copied[db_idx].acquire()
        traj_infos = sampler.obtain_samples(itr, db_idx)
        ctrl.sample_ready[db_idx].release()
        with ctrl.sampler_itr.get_lock():
            for traj_info in traj_infos:
                traj_infos_queue.put(traj_info)
            ctrl.sampler_itr.value = itr
        db_idx ^= 1  # Double buffer.
    logger.log(f"Async sampler reached final itr: {itr + 1}, quitting.")
    ctrl.quit.value = True  # This ends the experiment.
    sampler.shutdown()
    for s in ctrl.sample_ready:
        s.release()  # Let memcpy workers finish and quit.


def run_async_sampler_eval(sampler, affinity, ctrl, traj_infos_queue,
        n_itr, eval_itrs):
    """
    Target function running the sampler with offline performance evaluation.
    """
    sampler.initialize(affinity)
    db_idx = 0
    for itr in range(n_itr + 1):  # +1 to get last eval :)
        ctrl.sample_copied[db_idx].acquire()
        # assert not ctrl.sample_copied[db_idx].acquire(block=False)  # Debug check.
        sampler.obtain_samples(itr, db_idx)
        ctrl.sample_ready[db_idx].release()
        if itr % eval_itrs == 0:
            eval_time = -time.time()
            traj_infos = sampler.evaluate_agent(itr)
            eval_time += time.time()
            ctrl.eval_time.value += eval_time  # Not atomic but only writer.
            with ctrl.sampler_itr.get_lock():
                for traj_info in traj_infos:
                    traj_infos_queue.put(traj_info)
                traj_infos_queue.put(None)  # Master will get until None sentinel.
                ctrl.sampler_itr.value = itr
        else:
            ctrl.sampler_itr.value = itr
        db_idx ^= 1  # Double buffer
    logger.log(f"Async sampler reached final itr: {itr + 1}, quitting.")
    ctrl.quit.value = True  # This ends the experiment.
    sampler.shutdown()
    for s in ctrl.sample_ready:
        s.release()  # Let memcpy workers finish and quit.


def memory_copier(sample_buffer, samples_to_buffer, replay_buffer, ctrl):
    """
    Target function for the process which will copy the sampler's minibatch buffer
    into the algorithm's main replay buffer.

    Args:
        sample_buffer: The (single) minibatch buffer from the sampler, on shared memory.
        samples_to_buffer:  A function/method from the algorithm to process samples from the minibatch buffer into the replay buffer (e.g. select which fields, compute some prioritization).
        replay_buffer: Algorithm's main replay buffer, on shared memory.
        ctrl: Structure for communicating when the minibatch is ready to copy/done copying.
    Warning:
        Although this function may use the algorithm's ``samples_to_buffer()``
        method, here it is running in a separate process, so will not be aware
        of changes in the algorithm from the optimizer process.  Furthermore,
        it may not be able to store state across iterations--in the
        implemented setup, two separate memory copier processes are used, so
        each one only sees every other minibatch.  (Could easily change to
        single copier if desired, and probably without peformance loss.)

    """
    # Needed on some systems to avoid mysterious hang:
    torch.set_num_threads(1)
    # (Without torch.set_num_threads, experienced hang on Ubuntu Server 16.04
    # machines (but not Desktop) when appending samples to make replay buffer
    # full, but only for batch_B > 84 (dqn + r2d1 atari), regardless of replay
    # size or batch_T.  Would seem to progress through all code in
    # replay.append_samples() but simply would not return from it.  Some
    # tipping point for MKL threading?)
    while True:
        ctrl.sample_ready.acquire()
        # assert not ctrl.sample_ready.acquire(block=False)  # Debug check.
        if ctrl.quit.value:
            break
        replay_buffer.append_samples(samples_to_buffer(sample_buffer))
        ctrl.sample_copied.release()
    logger.log("Memory copier shutting down.")


def placeholder(x):

    pass