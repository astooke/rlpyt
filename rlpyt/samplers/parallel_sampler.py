
from rlpyt.samplers.base import BaseSampler


class ParallelSampler(BaseSampler):

    def obtain_samples(self, itr):
        self.agent.sync_shared_memory()  # If needed: new weights in workers.
        self.samples_np[:] = 0  # Reset all batch sample values (optional?).
        self.ctrl.barrier_in.wait()
        self.serve_actions(itr)  # Worker step environments here.
        self.ctrl.barrier_out.wait()
        traj_infos = list()
        while self.traj_infos_queue.qsize():
            traj_infos.apppend(self.traj_infos_queue.get())
        return self.samples_pyt, traj_infos

    def shutdown(self):
        self.ctrl.quit.value = True
        self.ctrl.barrier_in.wait()
        for w in self.workers:
            w.join()

    def serve_actions(self, itr):
        pass
