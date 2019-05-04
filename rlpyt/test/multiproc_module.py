
import torch
import torch.multiprocessing as mp


W = 2


def main():

    lin = torch.nn.Linear(3, 5)
    lin_share = torch.nn.Linear(4, 6)
    lin_gpu = torch.nn.Linear(4, 6).to(torch.device("cuda"))
    # lin = torch.nn.BatchNorm1d(10)
    # lin_share = torch.nn.BatchNorm1d(5)
    lin_share.share_memory()
    barrier = mp.Barrier(W + 1)
    procs = [mp.Process(target=worker, args=(lin, lin_share, barrier))
        for _ in range(W)]
    for p in procs:
        p.start()

    bar = 0
    printer(bar, "master lin", lin.state_dict())
    printer(bar, "master lin_share", lin_share.state_dict())
    printer(bar, "master lin_gpu", lin_gpu.state_dict())
    barrier.wait()
    bar += 1
    barrier.wait()
    bar += 1
    list(lin.parameters())[0][0] = 0.
    list(lin_share.parameters())[0][1] = 0.
    # lin.state_dict()["running_mean"][0] = 10.
    # lin_share.state_dict()["running_mean"][1] = 11.
    printer(bar, "master lin", lin.state_dict())
    printer(bar, "master lin_share", lin_share.state_dict())
    barrier.wait()
    bar += 1
    barrier.wait()
    lin_share.load_state_dict(lin_gpu.state_dict())
    printer(bar, "master lin_share", lin_share.state_dict())
    printer(bar, "master lin_gpu", lin_gpu.state_dict())
    barrier.wait()

    for p in procs:
        p.join()


def worker(lin, lin_share, barrier):
    bar = 0
    barrier.wait()
    bar += 1
    printer(bar, "worker lin", lin.state_dict())
    printer(bar, "worker lin_share", lin_share.state_dict())
    barrier.wait()
    bar += 1
    barrier.wait()
    bar += 1
    printer(bar, "worker lin", lin.state_dict())
    printer(bar, "worker lin_share", lin_share.state_dict())
    barrier.wait()
    bar += 1
    barrier.wait()
    bar += 1
    printer(bar, "worker lin_share", lin_share.state_dict())

def printer(bar, name, state_dict):
    print(f"\nbar: {bar} - {name}: ")
    for k, v in state_dict.items():
        print(k, v)


if __name__ == "__main__":
    main()


