

# Readable-to-less-readable abbreviations.
N_GPU = "gpu"
CONTEXTS_PER_GPU = "cxg"  # CUDA contexts.
CONTEXTS_PER_RUN = "cxr"
N_CPU_CORES = "cpu"
HYPERTHREAD_OFFSET = "hto"  # Can specify if different from n_cpu_cores.
N_SOCKET = "skt"
RUN_SLOT = "slt"
CPU_PER_WORKER = "cpw"
CPU_PER_RUN = "cpr"  # For cpu-only.
CPU_RESERVED = "res"  # Reserve CPU cores per master, not allowed by workers.

ABBREVS = [N_GPU, CONTEXTS_PER_GPU, CONTEXTS_PER_RUN, N_CPU_CORES,
    HYPERTHREAD_OFFSET, N_SOCKET, CPU_PER_RUN, CPU_PER_WORKER, CPU_RESERVED]


# API

def quick_affinity_code(n_parallel=None, use_gpu=True):
    assert use_gpu or n_parallel
    import psutil
    n_cpu_cores = psutil.cpu_count(logical=False)
    has_hyperthreads = psutil.cpu_count() == n_cpu_cores * 2
    hyperthread_offset = n_cpu_cores if has_hyperthreads else 0
    if use_gpu:
        import torch
        n_gpu = torch.cuda.device_count()
    else:
        n_gpu = 0
    if n_gpu > 0:
        if n_parallel is not None:
            n_gpu = min(n_parallel, n_gpu)
        n_cpu_cores = (n_cpu_cores // n_gpu) * n_gpu  # Same for all.
        import subprocess
        n_socket = max(1, int(subprocess.check_output(
            'cat /proc/cpuinfo | grep "physical id" | sort -u | wc -l', shell=True)))
        return encode_affinity(n_cpu_cores=n_cpu_cores, n_gpu=n_gpu,
            hyperthread_offset=hyperthread_offset, n_socket=n_socket)
    else:
        n_parallel = min(n_parallel, n_cpu_cores)
        n_cpu_cores = (n_cpu_cores // n_parallel) * n_parallel  # Same for all.
        cpu_per_run = n_cpu_cores // n_parallel
        return encode_affinity(n_cpu_cores=n_cpu_cores, n_gpu=0,
            cpu_per_run=cpu_per_run, hyperthread_offset=hyperthread_offset)


def encode_affinity(n_cpu_cores=1, n_gpu=0, cpu_reserved=0,
        contexts_per_gpu=1, contexts_per_run=1, cpu_per_run=1, cpu_per_worker=1,
        hyperthread_offset=None, n_socket=1, run_slot=None):
    """Use in run script to specify computer configuration."""
    affinity_code = f"{n_cpu_cores}{N_CPU_CORES}_{n_gpu}{N_GPU}"
    if contexts_per_gpu > 1:
        affinity_code += f"_{contexts_per_gpu}{CONTEXTS_PER_GPU}"
    if contexts_per_run > 1:
        raise NotImplementedError  # (parallel training)
    if n_gpu == 0:
        affinity_code += f"_{cpu_per_run}{CPU_PER_RUN}"
    if cpu_per_worker > 1:
        affinity_code += f"_{cpu_per_worker}{CPU_PER_WORKER}"
    if hyperthread_offset is not None:
        affinity_code += f"_{hyperthread_offset}{HYPERTHREAD_OFFSET}"
    if n_socket > 1:
        affinity_code += f"_{n_socket}{N_SOCKET}"
    if cpu_reserved > 0:
        affinity_code += f"_{cpu_reserved}{CPU_RESERVED}"
    if run_slot is not None:
        assert run_slot <= (n_gpu * contexts_per_gpu) // contexts_per_run
        affinity_code = f"{run_slot}{RUN_SLOT}_" + affinity_code
    return affinity_code


def prepend_run_slot(run_slot, affinity_code):
    """Use in launch manager when assigning run slot."""
    return f"{run_slot}{RUN_SLOT}_" + affinity_code


def get_affinity(run_slot_affinity_code):
    """Use in individual experiment script; pass output to Runner."""
    run_slot, aff_code = remove_run_slot(run_slot_affinity_code)
    aff_params = decode_affinity(aff_code)
    if aff_params.get("gpu", 0) > 0:
        return build_gpu_affinity(run_slot, **aff_params)
    return build_cpu_affinity(run_slot, **aff_params)


# Helpers

def get_n_run_slots(affinity_code):
    aff = decode_affinity(affinity_code)
    if aff.get("gpu", 0) > 0:
        n_run_slots = (aff["gpu"] * aff.get("cxg", 1)) // aff.get("cxr", 1)
    else:
        n_run_slots = aff["cpu"] // aff["cpr"]
    return n_run_slots


def remove_run_slot(run_slot_affinity_code):
    run_slot_str, aff_code = run_slot_affinity_code.split("_", 1)
    assert run_slot_str[-3:] == RUN_SLOT
    run_slot = int(run_slot_str[:-3])
    return run_slot, aff_code


def decode_affinity(affinity_code):
    codes = affinity_code.split("_")
    aff_kwargs = dict()
    for code in codes:
        abrv = code[-3:]
        if abrv not in ABBREVS:
            raise ValueError(f"Unrecognized affinity code abbreviation: {abrv}")
        value = int(code[:-3])
        aff_kwargs[abrv] = value
    return aff_kwargs


def build_cpu_affinity(slt, cpu, cpr, cpw=1, hto=None, res=0, skt=1, gpu=0):
    assert gpu == 0
    assert cpu % cpr == 0
    if hto is None:  # default, different from 0, which turns them OFF.
        hto = cpu
    n_run_slots = cpu // cpr
    assert slt <= n_run_slots
    min_core = slt * cpr + offset_for_socket(hto, cpu, skt, slt, n_run_slots)
    cores = tuple(range(min_core, min_core + cpr))
    worker_cores = cores[res:]
    assert len(worker_cores) % cpw == 0
    master_cpus = get_master_cpus(cores, hto)
    workers_cpus = get_workers_cpus(worker_cores, cpw, hto)
    affinity = dict(
        all_cpus=master_cpus,
        master_cpus=master_cpus,
        workers_cpus=workers_cpus,
        master_torch_threads=len(cores),
        worker_torch_threads=cpw,
    )
    return affinity


def build_gpu_affinity(slt, gpu, cpu, cxg=1, cxr=1, cpw=1, hto=None, res=0,
        skt=1):
    """Divides CPUs evenly among GPUs."""
    if cxr > 1:
        raise NotImplementedError  # (parallel training)
    n_ctx = gpu * cxg
    n_run_slots = n_ctx // cxr
    assert slt < n_run_slots
    assert cpu % n_ctx == 0
    cpr = cpu // n_ctx
    slt = (slt % gpu) * cxg + slt // gpu  # Re-order to spread over GPUs first.
    affinity = build_cpu_affinity(slt, cpu, cpr, cpw, hto, res, skt)
    affinity["cuda_idx"] = slt // cxg
    return affinity


def offset_for_socket(hto, cpu, skt, slt, n_run_slots):
    """If hto==cpu or skt==1, returns 0."""
    assert (hto - cpu) % skt == 0
    rem_cpu_per_skt = (hto - cpu) // skt
    slt_per_skt = n_run_slots // skt
    my_skt = slt // slt_per_skt
    return my_skt * rem_cpu_per_skt


def get_master_cpus(cores, hto):
    hyperthreads = tuple(c + hto for c in cores) if hto > 0 else ()
    return cores + hyperthreads


def get_workers_cpus(cores, cpw, hto):
    cpus = tuple(cores[i:i + cpw]
        for i in range(0, len(cores), cpw))
    if hto > 0:
        hyperthreads = tuple(c + hto for c in cores)
        hyperthreads = tuple(hyperthreads[i:i + cpw]
            for i in range(0, len(cores), cpw))
        cpus = tuple(c + h for c, h in zip(cpus, hyperthreads))
    return cpus


def build_affinities_gpu_1cpu_drive(slt, gpu, cpu, cxg=1, cxr=1, cpw=1,
        hto=None, skt=1):
    """OLD.
    Divides CPUs evenly among GPUs, with one CPU held open for each GPU, to
    drive it.  Workers assigned on the remaining CPUs.  Master permitted to use
    driver core + worker cores (good in case of multi-context per GPU and old
    alternating action server sampler, from accel_rl). GPU-driving CPUs grouped
    at the lowest numbered cores of each CPU socket.
    """
    if cxr > 1:
        raise NotImplementedError  # (parallel training)
    n_ctx = gpu * cxg
    n_run_slots = n_ctx // cxr
    assert slt < n_run_slots
    cpu_per_gpu = cpu // gpu
    sim_cpu_per_gpu = cpu_per_gpu - 1
    n_sim_cpu = cpu - gpu
    sim_cpu_per_ctx = n_sim_cpu // n_ctx

    assert gpu >= skt
    assert gpu % skt == 0
    gpu_per_skt = gpu // skt
    assert cpu % skt == 0
    cpu_per_skt = cpu // skt

    my_ctx = slt  # Different for multi-context run, not implemented.
    my_gpu = my_ctx // cxg
    my_skt = my_gpu // gpu_per_skt
    gpu_in_skt = my_gpu % gpu_per_skt
    gpu_core = gpu_in_skt + my_skt * cpu_per_skt
    ctx_in_gpu = my_ctx % cxg

    min_sim_core = (my_skt * cpu_per_skt + gpu_per_skt +
        gpu_in_skt * sim_cpu_per_gpu + ctx_in_gpu * sim_cpu_per_ctx)
    sim_cores = tuple(range(min_sim_core, min_sim_core + sim_cpu_per_ctx))

    assert len(sim_cores) % cpw == 0
    if hto is None:
        hto = cpu
    if hto > 0:
        hyperthreads = tuple(c + hto for c in sim_cores)
        workers_cpus = tuple(sim_cores[i:i + cpw] + hyperthreads[i:i + cpw]
            for i in range(0, len(sim_cores), cpw))
        master_cpus = (gpu_core,) + sim_cores + (gpu_core + hto,) + hyperthreads
    else:
        workers_cpus = tuple(sim_cores[i:i + cpw]
            for i in range(0, len(sim_cores), cpw))
        master_cpus = (gpu_core,) + sim_cores

    affinity = dict(
        all_cpus=master_cpus,
        master_cpus=master_cpus,
        workers_cpus=workers_cpus,
        master_torch_threads=1,
        worker_torch_threads=cpw,
        cuda_idx=my_gpu,
    )
    return affinity
