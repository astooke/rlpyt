

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

ABBREVS = [N_GPU, CONTEXTS_PER_GPU, CONTEXTS_PER_RUN, N_CPU_CORES,
    HYPERTHREAD_OFFSET, N_SOCKET, CPU_PER_RUN, CPU_PER_WORKER]


# API

def encode_affinity(n_cpu_cores=1, n_gpu=0, contexts_per_gpu=1,
        contexts_per_run=1, cpu_per_run=1, cpu_per_worker=1,
        hyperthread_offset=None, n_socket=1, run_slot=None):
    """Use in run script to specify computer configuration."""
    affinity_code = f"{n_cpu_cores}{N_CPU_CORES}_{n_gpu}{N_GPU}"
    if contexts_per_gpu > 1:
        affinity_code += f"_{contexts_per_gpu}{CONTEXT_PER_GPU}"
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
        return build_affinities_gpu(run_slot, **aff_params)
    return build_affinities_cpu(run_slot, **aff_params)


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


def build_affinities_cpu(slt, cpu, cpr, cpw=1, hto=None, gpu=0):
    assert gpu == 0
    n_run_slots = cpu // cpr
    assert slt <= n_run_slots
    cores = tuple(range(slt * cpr, (slt + 1) * cpr))
    assert len(cores) % cpw == 0
    if hto is None:  # default, different from 0, which turns them OFF.
        hto = cpu
    if hto > 0:
        hyperthreads = tuple(c + hto for c in cores)
        workers_cpus = tuple(cores[i:i + cpw] + hyperthreads[i:i + cpw]
            for i in range(0, cpu, cpw))
        master_cpus = cores + hyperthreads
    else:
        workers_cpus = tuple(cores[i:i + cpw] for i in range(0, cpu, cpw))
        master_cpus = cores
    return dict(master_cpus=master_cpus, workers_cpus=workers_cpus)


def build_affinities_gpu(slt, gpu, cpu, cxg=1, cxr=1, cpw=1, hto=None, skt=1):
    """Divides CPUs evenly among GPUs, with one CPU held open for each GPU, to
    drive it.  Workers assigned on the remaining CPUs.  Master permitted to use
    all cores (good in case of multi-context per GPU).  GPU-driving CPUs grouped
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
    n_worker = len(sim_cores) // cpw
    if hto is None:
        hto = cpu
    if hto > 0:
        hyperthreads = tuple(c + hto for c in sim_cores)
        workers_cpus = tuple(sim_cores[i:i + cpw] + hyperthreads[i:i + cpw]
            for i in range(0, len(sim_cores), cpw))
        master_cpus = (gpu_core,) + sim_cores + (gpu_core + hto,) + hyperthreads
    else:
        workers_cpus = tuple(cores[i:i + cpw]
            for i in range(0, len(sim_cores), cpw))
        master_cpus = (gpu_core,) + sim_cores

    return dict(master_cpus=master_cpus, workers_cpus=workers_cpus,
        cuda_idx=my_gpu)
