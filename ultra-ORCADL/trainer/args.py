import os
import re
import torch
import importlib
import transformers
from typing import Optional
from dataclasses import field, dataclass
from .utils_hf import cached_property, logging, requires_backends


logger = logging.get_logger(__name__)
log_levels = logging.get_log_levels_dict().copy()
trainer_log_levels = dict(**log_levels, passive=-1)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model: str = field(
        default='swin'
    )
    dataset: str = field(
        default='cmip6', metadata={"help": "Training dataset."}
    )
    lr_scheduler_type_custom: str = field(
        default=None
    )
    min_learning_rate: float = field(
        default=0.0, metadata={"help": "The minimum of learning rate."}
    )
    dist_port: Optional[int] = field(
        default=21111, metadata={"help": "The port used for distributed training"}
    )
    inputs_key_for_metrics: Optional[str] = field(
        default=None, metadata={"help": "The key in dictionary of inputs that will be passed to the `compute_metrics` function."}
    )

    def __post_init__(self):
        super().__post_init__()
        if self.include_inputs_for_metrics and self.inputs_key_for_metrics is None:
            raise ValueError(
                "--include_inputs_for_metrics requires inputs_key_for_metrics not to be None"
            )

    @property
    def world_size(self):
        """
        The number of processes used in parallel.
        """
        requires_backends(self, ["torch"])

        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            return int(os.environ['WORLD_SIZE'])
        elif 'SLURM_PROCID' in os.environ:
            return int(os.environ['SLURM_NTASKS'])
        else:
            assert 0, "Can't get the correct world_size."

    # @cached_property
    # def _setup_devices(self) -> "torch.device":
    #     requires_backends(self, ["torch"])

    #     logger.info("PyTorch: setting up devices")
    #     if torch.distributed.is_available() and torch.distributed.is_initialized() and self.local_rank == -1:
    #         logger.warning(
    #             "torch.distributed process group is initialized, but local_rank == -1. "
    #             "In order to use Torch DDP, launch your script with `python -m torch.distributed.launch"
    #         )

    #     if self.no_cuda:
    #         device = torch.device("cpu")
    #         self._n_gpu = 0
    #         self.local_rank = get_int_from_env(
    #             ["LOCAL_RANK", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK"],
    #             self.local_rank,
    #         )
    #         if self.local_rank != -1 and not torch.distributed.is_initialized():
    #             # Initializes distributed backend for cpu
    #             if self.xpu_backend not in ("mpi", "ccl", "gloo"):
    #                 raise ValueError(
    #                     "CPU distributed training backend is not properly set. "
    #                     "Please set '--xpu_backend' to either 'mpi' or 'ccl' or 'gloo'."
    #                 )
    #             if self.xpu_backend == "ccl":
    #                 assert 0

    #             # Try to get launch configuration from environment variables set by MPI launcher - works for Intel MPI, OpenMPI and MVAPICH
    #             rank = get_int_from_env(["RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK"], 0)
    #             size = get_int_from_env(["WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE"], 1)
    #             local_size = get_int_from_env(
    #                 ["MPI_LOCALNRANKS", "OMPI_COMM_WORLD_LOCAL_SIZE", "MV2_COMM_WORLD_LOCAL_SIZE"], 1
    #             )
    #             os.environ["RANK"] = str(rank)
    #             os.environ["WORLD_SIZE"] = str(size)
    #             os.environ["LOCAL_RANK"] = str(self.local_rank)
    #             if not os.environ.get("MASTER_PORT", None):
    #                 os.environ["MASTER_PORT"] = "29500"
    #             if not os.environ.get("MASTER_ADDR", None):
    #                 if local_size != size or self.xpu_backend != "mpi":
    #                     raise ValueError(
    #                         "Looks like distributed multinode run but MASTER_ADDR env not set, "
    #                         "please try exporting rank 0's hostname as MASTER_ADDR"
    #                     )
    #             if (
    #                 torch.get_num_threads() == 1
    #                 and get_int_from_env(["OMP_NUM_THREADS", "MKL_NUM_THREADS"], 0) == 0
    #                 and is_psutil_available()
    #             ):
    #                 import psutil

    #                 num_cpu_threads_per_process = int(psutil.cpu_count(logical=False) / local_size)
    #                 if num_cpu_threads_per_process == 0:
    #                     num_cpu_threads_per_process = 1
    #                 torch.set_num_threads(num_cpu_threads_per_process)
    #                 logger.info(
    #                     f"num_cpu_threads_per_process unset, we set it at {num_cpu_threads_per_process} to improve oob"
    #                     " performance."
    #                 )
    #             torch.distributed.init_process_group(
    #                 backend=self.xpu_backend, rank=rank, world_size=size, timeout=self.ddp_timeout_delta
    #             )

    #     elif self.local_rank == -1:
    #         if torch.distributed.is_available() and not torch.distributed.is_initialized():
    #             if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    #                 self.global_rank = int(os.environ["RANK"])
    #                 self.local_rank = int(os.environ['LOCAL_RANK'])
    #                 init_method = None
    #             elif 'SLURM_PROCID' in os.environ:
    #                 self.global_rank = int(os.environ['SLURM_PROCID'])
    #                 self.local_rank = int(os.environ['SLURM_LOCALID'])
    #                 ip_addr = get_ip(os.environ['SLURM_STEP_NODELIST'])
    #                 init_method = ip_addr + str(self.dist_port)
    #             else:
    #                 assert 0
    #             if self.xpu_backend and self.xpu_backend in ("mpi", "gloo"):
    #                 torch.distributed.init_process_group(
    #                     backend=self.xpu_backend, init_method=init_method, world_size=self.world_size, rank=self.global_rank, timeout=self.ddp_timeout_delta)
    #             else:
    #                 torch.distributed.init_process_group(
    #                     backend="nccl", init_method=init_method, world_size=self.world_size, rank=self.global_rank, timeout=self.ddp_timeout_delta)
    #             device = torch.device("cuda", self.local_rank)
    #             self._n_gpu = 1
    #         else:
    #             # if n_gpu is > 1 we'll use nn.DataParallel.
    #             # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
    #             # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
    #             # trigger an error that a device index is missing. Index 0 takes into account the
    #             # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
    #             # will use the first GPU in that env, i.e. GPU#1
    #             device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #             # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
    #             # the default value.
    #             self._n_gpu = torch.cuda.device_count()

    #     else:
    #         # Here, we'll use torch.distributed.
    #         # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    #         if not torch.distributed.is_initialized():
    #             if self.xpu_backend and self.xpu_backend in ("mpi", "gloo"):
    #                 torch.distributed.init_process_group(backend=self.xpu_backend, timeout=self.ddp_timeout_delta)
    #             else:
    #                 torch.distributed.init_process_group(backend="nccl", timeout=self.ddp_timeout_delta)
    #         device = torch.device("cuda", self.local_rank)
    #         self.global_rank = int(os.environ["RANK"])
    #         self._n_gpu = 1

    #     if device.type == "cuda":
    #         torch.cuda.set_device(device)

    #     return device


def is_psutil_available():
    return importlib.util.find_spec("psutil") is not None


def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


def get_ip(ip_list):
    if "," in ip_list:
        ip_list = ip_list.split(',')[0]
    if "[" in ip_list:
        ipbefore_4, ip4 = ip_list.split('[')
        ip4 = re.findall(r"\d+", ip4)[0]
        ip1, ip2, ip3 = ipbefore_4.split('-')[-4:-1]
    else:
        ip1, ip2, ip3, ip4 = ip_list.split('-')[-4:]
    ip_addr = "tcp://" + ".".join([ip1, ip2, ip3, ip4]) + ":"
    return ip_addr
