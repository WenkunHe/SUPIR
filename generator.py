# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from typing import Optional

from omegaconf import OmegaConf, MISSING

from supir_launcher import ImageGeneratorConfig

from typing import TypeVar
T = TypeVar("T")

def get_config(config_class) -> T:
    cfg = OmegaConf.structured(config_class)

    additional_cfg = OmegaConf.from_cli()
    if "yaml" in additional_cfg:
        yaml_cfg = OmegaConf.load(additional_cfg.yaml)
        yaml_cfg = OmegaConf.masked_copy(yaml_cfg, cfg.keys())
        additional_cfg = OmegaConf.merge(yaml_cfg, additional_cfg)
        additional_cfg.pop("yaml")

    if "json" in additional_cfg:
        additional_cfg = OmegaConf.merge(additional_cfg.json, additional_cfg)
        additional_cfg.pop("json")

    cfg = OmegaConf.to_object(OmegaConf.merge(cfg, additional_cfg))
    return cfg


@dataclass
class SlurmBaseConfig:
    account: str = "nvr_elm_llm"
    cluster: str = "draco"
    partition: Optional[str] = None
    time: str = "04:00:00"
    num_nodes: int = 1
    job_name: str = "test"
    gpus_per_node: int = 8
    exclusive: bool = True
    cpus_per_task: Optional[int] = None
    mem_per_cpu: Optional[str] = None
    container_image: Optional[str] = None
    container_mounts: Optional[str] = None
    container_mount_home: bool = False

    pre_cmd: Optional[str] = None

    first_task_id: int = 1
    last_task_id: int = 20
    max_running_tasks: int = 1

    global_batch_size: int = MISSING
    eval_global_batch_size: int = 1
    run_dir: Optional[str] = None

    interactive: bool = False
    only_print_script: bool = False
    only_generate_script: bool = False

    entrance: str = MISSING


class SlurmBase:
    def __init__(self, cfg: SlurmBaseConfig):
        self.cfg = cfg

        if cfg.partition is None:
            if cfg.cluster == "cw":
                cfg.partition = "batch"
            elif cfg.cluster == "eos":
                cfg.partition = "batch"
            elif cfg.cluster == "draco":
                cfg.partition = "batch_block1"
            elif cfg.cluster == "nrt":
                cfg.partition = "batch_block1"
            elif cfg.cluster == "cs":
                cfg.partition = "polar3,polar,grizzly,polar4"
            elif cfg.cluster == "hanlab":
                cfg.partition = ""
            else:
                raise ValueError(f"cluster {cfg.cluster} is not supported")

        self.project_dir = os.path.dirname(os.path.dirname(__file__))
        self.environment_dir = os.path.dirname(os.popen("which python").read())
        home_dir = os.path.expanduser("~")
        if home_dir in ["/home/junyuc", "/homes/junyuc", "/FirstIntelligence/home/junyuc"]:
            self.user = "junyuc"
        elif home_dir == "/home/hcai":
            self.user = "hcai"
        elif home_dir in ["/home/wenkunh", "/FirstIntelligence/home/wenkunh"]:
            self.user = "wenkunh"
        elif home_dir == "/home/yujlin":
            self.user = "yujlin"
        elif home_dir == "/home/dongyun":
            self.user = "dongyun"
        elif home_dir == "/home/jasonlu":
            self.user = "jasonlu"
        elif home_dir in ["/home/yuchao", "/home/svu/e0974140", "/root"]:
            self.user = "yuchao"
        else:
            raise ValueError(f"user {home_dir} is not supported")
        self.root_exp_dir = f"exp_{self.user}"

        self.batch_size = cfg.global_batch_size // cfg.num_nodes // cfg.gpus_per_node
        self.eval_batch_size = cfg.eval_global_batch_size // cfg.num_nodes // cfg.gpus_per_node

        self.run_dir = self.generate_run_dir()
        self.yaml_path = self.generate_yaml_path()
        self.bash_script_path = self.generate_bash_script_path()

    def generate_run_dir(self) -> str:
        raise NotImplementedError

    def generate_yaml_path(self) -> str:
        raise NotImplementedError

    def generate_bash_script_path(self) -> str:
        raise NotImplementedError

    def generate_header(self):
        header_length = max(len(self.cfg.account), len(self.cfg.partition), len(self.cfg.time), len(self.cfg.job_name))

        header = (
            f"#!/bin/bash\n"
            f"#SBATCH -A {self.cfg.account: <{header_length}} #account\n"
            f"#SBATCH -p {self.cfg.partition: <{header_length}} #partition\n"
            f"#SBATCH -t {self.cfg.time: <{header_length}} #wall time limit, hr:min:sec\n"
            f"#SBATCH -N {str(self.cfg.num_nodes): <{header_length}} #number of nodes\n"
            f"#SBATCH -J {self.cfg.job_name: <{header_length}} #job name\n"
            f"#SBATCH --array={self.cfg.first_task_id}-{self.cfg.last_task_id}%{self.cfg.max_running_tasks}\n"
            f"#SBATCH --output={self.run_dir}/slurm_out/%A_%a.out\n"
        )

        if self.cfg.cluster in ["cw", "draco", "nrt", "cs"]:
            header += f"#SBATCH --gpus-per-node {self.cfg.gpus_per_node}\n"
            if self.cfg.exclusive:
                header += f"#SBATCH --exclusive\n"
        elif self.cfg.cluster in ["eos"]:
            pass
        else:
            raise ValueError(f"cluster {self.cfg.cluster} is not supported")

        if self.cfg.cpus_per_task is not None:
            header += f"#SBATCH --cpus-per-task {self.cfg.cpus_per_task}\n"
        if self.cfg.mem_per_cpu is not None:
            header += f"#SBATCH --mem-per-cpu {self.cfg.mem_per_cpu}\n"

        header += f"\n"

        if self.cfg.num_nodes > 1:
            header += (
                f"nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )\n"
                f"nodes_array=($nodes)\n"
                f"head_node=${{nodes_array[0]}}\n"
                f'head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)\n'
                f"\n"
            )

        header += f"export LOGLEVEL=INFO\n" f'export PATH="{self.environment_dir}:$PATH"\n' f"\n"
        header += "export TORCHRUN_PORT=$((SLURM_ARRAY_TASK_ID + 38344))\n"

        header += f"cd {self.project_dir}\n" f"\n" f"read -r -d '' cmd <<EOF\n"

        if self.cfg.pre_cmd is not None:
            header += f"{self.cfg.pre_cmd}; \\\n"
        if self.cfg.num_nodes == 1:
            if self.cfg.gpus_per_node == 1:
                header += f"python "
            else:
                header += f"torchrun --nnodes=1 --nproc_per_node={self.cfg.gpus_per_node} --rdzv_endpoint localhost:$TORCHRUN_PORT \\\n"
        elif self.cfg.num_nodes > 1:
            header += f"torchrun --nnodes={self.cfg.num_nodes} --nproc_per_node={self.cfg.gpus_per_node} --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:$TORCHRUN_PORT \\\n"
        else:
            raise ValueError(f"num nodes {self.cfg.num_nodes} is not supported")

        return header

    def generate_footer(self):
        srun_args = ""
        if self.cfg.container_image is not None:
            srun_args += f"--container-image {self.cfg.container_image} "
        if self.cfg.container_mount_home:
            srun_args += f"--container-mount-home "
        if self.cfg.container_mounts:
            srun_args += f"--container-mounts {self.cfg.container_mounts} "
        footer = f"EOF\n" f"\n" f'srun {srun_args}bash -c "${{cmd}}"\n'
        return footer

    def generate_bash_script(self):
        header, footer = self.generate_header(), self.generate_footer()
        script = header
        script += f"-m {self.cfg.entrance} yaml={self.yaml_path}\n"
        script += footer
        return script

    def launch(self):
        print(f"run_dir: {self.run_dir}")
        if self.cfg.only_print_script:
            print(self.generate_bash_script())
            return
        os.makedirs(self.run_dir, exist_ok=True)
        OmegaConf.save(self.cfg, self.yaml_path)
        if self.cfg.interactive:
            print(f"python -m {self.cfg.entrance} yaml={self.yaml_path}")
            print(
                f"torchrun --nnodes=1 --nproc_per_node={self.cfg.gpus_per_node} -m {self.cfg.entrance} yaml={self.yaml_path}"
            )
        else:
            with open(self.bash_script_path, "w") as f:
                f.write(self.generate_bash_script())
            if self.cfg.only_generate_script:
                print(f"sbatch {self.bash_script_path}")
            else:
                os.system(f"sbatch {self.bash_script_path}")


@dataclass
class SlurmImageGeneratorConfig(ImageGeneratorConfig, SlurmBaseConfig):
    job_name: str = "t2i_image_generation"
    entrance: str = ".supir_launcher"
    global_batch_size: int = 0
    gpus_per_node: int = 1
    cpus_per_task: Optional[int] = 12
    mem_per_cpu: Optional[str] = "16G"
    first_task_id: int = 0
    max_running_tasks: int = 16
    exclusive: bool = False


class SlurmImageGenerator(SlurmBase):
    def __init__(self, cfg: SlurmImageGeneratorConfig):
        super().__init__(cfg)
        self.cfg: SlurmImageGeneratorConfig

    def generate_run_dir(self) -> str:
        assert isinstance(self.cfg.run_dir, str)
        if "tmp" in self.cfg.run_dir:
            return self.cfg.run_dir
        else:
            self.cfg.run_dir += f"_{self.cfg.cluster}"
            return self.cfg.run_dir

    def generate_bash_script(self):
        header, footer = self.generate_header(), self.generate_footer()
        script = header
        script += f"-m {self.cfg.entrance} yaml={self.yaml_path} task_id=${{SLURM_ARRAY_TASK_ID}}\n"
        script += footer
        return script

    def generate_yaml_path(self) -> str:
        return os.path.join(self.run_dir, f"config.yaml")

    def generate_bash_script_path(self) -> str:
        return os.path.join(self.run_dir, f"slurm.sh")


def main():
    cfg = get_config(SlurmImageGeneratorConfig)
    slurm = SlurmImageGenerator(cfg)
    slurm.launch()


if __name__ == "__main__":
    main()
