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

import json
import os
import shutil
import tarfile
from dataclasses import dataclass
import time

import numpy as np
import torch
from omegaconf import OmegaConf, MISSING
from PIL import Image
from torch.utils.data import Dataset

from SUPIR.util import create_SUPIR_model, convert_dtype, PIL2Tensor, Tensor2PIL
from llava.llava_agent import LLavaAgent
from CKPT_PTH import LLAVA_MODEL_PATH
import io

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
class ImageGeneratorConfig:
    # slurm
    task_id: int = 0

    # file paths
    prompt_dir: str = MISSING
    save_dir: str = MISSING
    num_jsons_per_task: int = 1

    # Range
    range_id: int = 0
    num_samples_per_range: int = 70
    upscale: int = 4


class PromptDataset(Dataset):
    def __init__(self, json_path: str):
        with open(json_path, "r") as json_file:
            self.meta = json.load(json_file)
        self.prompts = []
        for key, value in self.meta.items():
            self.prompts.append(
                {
                    "key": key.replace("/", "_"),
                    "meta": value,
                }
            )

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


class ImageGenerator:
    def __init__(self, cfg: ImageGeneratorConfig):
        self.cfg = cfg
        self.device = torch.device("cuda")
        self.cfg.save_dir = self.cfg.save_dir + '_' + str(self.cfg.range_id).zfill(2)

        self.model = create_SUPIR_model('options/SUPIR_v0.yaml', SUPIR_sign='Q')
        self.model.init_tile_vae(encoder_tile_size=2048, decoder_tile_size=256)
        self.model.ae_dtype = convert_dtype("bf16")
        self.model.model.dtype = convert_dtype("fp16")
        self.model = self.model.to(self.device)
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        # load LLaVA
        self.llava_agent = LLavaAgent(LLAVA_MODEL_PATH, device=self.device, load_8bit=False, load_4bit=False)

        self.a_prompt = 'Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations.'
        self.n_prompt = 'painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth'
    
    def upsample(self, LQ_ips: Image):
        start_time = time.time()

        LQ_img, h0, w0 = PIL2Tensor(LQ_ips, upsacle=self.cfg.upscale, min_size=1024)
        LQ_img = LQ_img.unsqueeze(0).to(self.device)[:, :3, :, :]

        # step 1: Pre-denoise for LLaVA, resize to 512
        LQ_img_512, h1, w1 = PIL2Tensor(LQ_ips, upsacle=self.cfg.upscale, min_size=1024, fix_resize=512)
        LQ_img_512 = LQ_img_512.unsqueeze(0).to(self.device)[:, :3, :, :]
        clean_imgs = self.model.batchify_denoise(LQ_img_512)
        clean_PIL_img = Tensor2PIL(clean_imgs[0], h1, w1)

        # step 2: LLaVA
        captions = self.llava_agent.gen_image_caption([clean_PIL_img])


        # step 3: Diffusion Process
        samples = self.model.batchify_sample(LQ_img, captions, num_steps=16, restoration_scale=-1, s_churn=5,
                                        s_noise=1.01, cfg_scale=5.5, control_scale=0.96, seed=1234,
                                        num_samples=1, p_p=self.a_prompt, n_p=self.n_prompt, color_fix_type='Wavelet',
                                        use_linear_CFG=True, use_linear_control_scale=False,
                                        cfg_scale_start=1.0, control_scale_start=0.)

        end_time = time.time()
        print("Finished Processing! Time", end_time-start_time)

        return Tensor2PIL(samples[0], h0, w0)

    def process_single_tar(self, tar_rel_path: str, ranges):
        tar_path = os.path.join(self.cfg.prompt_dir, tar_rel_path)
        save_tar_path = os.path.join(self.cfg.save_dir, tar_rel_path)
        L, R = ranges
        print(L, R)

        if os.path.exists(save_tar_path):
            try:
                with tarfile.open(save_tar_path, "r") as tar:
                    tar.getmembers()
                print(f"TarFile {tar_path} already exists")
                return

            except:
                pass

        tmp_tar_dir = save_tar_path.removesuffix(".tar")
        os.makedirs(tmp_tar_dir, exist_ok=True)

        output_jpg_paths = []
        output_json_paths = []

        with tarfile.open(tar_path, 'r') as tar:
            member_names = tar.getnames()

        with tarfile.open(tar_path, 'r') as tar:
            for idx in range(L, R):
                jpg_name = member_names[idx*2]
                json_name = member_names[idx*2+1]

                output_jpg_path = os.path.join(tmp_tar_dir, jpg_name)
                output_json_path = os.path.join(tmp_tar_dir, json_name)
                output_jpg_paths.append(output_jpg_path)
                output_json_paths.append(output_json_path)
                if os.path.exists(output_jpg_path) and os.path.exists(output_json_path):
                    try:
                        with Image.open(output_jpg_path) as img:
                            img.verify()
                        with open(output_json_path, 'r') as f:
                            json.load(f)
                    except (IOError, OSError, json.JSONDecodeError):
                        pass

                with tar.extractfile(json_name) as f:
                    content = f.read()
                    meta = json.loads(content)

                height, width = meta["height"] * 4, meta["width"] * 4
                prompt = meta["prompt"] if "prompt" in meta else meta["captions"][0]
                captions = meta["captions"] if "captions" in meta else [meta["prompt"]]
                clip_scores = meta["clip_scores"] if "clip_scores" in meta else [30.0] * len(captions)

                with tar.extractfile(jpg_name) as f:
                    image_data = f.read()
                    img = Image.open(io.BytesIO(image_data))

                image = self.upsample(img)
                image.save(output_jpg_path)
                    
                meta = {
                    "prompt": prompt,
                    "captions": captions,
                    "clip_scores": clip_scores,
                    "height": height,
                    "width": width,
                }
                with open(output_json_path, "w") as json_file:
                    json.dump(meta, json_file, indent=4)

        with tarfile.open(save_tar_path, "w") as tar:
            for output_jpg_path, output_json_path in zip(output_jpg_paths, output_json_paths):
                tar.add(output_jpg_path, arcname=os.path.basename(output_jpg_path))
                tar.add(output_json_path, arcname=os.path.basename(output_json_path))

        shutil.rmtree(f"{tmp_tar_dir}")

    def work(self):
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        prompt_filenames_all = os.listdir(self.cfg.prompt_dir)
        prompt_filenames = []
        for filename in prompt_filenames_all:
            if filename.endswith(".tar"):
                prompt_filenames.append(filename)
        prompt_filenames.sort()
        prompt_filenames = prompt_filenames[
            self.cfg.task_id * self.cfg.num_jsons_per_task : (self.cfg.task_id + 1) * self.cfg.num_jsons_per_task
        ]
        ranges = (self.cfg.range_id * self.cfg.num_samples_per_range, (self.cfg.range_id + 1) * self.cfg.num_samples_per_range)

        for prompt_filename in prompt_filenames:
            self.process_single_tar(prompt_filename, ranges)


def main():
    torch.set_grad_enabled(False)
    cfg = get_config(ImageGeneratorConfig)
    image_generator = ImageGenerator(cfg)
    image_generator.work()


if __name__ == "__main__":
    main()


"""
python -m generator \
    prompt_dir=1k \
    save_dir=4k \
    run_dir=exp_wenkunh/upsample/4k \
    range_id=1 \
    num_samples_per_range=5 \
    interactive=True
"""
