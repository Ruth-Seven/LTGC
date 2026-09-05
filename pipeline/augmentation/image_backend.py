"""Diffusers implementation; GPU/model details do not leak into orchestration."""
import os
import time
from dataclasses import dataclass
from typing import Any
from .configuration import GenerationConfig
from .records import GenerationJob
from .references import prepare_reference


@dataclass
class RenderResult:
    image: Any
    references: list[Any]
    peak_memory_gib: float


class DiffusersBackend:
    def __init__(self, config: GenerationConfig):
        import torch
        from diffusers import FluxKontextPipeline, Flux2KleinPipeline
        from pathlib import Path
        self.config, self.torch = config, torch
        start = time.perf_counter()
        klein = config.backend == 'flux-klein'
        kind = Flux2KleinPipeline if klein else FluxKontextPipeline
        self.pipe = kind.from_pretrained(config.model_path, torch_dtype=torch.bfloat16,
                                         local_files_only=Path(config.model_path).is_dir())
        runtime = dict(backend=config.backend, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        if klein:
            self.pipe.to('cuda:0')
            residency = {name:sorted({str(param.device) for param in module.parameters()})
                         for name, module in self.pipe.components.items() if isinstance(module, torch.nn.Module)}
            assert all(devices == ['cuda:0'] for devices in residency.values()), residency
            runtime.update(cpu_offload=False, component_devices=residency)
        else:
            from accelerate import cpu_offload
            transformer = self.pipe.transformer
            runtime.update(transformer_blocks=len(transformer.transformer_blocks) + len(transformer.single_transformer_blocks),
                blocks_per_group=config.kontext_blocks_per_group, use_stream=False, cpu_offload=True)
            transformer.enable_group_offload(onload_device=torch.device('cuda:0'), offload_device=torch.device('cpu'),
                offload_type='block_level', num_blocks_per_group=config.kontext_blocks_per_group, use_stream=False)
            self.pipe.text_encoder.to('cuda:0')
            cpu_offload(self.pipe.text_encoder_2, execution_device=torch.device('cuda:0'))
            self.pipe.vae.to('cuda:0')
        torch.cuda.synchronize()
        runtime['model_load_seconds'] = time.perf_counter() - start
        self.runtime = runtime
        self.pipe.vae.enable_tiling()
        self.pipe.set_progress_bar_config(disable=False)

    def render(self, job: GenerationJob):
        config, torch = self.config, self.torch
        torch.cuda.reset_peak_memory_stats()
        references = [prepare_reference(path, config.size, self.pipe.vae_scale_factor * 2) for path in job.image_paths]
        size_kwargs = {} if config.backend == 'flux-klein' else dict(max_area=config.size**2, _auto_resize=config.size >= 1024)
        result = self.pipe(image=references if config.two_references else references[0],
            prompt=job.prompt, width=config.size, height=config.size,
            num_inference_steps=config.steps, guidance_scale=config.guidance,
            generator=torch.Generator('cpu').manual_seed(job.record.sample.generation_seed), **size_kwargs).images[0]
        return RenderResult(result, references, torch.cuda.max_memory_allocated() / 1024**3)

    def close(self):
        import gc
        del self.pipe
        gc.collect()
        self.torch.cuda.empty_cache()
