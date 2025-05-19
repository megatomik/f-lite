import argparse
import datetime
import logging
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from einops import rearrange
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from f_lite.pipeline import FLitePipeline
from f_lite.precomputed_utils import (
    create_precomputed_data_loader,
    forward_with_precomputed_data,
)

# Set up logger
logger = get_logger(__name__)

# Enable TF32 for faster training (only on NVIDIA Ampere or newer GPUs)
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.multiprocessing.set_sharing_strategy('file_system')

import json
def save_log_line(log, path="log_db.jsonl"):
    with open(path, "a") as f:
        json.dump(log, f)
        f.write("\n")

# --- START: Added from utils.py and InternalData_ms.py ---
ASPECT_RATIO_1024 = {
    '0.25': [512., 2048.], '0.26': [512., 1984.], '0.27': [512., 1920.], '0.28': [512., 1856.],
    '0.32': [576., 1792.], '0.33': [576., 1728.], '0.35': [576., 1664.], '0.4':  [640., 1600.],
    '0.42':  [640., 1536.], '0.48': [704., 1472.], '0.5': [704., 1408.], '0.52': [704., 1344.],
    '0.57': [768., 1344.], '0.6': [768., 1280.], '0.68': [832., 1216.], '0.72': [832., 1152.],
    '0.78': [896., 1152.], '0.82': [896., 1088.], '0.88': [960., 1088.], '0.94': [960., 1024.],
    '1.0':  [1024., 1024.], '1.07': [1024.,  960.], '1.13': [1088.,  960.], '1.21': [1088.,  896.],
    '1.29': [1152.,  896.], '1.38': [1152.,  832.], '1.46': [1216.,  832.], '1.67': [1280.,  768.],
    '1.75': [1344.,  768.], '2.0':  [1408.,  704.], '2.09':  [1472.,  704.], '2.4':  [1536.,  640.],
    '2.5':  [1600.,  640.], '2.89':  [1664.,  576.], '3.0':  [1728.,  576.], '3.11':  [1792.,  576.],
    '3.62':  [1856.,  512.], '3.75':  [1920.,  512.], '3.88':  [1984.,  512.], '4.0':  [2048.,  512.],
}

ASPECT_RATIO_512 = {
     '0.25': [256.0, 1024.0], '0.26': [256.0, 992.0], '0.27': [256.0, 960.0], '0.28': [256.0, 928.0],
     '0.32': [288.0, 896.0], '0.33': [288.0, 864.0], '0.35': [288.0, 832.0], '0.4': [320.0, 800.0],
     '0.42': [320.0, 768.0], '0.48': [352.0, 736.0], '0.5': [352.0, 704.0], '0.52': [352.0, 672.0],
     '0.57': [384.0, 672.0], '0.6': [384.0, 640.0], '0.68': [416.0, 608.0], '0.72': [416.0, 576.0],
     '0.78': [448.0, 576.0], '0.82': [448.0, 544.0], '0.88': [480.0, 544.0], '0.94': [480.0, 512.0],
     '1.0': [512.0, 512.0], '1.07': [512.0, 480.0], '1.13': [544.0, 480.0], '1.21': [544.0, 448.0],
     '1.29': [576.0, 448.0], '1.38': [576.0, 416.0], '1.46': [608.0, 416.0], '1.67': [640.0, 384.0],
     '1.75': [672.0, 384.0], '2.0': [704.0, 352.0], '2.09': [736.0, 352.0], '2.4': [768.0, 320.0],
     '2.5': [800.0, 320.0], '2.89': [832.0, 288.0], '3.0': [864.0, 288.0], '3.11': [896.0, 288.0],
     '3.62': [928.0, 256.0], '3.75': [960.0, 256.0], '3.88': [992.0, 256.0], '4.0': [1024.0, 256.0]
}

ASPECT_RATIO_256 = {
     '0.25': [128.0, 512.0], '0.26': [128.0, 496.0], '0.27': [128.0, 480.0], '0.28': [128.0, 464.0],
     '0.32': [144.0, 448.0], '0.33': [144.0, 432.0], '0.35': [144.0, 416.0], '0.4': [160.0, 400.0],
     '0.42': [160.0, 384.0], '0.48': [176.0, 368.0], '0.5': [176.0, 352.0], '0.52': [176.0, 336.0],
     '0.57': [192.0, 336.0], '0.6': [192.0, 320.0], '0.68': [208.0, 304.0], '0.72': [208.0, 288.0],
     '0.78': [224.0, 288.0], '0.82': [224.0, 272.0], '0.88': [240.0, 272.0], '0.94': [240.0, 256.0],
     '1.0': [256.0, 256.0], '1.07': [256.0, 240.0], '1.13': [272.0, 240.0], '1.21': [272.0, 224.0],
     '1.29': [288.0, 224.0], '1.38': [288.0, 208.0], '1.46': [304.0, 208.0], '1.67': [320.0, 192.0],
     '1.75': [336.0, 192.0], '2.0': [352.0, 176.0], '2.09': [368.0, 176.0], '2.4': [384.0, 160.0],
     '2.5': [400.0, 160.0], '2.89': [416.0, 144.0], '3.0': [432.0, 144.0], '3.11': [448.0, 144.0],
     '3.62': [464.0, 128.0], '3.75': [480.0, 128.0], '3.88': [496.0, 128.0], '4.0': [512.0, 128.0]
}

def get_closest_ratio(height: float, width: float, ratios: dict):
    aspect_ratio = height / width
    closest_ratio_key_str = min(ratios.keys(), key=lambda ratio_str: abs(float(ratio_str) - aspect_ratio))
    return ratios[closest_ratio_key_str], float(closest_ratio_key_str)

def get_aspect_ratio_map(base_resolution: int):
    if base_resolution == 256:
        return ASPECT_RATIO_256
    elif base_resolution == 512:
        return ASPECT_RATIO_512
    elif base_resolution == 1024:
        return ASPECT_RATIO_1024
    else:
        logger.warning(
            f"No predefined aspect ratio map for base_resolution {base_resolution}. "
            "Aspect ratio bucketing might not work as expected. "
            "Supported: 256, 512, 1024."
        )
        return None
# --- END: Added from utils.py and InternalData_ms.py ---


def parse_args():
    parser = argparse.ArgumentParser(description="DiT (Diffusion Transformer) Fine-tuning Script")
    
    # Model parameters
    parser.add_argument("--pretrained_model_path", type=str, default=None, required=True,
                        help="Path to pretrained model")
    parser.add_argument("--model_width", type=int, default=3072, 
                        help="Model width")
    parser.add_argument("--model_depth", type=int, default=40,
                        help="Model depth")
    parser.add_argument("--model_head_dim", type=int, default=256,
                        help="Attention head dimension")
    parser.add_argument("--rope_base", type=int, default=10_000,
                        help="Base for RoPE positional encoding")
    
    # Data parameters
    parser.add_argument("--train_data_path", type=str, required=False,
                        help="Path to training dataset, supports CSV files or image directories")
    parser.add_argument("--val_data_path", type=str, default=None,
                        help="Path to validation dataset, supports CSV files or image directories")
    parser.add_argument("--base_image_dir", type=str, default=None,
                        help="Base directory for image paths in CSV files")
    parser.add_argument("--image_column", type=str, default="image_path",
                        help="Column name in CSV containing image paths")
    parser.add_argument("--caption_column", type=str, default="caption",
                        help="Column name in CSV containing text captions")
    parser.add_argument("--resolution", type=int, default=None, # Changed default to None, will be handled
                        help="Base image resolution for training (e.g., 512 for 512xN buckets). If None, old logic might apply or error if bucketing enabled.")
    parser.add_argument("--center_crop", action="store_true", # This is now implicitly handled by aspect ratio logic
                        help="Whether to center crop images (used if not using aspect ratio bucketing)")
    parser.add_argument("--random_flip", action="store_true",
                        help="Whether to randomly flip images horizontally")
    parser.add_argument("--use_resolution_buckets", action="store_true",
                        help="Group images with same resolution into batches using aspect ratio maps.")
    
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=1,
                        help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum number of training steps, overrides num_epochs if provided")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay coefficient")
    parser.add_argument("--lr_scheduler", type=str, default="linear",
                        choices=["linear", "cosine", "constant"],
                        help="Learning rate scheduler type")
    parser.add_argument("--num_warmup_steps", type=int, default=0,
                        help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="Whether to use 8-bit Adam optimizer from bitsandbytes")
    parser.add_argument("--use_precomputed_data", action="store_true",
                help="Whether to use precomputed VAE latents and text embeddings")
    parser.add_argument("--precomputed_data_dir", type=str, default=None,
                    help="Directory containing precomputed data")

    # LoRA parameters
    parser.add_argument("--use_lora", action="store_true",
                        help="Whether to use LoRA for fine-tuning")
    parser.add_argument("--train_only_lora", action="store_true",
                        help="Whether to freeze base model and train only LoRA weights")
    parser.add_argument("--lora_rank", type=int, default=64,
                        help="Rank of LoRA matrices")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="Scaling factor for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="Dropout probability for LoRA layers")
    parser.add_argument("--lora_target_modules", type=str, default="qkv,q,context_kv,proj",
                        help="Comma-separated list of target modules for LoRA")
    parser.add_argument("--lora_checkpoint", type=str, default=None,
                        help="Path to LoRA checkpoint to resume from")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="dit-finetuned",
                        help="Output directory for saving model and checkpoints")
    parser.add_argument("--checkpointing_steps", type=int, default=500,
                        help="Save a checkpoint of the training state every X updates")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None,
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume training from checkpoint")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],
                        help="Mixed precision training type")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for gradient clipping")
    
    # Logging and evaluation parameters
    parser.add_argument("--logging_dir", type=str, default="logs",
                        help="Logging directory")
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        choices=["tensorboard", "wandb", "all"],
                        help="Logging integration to use")
    parser.add_argument("--project_name", type=str, default="dit-finetune",
                        help="Project name for wandb logging")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name for the experiment")
    parser.add_argument("--sample_every", type=int, default=500,
                        help="Generate sample images every X steps")
    parser.add_argument("--eval_every", type=int, default=500,
                        help="Run evaluation every X steps")
    parser.add_argument("--batch_multiplicity", type=int, default=1,
                        help="Repeat batch samples this many times")
    parser.add_argument("--sample_prompts_file", type=str, default=None,
                    help="Path to a text file containing prompts for sample image generation, one per line")
    
    return parser.parse_args()

class ResolutionBucketSampler(torch.utils.data.BatchSampler):
    """Group images by target resolution (from aspect ratio map) to ensure consistent resolution within a batch"""
    
    def __init__(self, dataset, batch_size, aspect_ratio_map, shuffle=True, drop_last=True):
        if aspect_ratio_map is None:
            raise ValueError("ResolutionBucketSampler requires an aspect_ratio_map.")
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.aspect_ratio_map = aspect_ratio_map
        
        # Group images by target resolution
        self.buckets = {}
        logger.info("Creating resolution buckets for ResolutionBucketSampler...")
        for idx in tqdm(range(len(dataset)), desc="Analyzing image dimensions for bucketing"):
            image_path = dataset.data_entries[idx]["image_path"]
            try:
                with Image.open(image_path) as img:
                    ori_w, ori_h = img.size
            except Exception as e:
                logger.warning(f"Could not open image {image_path} for bucketing: {e}. Skipping.")
                continue
            
            # Calculate target H, W based on the aspect ratio map
            closest_size_dims, _ = get_closest_ratio(float(ori_h), float(ori_w), self.aspect_ratio_map)
            target_h, target_w = int(closest_size_dims[0]), int(closest_size_dims[1])
            bucket_key = (target_h, target_w)
            
            if bucket_key not in self.buckets:
                self.buckets[bucket_key] = []
            self.buckets[bucket_key].append(idx)
        
        logger.info(f"Created {len(self.buckets)} resolution buckets.")
        if accelerator.is_main_process: # Print bucket sizes from main process
            for res_key, indices in self.buckets.items():
                logger.info(f"  Bucket {res_key}: {len(indices)} images")

    def __iter__(self):
        batches = []
        for resolution_key, indices in self.buckets.items():
            if self.shuffle:
                # Shuffle copy of indices, not in-place
                shuffled_indices = random.sample(indices, len(indices))
            else:
                shuffled_indices = indices
            
            for i in range(0, len(shuffled_indices), self.batch_size):
                batch = shuffled_indices[i:i+self.batch_size]
                if len(batch) == self.batch_size or (not self.drop_last and len(batch) > 0) :
                    batches.append(batch)
        
        if self.shuffle:
            random.shuffle(batches)
        
        return iter(batches)
    
    def __len__(self):
        num_batches = 0
        for indices in self.buckets.values():
            if self.drop_last:
                num_batches += len(indices) // self.batch_size
            else:
                num_batches += (len(indices) + self.batch_size - 1) // self.batch_size
        return num_batches


class DiffusionDataset(Dataset):
    """
    Dataset for loading images and captions for DiT fine-tuning.
    Supports aspect ratio bucketing if an aspect_ratio_map is provided.
    """
    def __init__(
        self,
        data_path,
        base_image_dir=None,
        image_column="image_path",
        caption_column="caption",
        random_flip=False,
        # Arguments for aspect ratio bucketing logic
        aspect_ratio_map: dict = None, 
        # Fallback arguments if aspect_ratio_map is None (old logic)
        resolution=512, 
        center_crop=True,
        keep_aspect_ratio=True, # Used by old logic
    ):
        self.base_image_dir = base_image_dir
        self.random_flip = random_flip
        self.aspect_ratio_map = aspect_ratio_map
        
        # Fallback parameters (used if aspect_ratio_map is None)
        self.fallback_resolution = resolution
        self.fallback_center_crop = center_crop
        self.fallback_keep_aspect_ratio = keep_aspect_ratio

        # Load data from CSV or directory
        self.data_entries = []
        
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            for _, row in df.iterrows():
                image_path = row[image_column]
                if base_image_dir is not None:
                    image_path = os.path.join(base_image_dir, image_path)
                
                caption = row[caption_column] if caption_column in df.columns else ""
                self.data_entries.append({
                    "image_path": image_path,
                    "caption": caption
                })
        else:
            image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
            for file in os.listdir(data_path):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_path = os.path.join(data_path, file)
                    caption = os.path.splitext(file)[0].replace('_', ' ').replace('-', ' ')
                    self.data_entries.append({
                        "image_path": image_path,
                        "caption": caption
                    })
                    
        logger.info(f"Loaded dataset with {len(self.data_entries)} entries")
        
        # Image transforms are now mostly dynamic in __getitem__ if using aspect_ratio_map
        if self.aspect_ratio_map is None:
            logger.info("No aspect_ratio_map provided. Using fallback image transformations.")
            self.static_transforms = self._get_fallback_transforms()
        else:
            logger.info("Using aspect_ratio_map for dynamic image transformations.")
            self.static_transforms = None # Transforms will be composed in __getitem__

    def _get_fallback_transforms(self):
        transform_list = []
        if self.fallback_resolution is not None:
            if self.fallback_keep_aspect_ratio:
                transform_list.append(transforms.Resize(
                    self.fallback_resolution, 
                    interpolation=transforms.InterpolationMode.BILINEAR
                ))
            else:
                transform_list.append(transforms.Resize(
                    (self.fallback_resolution, self.fallback_resolution), 
                    interpolation=transforms.InterpolationMode.BILINEAR
                ))
            if self.fallback_center_crop:
                transform_list.append(transforms.CenterCrop(self.fallback_resolution))
        
        if self.random_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # Output [-1, 1]
        ])
        return transforms.Compose(transform_list)
        
    def __len__(self):
        return len(self.data_entries)
    
    def __getitem__(self, index):
        entry = self.data_entries[index]
        
        image = Image.open(entry["image_path"]).convert("RGB")
        image = exif_transpose(image)
        
        if self.aspect_ratio_map:
            ori_w, ori_h = image.size
            
            closest_size_dims, _ = get_closest_ratio(float(ori_h), float(ori_w), self.aspect_ratio_map)
            target_h, target_w = int(closest_size_dims[0]), int(closest_size_dims[1])

            # Calculate resize dimensions (H, W)
            # Scale to ensure one dim matches target, then center crop.
            # This is similar to InternalData_ms.py logic
            scale_h = target_h / ori_h
            scale_w = target_w / ori_w

            if scale_h * ori_w > target_w: # equivalent to scale_h > scale_w if preserving aspect ratio
                # resize by height, then crop width
                resize_h = target_h
                resize_w = int(ori_w * scale_h)
            else:
                # resize by width, then crop height
                resize_w = target_w
                resize_h = int(ori_h * scale_w)
            
            # Ensure resize dimensions are at least 1
            resize_h = max(1, resize_h)
            resize_w = max(1, resize_w)

            current_transforms_list = [
                transforms.Resize((resize_h, resize_w), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop((target_h, target_w)),
            ]
            if self.random_flip:
                current_transforms_list.append(transforms.RandomHorizontalFlip())
            current_transforms_list.extend([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]) # Output [-1, 1]
            ])
            image_tensor = transforms.Compose(current_transforms_list)(image)
        else:
            # Use pre-computed static transforms (fallback)
            image_tensor = self.static_transforms(image)
            
        return (
            image_tensor,
            [{
                "long_caption": entry["caption"],
                # "target_hw": (target_h, target_w) # Could be useful for debugging
            }]
        )

def create_data_loader(
    data_path,
    batch_size,
    aspect_ratio_map: dict, # Added: map for bucketing logic
    base_image_dir=None,
    shuffle=True,
    num_workers=4,
    seed=None,
    # These are now mostly for fallback or if aspect_ratio_map is None
    resolution=None, 
    center_crop=False, 
    random_flip=False,
    use_resolution_buckets=True,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    dataset = DiffusionDataset(
        data_path=data_path,
        base_image_dir=base_image_dir,
        random_flip=random_flip,
        aspect_ratio_map=aspect_ratio_map if use_resolution_buckets or aspect_ratio_map else None, # Pass map if bucketing or if map is explicitly given for processing
        # Fallback parameters if aspect_ratio_map is not used/provided
        resolution=resolution, 
        center_crop=center_crop,
        # keep_aspect_ratio is True by default in DiffusionDataset if not using aspect_ratio_map
    )
    
    if use_resolution_buckets:
        if aspect_ratio_map is None:
            raise ValueError("use_resolution_buckets is True, but no aspect_ratio_map was provided (derived from args.resolution).")
        # Use the new ResolutionBucketSampler
        sampler = ResolutionBucketSampler(dataset, batch_size, aspect_ratio_map=aspect_ratio_map, shuffle=shuffle)
        data_loader = DataLoader(
            dataset,
            batch_sampler=sampler, # Important: use batch_sampler
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        # Standard sampler, assumes all images processed by DiffusionDataset will be compatible
        # (e.g. all square if aspect_ratio_map is None and resolution is set,
        # or varying sizes if aspect_ratio_map is used by dataset but not for batching)
        # WARNING: If aspect_ratio_map is used by DiffusionDataset without ResolutionBucketSampler,
        # batches may contain images of different sizes, which can break the training loop's assumption.
        # This configuration should be used with caution or further logic to handle mixed-size batches.
        if not aspect_ratio_map: # Only if not using aspect ratio map processing at all
            logger.info("Not using resolution buckets. Batches may contain images of different resolutions if aspect_ratio_map is still active in Dataset without this sampler.")
        
        if shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True, # Usually True for standard training
        )
    
    return data_loader

def encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
    return_index=-1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids

    prompt_embeds = text_encoder(
        text_input_ids.to(device), return_dict=True, output_hidden_states=True
    )

    prompt_embeds = prompt_embeds.hidden_states[return_index]
    if return_index != -1: # PixArt-alpha uses final_layer_norm, DiT might not
        final_layer_norm = getattr(text_encoder.encoder, "final_layer_norm", None)
        if final_layer_norm:
            prompt_embeds = final_layer_norm(prompt_embeds)
        dropout = getattr(text_encoder.encoder, "dropout", None)
        if dropout:
            prompt_embeds = dropout(prompt_embeds)


    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds

def forward(
    dit_model,
    batch,
    vae_model,
    text_encoder,
    tokenizer,
    device,
    global_step,
    master_process,
    generator=None,
    binnings=None,
    batch_multiplicity=None,
    bs_rampup=None,
    batch_size=None, # This is actual batch size after multiplicity/rampup
    return_index=-1,
):
    (images_vae, metadatas) = batch

    # Images from dataloader should now be in [-1, 1]
    # So, remove: images_vae.sub_(0.5).mul_(2.0)
    
    captions = [metadata["long_caption"][0] for metadata in metadatas] # Assuming batch format
    
    preprocess_start = time.time()
    # Ensure correct type and device, original normalization is now done in dataset
    images_vae = images_vae.to(device=device, dtype=torch.float32) 
    
    with torch.no_grad():
        vae_latent = vae_model.encode(images_vae).latent_dist.sample()
        vae_latent = (
            vae_latent - vae_model.config.shift_factor
        ) * vae_model.config.scaling_factor
        vae_latent = vae_latent.to(torch.bfloat16)
        
        caption_encoded = encode_prompt_with_t5(
            text_encoder,
            tokenizer,
            prompt=captions,
            device=device,
            return_index=return_index,
        )
        caption_encoded = caption_encoded.to(torch.bfloat16)
    
    if batch_multiplicity is not None:
        vae_latent = vae_latent.repeat(batch_multiplicity, 1, 1, 1)
        caption_encoded = caption_encoded.repeat(batch_multiplicity, 1, 1)
        
        do_zero_out = torch.rand(caption_encoded.shape[0], device=device) < 0.01
        caption_encoded[do_zero_out] = 0
    
    if bs_rampup is not None and global_step < bs_rampup:
        # This batch_size argument for rampup is the original args.train_batch_size
        target_bs = math.ceil((global_step + 1) * batch_size / bs_rampup / 4) * 4 
        if vae_latent.size(0) > target_bs:
            keep_indices = torch.randperm(vae_latent.size(0))[:target_bs]
            vae_latent = vae_latent[keep_indices]
            caption_encoded = caption_encoded[keep_indices]
    
    current_batch_size = vae_latent.size(0) # This is the effective batch size for this step
    
    image_token_size = vae_latent.shape[2] * vae_latent.shape[3]
    z = torch.randn(current_batch_size, device=device, dtype=torch.float32, generator=generator)
    alpha = 2 * math.sqrt(image_token_size / (64 * 64))
    
    do_uniform = torch.rand(current_batch_size, device=device, dtype=torch.float32, generator=generator) < 0.1
    uniform = torch.rand(current_batch_size, device=device, dtype=torch.float32, generator=generator)
    t_sig = torch.nn.Sigmoid()(z) # Renamed to avoid conflict with t below
    lognormal = t_sig * alpha / (1 + (alpha - 1) * t_sig)
    
    t = torch.where(~do_uniform, lognormal, uniform).to(torch.bfloat16) # Timestep
    
    noise = torch.randn(
        vae_latent.shape, device=device, dtype=torch.bfloat16, generator=generator
    )
    
    preprocess_time = time.time() - preprocess_start
    if master_process:
        logger.debug(f"Preprocessing took {preprocess_time*1000:.2f}ms, alpha={alpha:.2f}")
    
    forward_start = time.time()
    
    tr = t.reshape(current_batch_size, 1, 1, 1)
    z_t = vae_latent * (1 - tr) + noise * tr
    
    v_objective = vae_latent - noise
     
    output = dit_model(z_t, caption_encoded, t)
    
    # Assuming output and v_objective are in latent space (e.g., B C H W)
    # The rearrange might depend on specific model patchifying, adapt if needed
    # If model output is already tokenized (B L D), this rearrange is not needed / different
    # Based on the context of DiT and patchifying, this rearrange is plausible for loss calculation on tokens.
    # Ensure p1*p2 = patch_size^2 if applicable. For DiT, it's often H/patch_size, W/patch_size.
    # The original code uses p1=2, p2=2. This implies a patch size of 2x2 on the latent map.
    # This is specific to how PixArt-alpha structures its output.
    
    # If dit_model outputs B C H W (latent map)
    diffusion_loss_batchwise = (
        (v_objective.float() - output.float()).pow(2).mean(dim=(1, 2, 3)) # Mean over C, H, W
    )
    # If dit_model outputs B L D (sequence of patches)
    # targ = rearrange(v_objective, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size_h, p2=patch_size_w)
    # pred = output # Assuming output is already B L D
    # diffusion_loss_batchwise = ((targ.float() - pred.float()).pow(2).mean(dim=(1, 2)))

    # Sticking to original rearrange for now, assuming it matches model's output logic for loss
    # This specific rearrange is from PixArt-Alpha, which pools 2x2 regions of the latent.
    # Standard DiT might calculate loss directly on B C H W or on B L D if it returns tokens.
    # Let's assume the original code's rearrange was intentional for the specific DiT variant.
    patch_p1 = 2 
    patch_p2 = 2
    # Check if latent dimensions are divisible by patch_p1, patch_p2
    if v_objective.shape[2] % patch_p1 != 0 or v_objective.shape[3] % patch_p2 != 0:
        # This can happen if latent size is not a multiple of the rearrange patch size
        # Fallback to direct L2 loss on latent maps if rearrange is not possible
        logger.warning(f"Latent shape {v_objective.shape} not divisible by patch size ({patch_p1}, {patch_p2}) for rearrange. Using direct L2 loss on latent maps.")
        diffusion_loss_batchwise = (v_objective.float() - output.float()).pow(2).mean(dim=(1,2,3))
    else:
        targ = rearrange(v_objective, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_p1, p2=patch_p2)
        pred = rearrange(output, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_p1, p2=patch_p2)
        diffusion_loss_batchwise = (
            (targ.float() - pred.float()).pow(2).mean(dim=(1, 2))
        )

    diffusion_loss = diffusion_loss_batchwise.mean()  
    
    total_loss = diffusion_loss
    
    tbins = [min(int(_t.item() * 10), 9) for _t in t] # .item() for single tensor element
    if binnings is not None:
        (
            diffusion_loss_binning,
            diffusion_loss_binning_count,
        ) = binnings
        for idx, tb in enumerate(tbins):
            diffusion_loss_binning[tb] += diffusion_loss_batchwise[idx].item()
            diffusion_loss_binning_count[tb] += 1
    
    forward_time = time.time() - forward_start
    if master_process:
        logger.debug(f"Forward pass took {forward_time*1000:.2f}ms")
    
    return total_loss, diffusion_loss

def sample_images(
    dit_model,
    vae_model,
    text_encoder,
    tokenizer,
    device,
    global_step,
    prompts=None,
    image_width=512, # Default, should be consistent with training or user specified
    image_height=512,
    prompts_per_gpu=1, # This seems unused in current logic, batch_size=1 for sampling
    num_inference_steps=50,
    cfg_scale=6.0,
    return_index=-8,  # Default for PixArt text encoder features
    prompts_file=None,
): 
    if prompts_file is not None and os.path.exists(prompts_file):
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                file_prompts = [line.strip() for line in f.readlines() if line.strip()]
            if file_prompts:
                logger.info(f"Using {len(file_prompts)} prompts from {prompts_file}")
                prompts = file_prompts
            else:
                logger.warning(f"Prompt file {prompts_file} is empty, using default prompts")
        except Exception as e:
            logger.error(f"Error reading prompts file: {e}. Using default prompts.")
    
    if prompts is None:
        prompts = [
            "a beautiful photograph of a mountain landscape at sunset",
            "a cute cat playing with a ball of yarn",
            "a futuristic cityscape with flying cars",
            "an oil painting of a flower garden",
        ]
    
    logger.info(f"Generating {len(prompts)} sample images at step {global_step} with HxW: {image_height}x{image_width}")
    
    previous_training_state = dit_model.training
    dit_model.eval()
    
    samples_dir = os.path.join(args.output_dir, "samples") # Use args.output_dir
    os.makedirs(samples_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, prompt in enumerate(tqdm(prompts, desc="Generating samples")):
            prompt_embeds = encode_prompt_with_t5(
                text_encoder,
                tokenizer,
                prompt=[prompt],
                device=device,
                return_index=return_index, # Use the passed return_index
            ).to(torch.bfloat16)
            
            negative_embeds = torch.zeros_like(prompt_embeds) # Simple zero uncond
            
            batch_size = 1 # Sample one image at a time
            # Use a consistent seed for each image but varied by global_step and prompt index
            generator = torch.Generator(device=device).manual_seed(args.seed + global_step + i if args.seed else global_step + i)
            
            # VAE downsampling factor is typically 8
            vae_scale_factor = 8 
            latent_height = image_height // vae_scale_factor
            latent_width = image_width // vae_scale_factor
            
            # Latent channels for DiT/PixArt style models are often 16 (e.g. VAE output C=4, then patched to 16 for DiT)
            # However, standard Diffusers VAE output C=4. FLite might use a VAE with C=4.
            # The model (dit_model) input shape will determine this.
            # If vae_model.config.latent_channels is available, use it. Otherwise, assume 4.
            latent_channels = getattr(vae_model.config, "latent_channels", 4)

            latent_shape = (batch_size, latent_channels, latent_height, latent_width)
            latents = torch.randn(latent_shape, device=device, generator=generator, dtype=torch.float32) # Start with float32
            
            image_token_size = latent_height * latent_width
            alpha = 2 * math.sqrt(image_token_size / (64 * 64)) # Standard PixArt alpha
            
            for j_idx, j_val in enumerate(tqdm(range(num_inference_steps, 0, -1), desc="Sampling steps", leave=False)):
                t_val = j_val / num_inference_steps
                t_next_val = (j_val - 1) / num_inference_steps
                
                t_val_adj = t_val * alpha / (1 + (alpha - 1) * t_val)   
                t_next_val_adj = t_next_val * alpha / (1 + (alpha - 1) * t_next_val)
                dt = t_val_adj - t_next_val_adj
                
                t_tensor = torch.tensor([t_val_adj] * batch_size, device=device, dtype=torch.bfloat16)
                
                # DiT model expects bfloat16 latents
                model_input_latents = latents.to(torch.bfloat16)
                
                model_output_uncond = dit_model(model_input_latents, negative_embeds, t_tensor)
                model_output_cond = dit_model(model_input_latents, prompt_embeds, t_tensor)
                
                model_output = model_output_uncond + cfg_scale * (model_output_cond - model_output_uncond)
                
                # Euler-like step for velocity model (v-prediction)
                # x_t = x_{t-dt} * (1 - dt/sigma_t) - v * dt  (approx)
                # For v-objective: x_prev = x_t * (1-dt) - v_pred * dt (if v_pred is (x0 - noise))
                # or more simply: x_prev = x_t - v_pred * dt (if v_pred is velocity d(xt)/dt)
                # The provided forward pass calculates v_objective = vae_latent - noise.
                # If model predicts this v_objective:
                # x_t = alpha_t * x0 + sigma_t * noise
                # v = x0 - noise = x0 - (x_t - alpha_t * x0) / sigma_t
                # This is complex. Let's assume simple Euler step for velocity for now:
                # x_prev = x_t + v_pred * dt (if dt is negative, i.e. t_next - t)
                # Here dt = t - t_next, so it's positive.
                # Latent update for velocity models (v-prediction, where v = alpha*noise - sigma*x0 for EDM-style)
                # Or if v_pred is x0 - noise (as in PixArt paper Eq.4 where model predicts v):
                # x_t-1 = x_t * (1-dt) + v_pred * dt
                # (This assumes model output is v_objective = x0 - noise)
                # Convert model_output to float32 for accumulation to prevent precision issues
                latents = latents * (1 - dt) + model_output.to(dtype=torch.float32) * dt

            # Decode latents
            # Undo VAE scaling for FLite
            latents = (latents / vae_model.config.scaling_factor) + vae_model.config.shift_factor
            image = vae_model.decode(latents.to(torch.float32)).sample # VAE decode often wants float32
            
            image = (image / 2 + 0.5).clamp(0, 1) # Denormalize from [-1, 1] to [0, 1]
            image = (image * 255).round().to(torch.uint8) # To [0, 255] and uint8
            image = image.permute(0, 2, 3, 1).cpu().numpy()[0] # BCHW -> BHWC -> HWC
            
            pil_image = Image.fromarray(image)
            prompt_slug = "".join(c if c.isalnum() or c in " _-" else "" for c in prompt[:50]).strip()
            image_path = os.path.join(samples_dir, f"sample_gs{global_step}_idx{i}_{prompt_slug}.png")
            pil_image.save(image_path)
             
    dit_model.train(previous_training_state)
    logger.info(f"Generated samples saved to {samples_dir}")


def train(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir,
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=project_config,
    )
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    if accelerator.is_main_process:
        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.WARNING)
    
    logger.info(accelerator.state, main_process_only=False)
    device = accelerator.device
    
    if args.seed is not None:
        set_seed(args.seed)
        common_seed = args.seed
    else:
        common_seed = random.randint(0, 1000000)
        set_seed(common_seed) # Set seed even if not provided by user for reproducibility of this run
    
    logger.info(f"Using random seed: {common_seed}")
    
    # Aspect Ratio Map Selection
    aspect_ratio_map = None
    if args.use_resolution_buckets:
        if args.resolution is None:
            logger.error("If use_resolution_buckets is True, --resolution (e.g., 256, 512, 1024) must be provided to select an aspect ratio map.")
            exit(1)
        aspect_ratio_map = get_aspect_ratio_map(args.resolution)
        if aspect_ratio_map is None:
            logger.error(f"Could not find aspect ratio map for resolution {args.resolution}. Exiting.")
            exit(1)
        logger.info(f"Using aspect ratio map for base resolution: {args.resolution}")
    elif args.resolution in [256, 512, 1024]: # If resolution is a base size, use AR map for processing even if not bucketing by sampler
        aspect_ratio_map = get_aspect_ratio_map(args.resolution)
        logger.info(f"Using aspect ratio map for base resolution {args.resolution} for image processing (sampler not bucketing by AR).")


    if accelerator.is_main_process and args.report_to in ["wandb", "all"]:
        import wandb
        run_name = args.run_name if args.run_name else f"DiT-finetune-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        wandb.init(project=args.project_name, name=run_name, config=vars(args))
     
    logger.info(f"Loading model from {args.pretrained_model_path}")
    pipeline = FLitePipeline.from_pretrained(
        args.pretrained_model_path,
        torch_dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32 # Ensure bfloat16 if specified
    )

    if args.use_precomputed_data:
        dit_model = pipeline.dit_model
        # Ensure model is on device and in correct dtype, especially if mixed precision is used
        dit_model = dit_model.to(device=device, dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32)
        vae_model, text_encoder, tokenizer = None, None, None
        logger.info("Using precomputed data - VAE and text encoder not loaded.")
    else:
        dit_model = pipeline.dit_model.to(device=device, dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32)
        vae_model = pipeline.vae.to(device=device) # VAE typically in float32 for precision
        text_encoder = pipeline.text_encoder.to(device=device, dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32)
        tokenizer = pipeline.tokenizer
        vae_model.requires_grad_(False)
        text_encoder.requires_grad_(False)
    
    dit_model.train() 
    param_count = sum(p.numel() for p in dit_model.parameters())
    logger.info(f"Number of parameters: {param_count / 1e6:.2f} million")

    if args.use_lora:
        logger.info("Setting up LoRA fine-tuning")
        if args.train_only_lora:
            dit_model.requires_grad_(False)
            logger.info("Freezing base model parameters, training only LoRA weights")
        target_modules = [module.strip() for module in args.lora_target_modules.split(",")]
        lora_config = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha, target_modules=target_modules,
            lora_dropout=args.lora_dropout, bias="none", init_lora_weights="gaussian"
        )
        dit_model.add_adapter(lora_config)
        for name, param in dit_model.named_parameters():
            if "lora" in name.lower(): param.requires_grad = True
        if args.lora_checkpoint:
            logger.info(f"Loading LoRA weights from {args.lora_checkpoint}")
            lora_state_dict = torch.load(args.lora_checkpoint, map_location=device)
            set_peft_model_state_dict(dit_model, lora_state_dict)
        lora_param_count = sum(p.numel() for n, p in dit_model.named_parameters() if "lora" in n.lower() and p.requires_grad)
        trainable_params = sum(p.numel() for p in dit_model.parameters() if p.requires_grad)
        logger.info(f"Added LoRA adapter. Rank: {args.lora_rank}, Target modules: {args.lora_target_modules}")
        logger.info(f"LoRA params: {lora_param_count/1e6:.2f}M. Trainable params: {trainable_params/1e6:.2f}M")
    
    if args.use_precomputed_data:
        if not args.precomputed_data_dir:
            raise ValueError("precomputed_data_dir must be specified when using precomputed data.")
        logger.info(f"Using precomputed data from {args.precomputed_data_dir}")
        # Note: create_precomputed_data_loader might also need to know about aspect_ratio_map
        # if precomputed latents are stored with varied resolutions and need bucketing.
        # This depends on its internal implementation.
        train_dataloader = create_precomputed_data_loader(
            precomputed_data_dir=args.precomputed_data_dir, batch_size=args.train_batch_size,
            shuffle=True, num_workers=4, random_flip=args.random_flip,
            use_resolution_buckets=args.use_resolution_buckets # Pass this along
        )
        val_dataloader = None # Simplified, assuming precomputed val data handling is separate or not primary focus
    else:
        if not args.train_data_path:
             raise ValueError("train_data_path must be specified if not using precomputed data.")
        train_dataloader = create_data_loader(
            data_path=args.train_data_path, batch_size=args.train_batch_size,
            aspect_ratio_map=aspect_ratio_map, # Pass the selected map
            base_image_dir=args.base_image_dir, shuffle=True, num_workers=4, seed=common_seed,
            resolution=args.resolution, center_crop=args.center_crop, random_flip=args.random_flip,
            use_resolution_buckets=args.use_resolution_buckets
        )
        val_dataloader = None
        if args.val_data_path:
            val_dataloader = create_data_loader(
                data_path=args.val_data_path, batch_size=args.eval_batch_size,
                aspect_ratio_map=aspect_ratio_map, # Also use for validation processing
                base_image_dir=args.base_image_dir, shuffle=False, num_workers=4,
                resolution=args.resolution, center_crop=args.center_crop, random_flip=False, # No random flip for val
                use_resolution_buckets=args.use_resolution_buckets # Bucketing for val if enabled
            )
    
    optimizer_class = torch.optim.AdamW
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
            logger.info("Using 8-bit AdamW optimizer.")
        except ImportError:
            logger.warning("bitsandbytes not installed, falling back to regular AdamW.")
    
    optimizer = optimizer_class(
        filter(lambda p: p.requires_grad, dit_model.parameters()), # Important: only pass trainable parameters
        lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    
    if args.max_steps:
        max_steps = args.max_steps
    else:
        # Ensure dataloader length is valid before calculating max_steps
        try:
            len_train_dataloader = len(train_dataloader)
        except TypeError: # Happens if dataloader has no __len__ (e.g. iterable dataset not wrapped)
            # Estimate steps if length is not available, or require max_steps
            if args.num_epochs > 1: # For safety, require max_steps if dataloader len is unknown for multi-epoch
                 raise ValueError("Dataloader has no length. Please specify --max_steps for multi-epoch training.")
            logger.warning("Dataloader has no __len__. Max_steps calculation might be an estimate or require --max_steps.")
            # A large number if num_epochs is 1, assuming user will stop manually or via other means.
            # Or, if possible, iterate once to count. For now, rely on user setting max_steps if __len__ is missing.
            # This shouldn't be an issue with the provided samplers as they implement __len__.
            len_train_dataloader = 1000000 # Fallback large number if __len__ is not there
        max_steps = len_train_dataloader * args.num_epochs // args.gradient_accumulation_steps

    
    if args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, args.num_warmup_steps, max_steps)
    elif args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, max_steps)
    else:
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, max_steps * 1000)
    
    # Note: if using LoRA, optimizer might need to be re-initialized AFTER accelerator.prepare(dit_model)
    # if only LoRA params are trainable and model was not prepared that way.
    # However, filter(lambda p: p.requires_grad, dit_model.parameters()) should handle this.
    dit_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        dit_model, optimizer, train_dataloader, lr_scheduler
    )
    if val_dataloader:
        val_dataloader = accelerator.prepare(val_dataloader)
    
    global_step = 0
    resume_step = 0
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        if checkpoint_path == "latest":
            # Simplified: look for step-based checkpoint folders
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.output_dir, d))]
            if dirs:
                dirs.sort(key=lambda d: int(d.split("-")[1]))
                checkpoint_path = os.path.join(args.output_dir, dirs[-1])
            else: checkpoint_path = None
        
        if checkpoint_path and os.path.exists(checkpoint_path): # Check if it's a directory for accelerator state
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            accelerator.load_state(checkpoint_path)
            try:
                global_step = int(checkpoint_path.split("-")[-1])
                resume_step = global_step
                logger.info(f"Resumed from step {global_step}")
            except ValueError:
                logger.warning(f"Could not parse step from checkpoint path {checkpoint_path}. Starting step count from 0 after resume.")
                # global_step will be implicitly handled by accelerator if it tracks it.
        else:
            logger.info(f"Checkpoint {args.resume_from_checkpoint} not found. Starting from scratch.")
    
    progress_bar = tqdm(range(global_step, max_steps), disable=not accelerator.is_main_process, desc="Training")
    
    diffusion_loss_binning = {k: 0 for k in range(10)}
    diffusion_loss_binning_count = {k: 0 for k in range(10)}
    
    dit_model.train()

    logger.info(f"Dataset size: {len(train_dataloader.dataset)} images")
    logger.info(f"Dataloader batches per epoch: {len(train_dataloader)}")
    logger.info(f"Max training steps: {max_steps}")

    # Determine sampling dimensions (use base resolution or a default like 1024)
    # If aspect ratio map is used, could sample at 1:1 ratio of the base size
    sample_res_h, sample_res_w = args.resolution, args.resolution
    if args.resolution is None : # Default if no resolution provided for training
        sample_res_h, sample_res_w = 1024, 1024 # Fallback sampling resolution
    if aspect_ratio_map and '1.0' in aspect_ratio_map: # If 1:1 exists in map, use it
        square_dims = aspect_ratio_map['1.0']
        sample_res_h, sample_res_w = int(square_dims[0]), int(square_dims[1])
    logger.info(f"Sample images will be generated at {sample_res_h}x{sample_res_w}")


    for epoch in range(args.num_epochs):
        logger.info(f"Starting epoch {epoch+1}/{args.num_epochs}")
        if hasattr(train_dataloader.sampler, 'set_epoch'): # For DistributedSampler
            train_dataloader.sampler.set_epoch(epoch)
        
        epoch_start_time = time.time()
        
        for step, batch in enumerate(train_dataloader):
            # Global step management if resuming
            if global_step < resume_step: # If accelerator.load_state handled step, this might not be needed
                if (step + epoch * len(train_dataloader)) < resume_step : # Rough check
                     # This part of logic depends on how accelerator handles resume_from_checkpoint step counting
                    if accelerator.sync_gradients: # only update progress bar if a real step would have occurred
                        progress_bar.update(1)
                    continue 
            
            with accelerator.accumulate(dit_model):
                if args.use_precomputed_data:
                    total_loss, diffusion_loss = forward_with_precomputed_data(
                        dit_model=dit_model, batch=batch, device=device, global_step=global_step,
                        master_process=accelerator.is_main_process,
                        binnings=(diffusion_loss_binning, diffusion_loss_binning_count),
                        batch_multiplicity=args.batch_multiplicity,
                        bs_rampup=None, # Accelerator handles grad accumulation
                        batch_size=args.train_batch_size, # Original batch size
                    )
                else:
                    total_loss, diffusion_loss = forward(
                        dit_model=dit_model, batch=batch, vae_model=vae_model, text_encoder=text_encoder,
                        tokenizer=tokenizer, device=device, global_step=global_step,
                        master_process=accelerator.is_main_process,
                        binnings=(diffusion_loss_binning, diffusion_loss_binning_count),
                        batch_multiplicity=args.batch_multiplicity,
                        bs_rampup=None, 
                        batch_size=args.train_batch_size, # Original batch size
                        return_index=-8, # Default for PixArt from example
                    )
                
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(filter(lambda p: p.requires_grad, dit_model.parameters()), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if accelerator.is_main_process:
                    logs = {"train/loss": total_loss.item(), "train/diffusion_loss": diffusion_loss.item(),
                            "train/lr": lr_scheduler.get_last_lr()[0], "train/epoch": epoch, "train/step": global_step}
                    accelerator.log(logs, step=global_step)
                    save_log_line(logs, f"{args.train_batch_size}bs_{args.resolution}px_{args.learning_rate}lr.jsonl")

                    progress_bar.set_postfix({"loss": f"{total_loss.item():.4f}", "lr": f"{lr_scheduler.get_last_lr()[0]:.7f}"})
                
                if global_step % (4 * args.sample_every // args.train_batch_size) == 0 and accelerator.is_main_process:
                    # Sampling logic
                    temp_vae, temp_text_encoder, temp_tokenizer = vae_model, text_encoder, tokenizer
                    if args.use_precomputed_data: # Need to load them if not available
                        logger.info("Temporarily loading VAE and text encoder for sampling (as using precomputed data)")
                        # Assuming pipeline was light due to precomputed, load full for sampling
                        sampling_pipeline = FLitePipeline.from_pretrained(args.pretrained_model_path, torch_dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32)
                        temp_vae = sampling_pipeline.vae.to(device)
                        temp_text_encoder = sampling_pipeline.text_encoder.to(device, dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32)
                        temp_tokenizer = sampling_pipeline.tokenizer
                    
                    sample_images(
                        dit_model=accelerator.unwrap_model(dit_model), vae_model=temp_vae,
                        text_encoder=temp_text_encoder, tokenizer=temp_tokenizer, device=device,
                        global_step=global_step, prompts=None, # Defaults will be used
                        image_width=sample_res_w, image_height=sample_res_h,
                        prompts_file=args.sample_prompts_file, return_index=-8, # Consistent return_index
                    )
                    if args.use_precomputed_data: # Clean up if loaded temporarily
                        del temp_vae, temp_text_encoder, temp_tokenizer, sampling_pipeline
                        torch.cuda.empty_cache()

                if val_dataloader and global_step % args.eval_every == 0:
                    # Eval logic (ensure VAE/TextEncoder are available if not precomputed)
                    pass # Placeholder for original eval logic
            
            if global_step >= max_steps:
                logger.info(f"Reached max steps ({max_steps}). Stopping training.")
                break
        
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds (Global step: {global_step})")
        
        # Save checkpoint (accelerator handles model unwrapping etc.)
        if accelerator.is_main_process and args.checkpointing_steps > 0 : # Also save per epoch if desired or based on steps
             if (epoch + 1) % (args.checkpointing_steps // len(train_dataloader) if len(train_dataloader) > 0 else args.checkpointing_steps) == 0 or global_step % args.checkpointing_steps == 0 : # Simplified logic for epoch end save + step based save
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path) # Saves model, optimizer, scheduler, sampler states
                logger.info(f"Saved checkpoint to {save_path} at step {global_step}")


        if global_step >= max_steps: break
    
    logger.info(f"Training completed after {global_step} steps!")
    if accelerator.is_main_process:
        final_save_path = os.path.join(args.output_dir, "final_model_state")
        accelerator.save_state(final_save_path) # Save full training state
        logger.info(f"Final training state saved to {final_save_path}")

        # Save unwrapped model for easier inference (optional, if pipeline expects raw state_dict)
        unwrapped_model = accelerator.unwrap_model(dit_model)
        torch.save(unwrapped_model.state_dict(), os.path.join(args.output_dir,"final_model_weights.pt"))
        logger.info(f"Final model weights saved to {os.path.join(args.output_dir, 'final_model_weights.pt')}")

        if args.use_lora:
            lora_state_dict = get_peft_model_state_dict(unwrapped_model)
            torch.save(lora_state_dict, os.path.join(args.output_dir, "final_lora_weights.pt"))
            logger.info(f"Saved final LoRA weights to {os.path.join(args.output_dir, 'final_lora_weights.pt')}")
        
        if args.report_to in ["wandb", "all"] and wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    # Perform checks for args consistency
    if args.use_resolution_buckets and args.resolution is None:
        print("ERROR: --resolution must be set (e.g., 256, 512, 1024) when --use_resolution_buckets is enabled.")
        exit(1)
    if args.use_resolution_buckets and args.resolution not in [256, 512, 1024]:
        print(f"ERROR: --resolution ({args.resolution}) is not a supported base size for bucketing. Use 256, 512, or 1024.")
        exit(1)
    
    # If not using resolution buckets but a resolution for AR map processing is given,
    # it implies that batches might contain different resolutions if a standard sampler is used.
    # This is generally okay if the model handles it, but can be inefficient.
    # The current forward pass expects same-resolution latents in a batch.
    # ResolutionBucketSampler ensures this. If not using it, ensure DiffusionDataset + standard sampler
    # produces batches of same-resolution images (e.g., by not using aspect_ratio_map in Dataset,
    # or by ensuring the model/forward pass can handle mixed resolutions, which is not the case here).
    # For safety, if an aspect_ratio_map is active (due to args.resolution), strongly recommend use_resolution_buckets.
    if args.resolution in [256, 512, 1024] and not args.use_resolution_buckets:
        logger.warning(f"args.resolution ({args.resolution}) implies aspect ratio processing in Dataset, "
                       "but --use_resolution_buckets is False. Batches from standard sampler might "
                       "contain mixed resolutions if not all images map to the same bucket. "
                       "This can cause issues or inefficiencies if not handled by the model's forward pass. "
                       "It's recommended to use --use_resolution_buckets with these resolutions.")


    train(args)
