# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import json
import h5py
import os.path as osp
from types import FunctionType
import deepspeed

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from sympy import im

from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import MAP_FUNC
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          StopWordStoppingCriteria)
import torch
from xtuner.model.utils import LoadWoInit, prepare_inputs_labels_for_multimodal
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
from xtuner.utils import PROMPT_TEMPLATE
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import numpy as np
from transformers import GenerationConfig, StoppingCriteriaList

import os

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

def parse_args():
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--test_json_path', default=None, help='Path to the test JSON file.')
    parser.add_argument('--question_embedding_path', default=None, help='Path to the question embeddings .pt file.')
    parser.add_argument('--output_log_json', default=None, help='Path to save detailed JSON log for analysis.')

    parser.add_argument(
    '--torch-dtype',
    default='bf16',
    choices=TORCH_DTYPE_MAP.keys(),
    help='Override the default `torch.dtype` and load the model under '
    'a specific `dtype`.')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def register_function(cfg_dict):
    if isinstance(cfg_dict, dict):
        for key, value in dict.items(cfg_dict):
            if isinstance(value, FunctionType):
                value_str = str(value)
                if value_str not in MAP_FUNC:
                    MAP_FUNC.register_module(module=value, name=value_str)
                cfg_dict[key] = value_str
            else:
                register_function(value)
    elif isinstance(cfg_dict, (list, tuple)):
        for value in cfg_dict:
            register_function(value)

def main():
    torch.cuda.set_device(0)
    args = parse_args()

    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    register_function(cfg._cfg_dict)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)
        
    runner.model.to(TORCH_DTYPE_MAP[args.torch_dtype])
    model_kwargs = {
        'trust_remote_code': True,
        'torch_dtype': TORCH_DTYPE_MAP[args.torch_dtype]
    }
    state_dict = guess_load_checkpoint(args.checkpoint)
    runner.model.load_state_dict(state_dict, strict=False)
    runner.model.eval()
    runner.model.to(model_kwargs['torch_dtype'])
    runner.logger.info(f'Load checkpoint from {args.checkpoint}')

    llm_name_or_path = 'Qwen/Qwen2.5-7B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(
                llm_name_or_path,
                trust_remote_code=True,
                encode_special_tokens=True)

    # Access the underlying model directly (user already provided this)
    llm = runner.model.llm
    LongNet_encoder = runner.model.LongNet_encoder
    projector = runner.model.projector
    group_selector = runner.model.group_selector
    patch_selector = runner.model.patch_selector
    question_prior_proj = runner.model.question_prior_proj
    
    IDX_TO_CLASS_MAP = {
        0: "malignant tissue", 1: "benign tissue", 2: "stroma", 3: "lymphocytes",
        4: "necrosis", 5: "adipose tissue", 6: "tissue artifact", 7: "blood vessel",
        8: "extracellular mucin", 9: "nerve", 10: "hemorrhage", 11: "smooth muscle", 12: "plasma cells"
    }
    CLASS_TO_IDX_MAP = {v: k for k, v in IDX_TO_CLASS_MAP.items()}
    
    if args.question_embedding_path:
        question_embeddings = torch.load(args.question_embedding_path, map_location='cpu')
    else:
        question_embeddings = None
    
    with open(args.test_json_path, 'r') as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} samples from {args.test_json_path}")
    
    closed_results = []
    open_results = []

    for i, item in enumerate(test_data):
        print('*'*30)
        
        image_path = item['image']
        # tumor_name becomes "TCGA-LUSC"
        tumor_name = os.path.dirname(image_path).replace('TCGA-', '')
        # base_filename becomes "TCGA-21-1083-01Z-00-DX1.pt"
        base_filename = os.path.basename(image_path)
        # case_name becomes "TCGA-21-1083-01Z-00-DX1"
        case_name = os.path.splitext(base_filename)[0]
        
        print(f"Processing ID: {i}, Slide: {case_name}, Tumor: {tumor_name}")
        
        compression_rate = 1.0
        sampler_selection_rate = 1.0
        feature_root = './data/features/conch_v1'
        weak_label_root = './data/weak_labels'
        test_image_file = f'{feature_root}/{tumor_name.lower()}_224x224_b20_t15/pt_files/{case_name}.pt'
        weak_label_file = f'{weak_label_root}/{tumor_name.lower()}/h5_files/{case_name}.h5'
        
        if not os.path.exists(test_image_file) or not os.path.exists(weak_label_file):
            print(f"Skipping slide {case_name} due to missing feature or label file.")
            with open("missing_WSI_log.txt", "a") as f:
                f.write(f"Image: {test_image_file}, WeakLabel: {weak_label_file}\n")
            continue
        
        with h5py.File(weak_label_file, 'r') as h5_label_file:
            label_list = [h5_label_file[key][()] for key in h5_label_file.keys()]
        
        byte_labels = label_list[1]
        string_labels = [label.decode('utf-8') for label in byte_labels]
        weak_label_indices = [CLASS_TO_IDX_MAP[label] for label in string_labels]
        weak_label_numpy = np.array(weak_label_indices, dtype=np.int64)
            
        image_cpu = torch.load(test_image_file, map_location='cpu')
        image = image_cpu.numpy()
        total_rows_before_sampling = image.shape[0]
        
        sample_num = 10000
        if total_rows_before_sampling >= sample_num:
            compression_rate = sample_num / total_rows_before_sampling
            indices = np.linspace(0, total_rows_before_sampling - 1, sample_num, dtype=int)
            image = image[indices]
            weak_label_numpy = weak_label_numpy[indices]
        
        image = torch.from_numpy(image.reshape(1, -1, 512)).to(TORCH_DTYPE_MAP[args.torch_dtype])
        weak_label = torch.from_numpy(weak_label_numpy).long().unsqueeze(0).cuda()
        image = image.cuda()

        prompt_template = PROMPT_TEMPLATE.qwen_chat
        SYSTEM = ''
        full_input_text = item['conversations'][0]['value']
        sample_input = full_input_text.replace('<image>\n', '', 1)
        ground_truth_answer = item['conversations'][1]['value']
        print('Input: ', sample_input)

        instruction = prompt_template.get('INSTRUCTION', '{input}')
        sample_input_with_token = DEFAULT_IMAGE_TOKEN + '\n' + sample_input
        inputs = (SYSTEM + instruction).format(input=sample_input_with_token, round=1, **runner.cfg)
        chunk_encode = []
        for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
            cur_encode = tokenizer.encode(chunk, add_special_tokens=(idx==0))
            chunk_encode.append(cur_encode)
        input_ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            input_ids.extend(cur_chunk_encode)
            if idx != len(chunk_encode) - 1:
                input_ids.append(IMAGE_TOKEN_INDEX)
        input_ids = torch.tensor(input_ids).cuda()
        
        with torch.no_grad():
            image = image.to(projector.dtype)
            image = LongNet_encoder(src_tokens=None, token_embeddings=image.permute(1, 0, 2))["encoder_out"]
            image = image.permute(1, 0, 2)
            pixel_values = projector(image)
        
        log_entry = {
            "slide_id": case_name,
            "question_id": item['id'],
            "question": sample_input,
        }
        
        # Use the correct variable name 'group_selector'
        if group_selector and question_embeddings is not None:
            q_embedding_key = item['id']
            current_q_embedding = question_embeddings.get(q_embedding_key)
                
            if current_q_embedding is not None:
                current_q_embedding = current_q_embedding.to(image.device, dtype=image.dtype)
                
                with torch.no_grad():
                    B, N, D_feat = pixel_values.shape
                    # Access num_groups directly from runner.model
                    K = runner.model.num_groups
                    
                    group_prototypes_sum = torch.zeros(B, K, D_feat, device=pixel_values.device, dtype=torch.float32)
                    group_counts = torch.zeros(B, K, device=pixel_values.device, dtype=torch.float32)
                    group_indices_map = weak_label.unsqueeze(-1).expand(-1, -1, D_feat)
                    group_prototypes_sum.scatter_add_(1, group_indices_map, pixel_values.float())
                    group_counts.scatter_add_(1, weak_label, torch.ones_like(pixel_values[..., 0], dtype=torch.float32))
                    group_prototypes = (group_prototypes_sum / (group_counts.unsqueeze(-1) + 1e-7)).to(image.dtype)
                    
                    group_mask = (group_counts == 0)
                    attention_mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=group_mask.device), group_mask], dim=1)
                    
                    # This entire block is replaced with the new hierarchical logic
                    
                    # 1. Project question embedding first
                    projected_q_emb = question_prior_proj(current_q_embedding.unsqueeze(0).to(image.dtype))
                    
                    # 2. Get group selection probabilities
                    group_selection_probs, group_logits = group_selector(
                        group_prototypes,
                        projected_q_emb, # Use projected_q_emb
                        attention_mask=attention_mask
                    )

                    # 3. Calculate target_prior for logging (same as in forward)
                    q_emb_norm = F.normalize(projected_q_emb, p=2, dim=-1).unsqueeze(1)
                    proto_norm = F.normalize(group_prototypes, p=2, dim=-1)
                    prior_scores = torch.bmm(q_emb_norm.float(), proto_norm.transpose(1, 2).float()).squeeze(1)
                    
                    # Access logit_scale, bias, and temp directly from runner.model
                    scaled_scores = prior_scores * runner.model.logit_scale.exp()
                    target_prior = torch.sigmoid((scaled_scores + runner.model.group_prior_bias) / runner.model.group_prior_temperature)
                    
                    final_selected_indices = []
                    actual_samples_per_group = {}

                    for group_idx in range(K):
                        group_count_i = group_counts[0, group_idx]
                        if group_count_i > 0:
                            # Get group probability
                            group_prob_i = group_selection_probs[0, group_idx]
                            
                            # Get patches for this group
                            original_indices_in_group = torch.where(weak_label.squeeze(0) == group_idx)[0]
                            patches_in_group = pixel_values.squeeze(0)[original_indices_in_group]
                            
                            # Get patch-level selection probabilities
                            patch_selection_probs_i, _ = patch_selector(
                                patches_in_group.to(image.dtype), 
                                projected_q_emb.to(image.dtype)
                            )
                            
                            # Determine K_i (number of patches to select from this group)
                            K_i_float = group_prob_i.detach() * group_count_i.detach()
                            # ceil (paper Eq. 5): every non-empty group keeps >= 1 patch
                            K_i = torch.ceil(K_i_float).int().item()
                            K_i = min(max(0, K_i), int(group_count_i.item()))
                            
                            actual_samples_per_group[group_idx] = K_i # Store for logging

                            if K_i > 0:
                                # Select top-k patches based on patch_selector scores
                                # squeeze patch_selection_probs_i in case it has an extra dim
                                _, selected_indices_in_group = torch.topk(patch_selection_probs_i.view(-1), k=K_i)
                                
                                # Map relative indices back to original indices
                                original_indices_sampled = original_indices_in_group[selected_indices_in_group]
                                final_selected_indices.append(original_indices_sampled)


                    original_token_count = pixel_values.shape[1]
                    if final_selected_indices:
                        all_selected_indices = torch.cat(final_selected_indices)
                        # match training token order: ascending patch index (training uses torch.nonzero)
                        all_selected_indices = torch.sort(all_selected_indices)[0]
                        pixel_values = pixel_values.squeeze(0)[all_selected_indices].unsqueeze(0)
                    else:
                        pixel_values = torch.zeros(B, 0, D_feat, device=pixel_values.device, dtype=pixel_values.dtype)
                    
                    
                    selected_token_count = pixel_values.shape[1]
                    if original_token_count > 0:
                        sampler_selection_rate = selected_token_count / original_token_count
                    
                    # Handle case where no patches are selected
                    if selected_token_count == 0:
                        print("No patches selected by the hierarchical selector. Using a zero vector.")
                        pixel_values = torch.zeros(B, 1, D_feat, device=pixel_values.device, dtype=pixel_values.dtype)
                        selected_token_count = 1 # Set to 1 to avoid division by zero if logic depends on it
                        
                    print(f"Selector reduced tokens from {original_token_count} to {pixel_values.shape[1]}")
                    
                    log_entry["patch_counts"] = {
                        "total_before_downsampling": int(total_rows_before_sampling),
                        "total_after_downsampling": original_token_count,
                        "total_after_selection": pixel_values.shape[1],
                    }
                    log_entry["sampler_selection_rate"] = sampler_selection_rate

            else:
                print("Could not find a corresponding Embedding.")
                
        # Access llm directly from runner.model
        mm_inputs = prepare_inputs_labels_for_multimodal(
            llm=runner.model.llm,
            input_ids=input_ids.unsqueeze(0),
            pixel_values=pixel_values)
        
        max_new_tokens=500
        gen_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        stop_criteria = StoppingCriteriaList()
        for word in prompt_template.get('STOP_WORDS', []):
            stop_criteria.append(StopWordStoppingCriteria(tokenizer, word))
        
        generate_output = llm.generate(**mm_inputs, generation_config=gen_config, streamer=None, bos_token_id=tokenizer.bos_token_id, stopping_criteria=stop_criteria)
        generation_output = tokenizer.decode(generate_output[0])
        if generation_output.endswith('<|im_end|>'):
            generation_output = generation_output[:-10]
        print('Output: ', generation_output)

        log_entry['Answer'] = ground_truth_answer
        log_entry['Output'] = generation_output
        
        if ground_truth_answer.strip().upper() in ['A', 'B', 'C', 'D']:
            closed_results.append(log_entry)
        else:
            open_results.append(log_entry)
            
    all_results = closed_results + open_results
    if args.output_log_json and all_results:
        with open(args.output_log_json, 'w') as f:
            json.dump(all_results, f, indent=4)

    print('Test ok!')


if __name__ == '__main__':
    main()