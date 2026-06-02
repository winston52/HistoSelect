# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
import os.path as osp
import warnings
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (AddedToken, AutoConfig, CLIPImageProcessor,
                          CLIPVisionModel, LlamaForCausalLM,
                          LlamaTokenizerFast, LlavaConfig,
                          LlavaForConditionalGeneration, LlavaProcessor)
from transformers.integrations import is_deepspeed_zero3_enabled
from peft import LoraConfig, get_peft_model

from xtuner.registry import BUILDER
from xtuner.utils import DEFAULT_IMAGE_TOKEN
from .modules import ProjectorConfig, ProjectorModel, dispatch_modules
from .modules.dispatch import SUPPORT_FLASH1, SUPPORT_FLASH2
from .utils import (LoadWoInit, find_all_linear_names,
                    get_peft_model_state_dict, guess_load_checkpoint,
                    make_inputs_require_grad,
                    prepare_inputs_labels_for_multimodal, traverse_dict)

from .torchscale.model.LongNet import make_longnet_from_name
import torch.nn.functional as F


# The PatchSelector class
class PatchSelector(nn.Module):
    """
    Predicts an independent selection probability for each patch within a group.
    """
    def __init__(self, visual_dim, question_dim, embed_dim=1024):
        super().__init__()
        self.question_proj = nn.Linear(question_dim, visual_dim)
        
        mlp_input_dim = visual_dim + visual_dim
        
        self.mlp_score_head = nn.Sequential(
            nn.Linear(mlp_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, patches_in_group, question_embedding):
        q_embed = question_embedding.squeeze(0)
        q_embed_expanded = q_embed.expand(patches_in_group.shape[0], -1)
        
        q_embed_projected = self.question_proj(
            q_embed_expanded.to(patches_in_group.dtype) 
        )
        
        combined_features = torch.cat([patches_in_group, q_embed_projected], dim=-1)
        
        # Get logits from the MLP
        logits = self.mlp_score_head(combined_features).squeeze(-1)
        
        # Apply sigmoid to get independent probabilities
        patch_probabilities = torch.sigmoid(logits)
        
        # Return both logits and probabilities
        return patch_probabilities, logits


class SimpleGroupSelector(nn.Module):
    """
    A simplified selector that processes each group independently.
    
    It concatenates each group's prototype with the question embedding and uses a 
    shared MLP to predict a selection score for each group.
    """
    def __init__(self, visual_dim, question_dim, embed_dim, K, **kwargs):
        """
        Initializes the SimpleGroupSelector.

        Args:
            visual_dim (int): The feature dimension of the group prototypes.
            question_dim (int): The feature dimension of the question embedding.
            embed_dim (int): The internal hidden dimension for the MLP.
            K (int): The total number of groups.
            **kwargs: Ignores unused arguments (like 'num_heads') to maintain a 
                      compatible interface.
        """
        super().__init__()
        self.K = K
        mlp_input_dim = visual_dim + question_dim
        
        self.mlp_score_head = nn.Sequential(
            nn.Linear(mlp_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, group_prototypes, question_embedding, attention_mask=None):
        """
        Forward pass for the SimpleGroupSelector.

        Args:
            group_prototypes (torch.Tensor): Tensor of group prototypes. 
                                             Shape: [B, K, visual_dim]
            question_embedding (torch.Tensor): Tensor of the question embedding. 
                                               Shape: [B, question_dim]
            attention_mask (torch.Tensor, optional): This argument is ignored but is
                                                     kept for interface compatibility.
        
        Returns:
            torch.Tensor: The independent selection score for each group. 
                          Shape: [B, K]
        """
        B, K, _ = group_prototypes.shape

        # Expand the question embedding to match the number of groups
        q_embed_expanded = question_embedding.unsqueeze(1).expand(-1, K, -1)
        
        # Concatenate features for each group
        combined_features = torch.cat([group_prototypes, q_embed_expanded], dim=-1)
        
        # Get logits from the MLP
        logits = self.mlp_score_head(combined_features).squeeze(-1)
        
        # Apply sigmoid to get independent selection scores between (0, 1)
        selection_scores = torch.sigmoid(logits)
        
        return selection_scores, logits


class GroupSelector(nn.Module):
    """
    A group selector that uses a full self-attention mechanism (Transformer) to
    predict selection probabilities in a global context.
    
    It processes the question and all group prototypes together, allowing it to
    make more sophisticated, context-aware decisions about the importance of each group.
    """
    def __init__(self, visual_dim, question_dim, embed_dim, K, num_heads=8):
        """
        Initializes the GroupSelector.

        Args:
            visual_dim (int): The feature dimension of the group prototypes.
            question_dim (int): The feature dimension of the question embedding.
            embed_dim (int): The internal dimension for the model (Transformer d_model).
            K (int): The total number of groups.
            num_heads (int): The number of attention heads for the Transformer.
        """
        super().__init__()
        self.K = K
        self.visual_proj = nn.Linear(visual_dim, embed_dim)
        self.question_proj = nn.Linear(question_dim, embed_dim)
        self.positional_embedding = nn.Embedding(K + 1, embed_dim)
        
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=0.1)
        
        self.mlp_score_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1))

    def forward(self, group_prototypes, question_embedding, attention_mask=None):
        """
        Forward pass for the TransformerGroupSampler.

        Args:
            group_prototypes (torch.Tensor): Tensor of group prototypes. 
                                             Shape: [B, K, visual_dim]
            question_embedding (torch.Tensor): Tensor of the question embedding. 
                                               Shape: [B, question_dim]
            attention_mask (torch.Tensor, optional): Mask to ignore non-existent groups. 
                                                     Shape: [B, K+1].

        Returns:
            (torch.Tensor, torch.Tensor): 
                - The independent sampling ratios for each group after sigmoid. Shape: [B, K]
                - The raw logits for each group before sigmoid. Shape: [B, K]
        """
        B, K, _ = group_prototypes.shape
        
        # Step 1: Project inputs to the model's uniform embedding dimension
        projected_prototypes = self.visual_proj(group_prototypes)
        projected_question = self.question_proj(question_embedding)
        
        # Step 2: Form a single sequence of [question, group_1, ..., group_K]
        question_token = projected_question.unsqueeze(1)
        sequence = torch.cat([question_token, projected_prototypes], dim=1)
        
        # Step 3: Add learnable positional embeddings to give each slot an identity
        positions = torch.arange(0, K + 1, device=sequence.device).unsqueeze(0)
        pos_enc = self.positional_embedding(positions)
        sequence_with_pos = sequence + pos_enc
        
        # Step 4: Pass the sequence through the Transformer to get contextual embeddings
        output_sequence = self.transformer_layer(
            src=sequence_with_pos,
            src_key_padding_mask=attention_mask
        )
        
        # Step 5: Extract only the outputs corresponding to the group tokens
        group_outputs = output_sequence[:, 1:, :]
        
        # Step 6: Pass each group's contextual embedding through an MLP to get a logit
        logits = self.mlp_score_head(group_outputs).squeeze(-1)
        
        # Step 7: Apply sigmoid to get independent selection probility between (0, 1)
        selection_probs = torch.sigmoid(logits)
        
        return selection_probs, logits

def convert_state_dict_to_hf(state_dict, mapping):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith('.inv_freq'):
            continue
        for key_to_modify, new_key in mapping.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)
        new_state_dict[key] = value
    return new_state_dict

class AdaptiveAvgPool1dLayer(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveAvgPool1dLayer, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return F.adaptive_avg_pool1d(x, self.output_size)
    
class LLaVAModel_Selector(BaseModel):

    def __init__(self,
                 llm,
                 freeze_llm=True,
                 freeze_projector=False,
                 visual_select_layer=-2,
                 pretrained_pth=None,
                 projector_depth=2,
                 llm_lora=None,
                 visual_encoder_lora=None,
                 use_activation_checkpointing=True,
                 max_position_embeddings=None,
                 hidden_size=512,
                 train_stage='2',
                 enable_long_net=True,
                 
                 use_selector=True,
                 selector_type='simple',
                 question_embedding_dim=512,
                 num_groups=13,
                 beta_initial=0.0,
                 beta_final=0.2,
                 gamma_initial=0.0,
                 gamma_final=0.1,
                 max_iters=5000,
                 group_prior_temperature=1.0,
                 group_prior_bias=0.0,
                 patch_prior_temperature=1.0,
                 patch_prior_bias=0.0,
                 text_encoder = 'conch',
                 log_path=None):
        super().__init__()
        
        self.enable_long_net = enable_long_net
        if enable_long_net:
            print('enable long net')
        else:
            print('disable long net')
            
        self.freeze_projector = freeze_projector
        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = True
        if train_stage == '1':
            print('train_stage == 1')
            self.freeze_llm = True
            self.freeze_long_net = False

        elif train_stage == '2':
            print('train_stage == 2')
            self.freeze_llm = True #False
            self.freeze_long_net = False #False

        with LoadWoInit():
            if isinstance(llm, dict):
                llm = self._dispatch_lm_model_cfg(llm, max_position_embeddings)

            self.llm = self._build_from_cfg_or_module(llm)

        self.encoder_name = "LongNet_{}_layers_{}_dim".format(2, 512)
        self.LongNet_encoder = make_longnet_from_name(self.encoder_name) # , drop_path_rate=0.3, dropout=0.3, segment_length=1024
        
        self.llm.config.use_cache = False
        dispatch_modules(self.llm)

        self.projector_depth = projector_depth

        projector_config = ProjectorConfig(
            visual_hidden_size=hidden_size,
            llm_hidden_size=self.llm.config.hidden_size,
            depth=self.projector_depth)        

        self.projector = ProjectorModel(projector_config).to(self.llm.dtype)
        
        self.question_prior_proj = nn.Linear(
            question_embedding_dim, self.llm.config.hidden_size)
        self.group_prior_temperature = group_prior_temperature
        self.group_prior_bias = group_prior_bias
        self.patch_prior_temperature = patch_prior_temperature
        self.patch_prior_bias = patch_prior_bias
        
        if text_encoder == 'conch':
            logit_scale_value = 4.0315 # the scaling factor for conch text encoder
            print(f"Using fixed logit_scale for 'conch': {logit_scale_value}")
        else:
            logit_scale_value = 0.0
            print(f"Using default logit_scale: {logit_scale_value}")
        self.register_buffer('logit_scale', torch.tensor(logit_scale_value))
        
        self.beta_initial = beta_initial
        self.beta_final = beta_final
        self.beta = self.beta_initial
        self.gamma_initial = gamma_initial
        self.gamma_final = gamma_final
        self.gamma = self.gamma_initial
        
        self.max_iters = max_iters
        self.num_groups = num_groups
        
        self.register_buffer('_iter_counter', torch.tensor(0, dtype=torch.long))

        self.use_selector = use_selector
        self.group_selector = None
        self.patch_selector = None

        if self.use_selector:
            # 1. Initialize the Group Selector
            selector_kwargs = {
                'visual_dim': self.llm.config.hidden_size,
                'question_dim': self.llm.config.hidden_size,
                'embed_dim': 1024,
                'K': self.num_groups,
                'num_heads': 8
            }
            if selector_type == 'simple':
                print("Initializing with SimpleGroupSelector as GroupSelector.")
                self.group_selector = SimpleGroupSelector(**selector_kwargs)
            elif selector_type == 'transformer':
                print("Initializing with GroupSelector.")
                self.group_selector = GroupSelector(**selector_kwargs)
            else:
                raise ValueError(f"A valid 'selector_type' ('simple' or 'transformer') is required when use_selector is True.")
            
            # 2. Initialize the Patch Selector
            print("Initializing with PatchSelector.")
            self.patch_selector = PatchSelector(
                visual_dim=self.llm.config.hidden_size,
                question_dim=self.llm.config.hidden_size
            )
            
        else:
            print("Hierarchical selection is disabled. The model will use all patches.")
        
        # log path and log function — defaults to <cwd>/selection.txt when
        # `log_path` is not provided. The training launcher (xtuner) sets cwd
        # to the work_dir, so by default the file lands inside the run dir.
        self.log_path = log_path if log_path is not None else os.path.join(
            os.getcwd(), 'selection.txt')
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        def _log_to_file(message: str):
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
        self._log = _log_to_file

        if self.freeze_llm:
            print('freeze_llm')
            self.llm.requires_grad_(False)        
        if self.freeze_long_net:
            print('freeze_long_net')
            self.LongNet_encoder.requires_grad_(False)
        if self.freeze_projector:
            print('freeze_projector')
            self.projector.requires_grad_(False)
            
        if llm_lora is not None:
            print("Found LoRA config, preparing model for LoRA training via helper function...")
            self._prepare_llm_for_lora(llm_lora)
            print("Model prepared for LoRA.")
            self.llm.print_trainable_parameters()
        
        # check the gradient of all part of the model
        print("\n" + "="*50)
        print("Verifying which model parts are trainable...")
        print("="*50)

        # A dictionary containing the main components of your model
        modules_to_check = {
            "LLM": self.llm,
            "LongNet Encoder": self.LongNet_encoder,
            "Projector": self.projector,
            "Group Selector": self.group_selector,
            "Patch Selector": self.patch_selector,
            "Prior Projector": self.question_prior_proj
        }

        total_trainable_params = 0

        # Iterate through each module to check its parameters
        for name, module in modules_to_check.items():
            # Calculate total and trainable parameters for the current module
            module_params = sum(p.numel() for p in module.parameters())
            module_trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            total_trainable_params += module_trainable_params
            
            # Determine the training status
            status = "FROZEN"
            if module_trainable_params > 0:
                if module_trainable_params == module_params:
                    status = "FULLY TRAINABLE"
                else:
                    # This case handles scenarios like LoRA where only some params are trainable
                    status = "PARTIALLY TRAINABLE"

            # Print a formatted summary line for the module
            print(f"Module: {name:<20} | Status: {status:<20} | Trainable params: {module_trainable_params}")

        print("-" * 50)
        print(f"Total Trainable Parameters in Model: {total_trainable_params}")
        print("="*50 + "\n")
        

        if use_activation_checkpointing:
            # For backward compatibility
            if hasattr(self.llm, 'enable_input_require_grads'):
                self.llm.enable_input_require_grads()
            else:
                self.llm.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad)
                
            self.projector.enable_input_require_grads()
            # enable gradient (activation) checkpointing for memory efficiency
            self.gradient_checkpointing_enable()

        self.use_llm_lora = None
        self.use_visual_encoder_lora =  None

        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora, use_activation_checkpointing)

        if pretrained_pth is not None: # load the pretrained checkpoint
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

            self.load_state_dict(pretrained_state_dict, strict=False)
            print_log(f'Load pretrained weight from {pretrained_pth}',
                      'current')

        self.visual_select_layer = visual_select_layer

        self._is_init = True

        self.is_first_iter = True
        
    def _pairwise_cosine_similarity(self, x: torch.Tensor) -> torch.Tensor:
        """Calculates pairwise cosine similarity for a batch of vectors."""
        x_norm = F.normalize(x, p=2, dim=1)
        return torch.mm(x_norm, x_norm.t())
    
    def _divprune_sampler(self, group_patches: torch.Tensor, num_to_sample: int) -> torch.Tensor:
        """
        Selects a diverse subset of patches from a group using DivPrune's logic.
        """
        num_patches_in_group = group_patches.shape[0]
        if num_to_sample >= num_patches_in_group:
            return torch.arange(num_patches_in_group, device=group_patches.device)
        if num_to_sample <= 0:
            return torch.tensor([], dtype=torch.long, device=group_patches.device)

        with torch.no_grad():
            distance_matrix = 1.0 - self._pairwise_cosine_similarity(group_patches)
            selected_indices = torch.empty(num_to_sample, dtype=torch.long, device=group_patches.device)
            
            first_idx = torch.argmax(distance_matrix.sum(dim=1))
            selected_indices[0] = first_idx
            min_distances = distance_matrix[first_idx, :].clone()

            for i in range(1, num_to_sample):
                next_idx = torch.argmax(min_distances)
                selected_indices[i] = next_idx
                min_distances[next_idx] = -1.0 
                
                distances_to_new_point = distance_matrix[next_idx, :]
                min_distances = torch.minimum(min_distances, distances_to_new_point)
                
        return selected_indices
        
    def _update_weight(self):
        """Anneals beta during training based on iteration progress."""
        if self.training and self.max_iters > 0:
            progress = min(self._iter_counter.item() / self.max_iters, 1.0)
            self.beta = self.beta_initial + (self.beta_final - self.beta_initial) * progress
            self.gamma = self.gamma_initial + (self.gamma_final - self.gamma_initial) * progress

    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
                lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self,
                              lora_config,
                              use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.llm = prepare_model_for_kbit_training(
            self.llm, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.llm)
            lora_config.target_modules = modules
        self.llm = get_peft_model(self.llm, lora_config)

    def _prepare_visual_encoder_for_lora(self,
                                         lora_config,
                                         use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.visual_encoder)
            lora_config.target_modules = modules
        self.visual_encoder = get_peft_model(self.visual_encoder, lora_config)

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        self.projector.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.projector.gradient_checkpointing_disable()

    def init_weights(self):
        pass

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.visual_encoder, state_dict=state_dict))
        elif not self.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'visual_encoder.' in k
            })
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(self.llm, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'llm.' in k})
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'projector.' in k})
        
        # Step 4. LongNet_encoder
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'LongNet_encoder.' in k})
        
        if self.use_selector:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'sampler.' in k})
            
        return to_return

    @staticmethod
    def _prepare_for_long_context_training(cfg, llm_cfg,
                                           max_position_embeddings):

        orig_rope_scaling = getattr(llm_cfg, 'rope_scaling', None)
        if orig_rope_scaling is None:
            orig_rope_scaling = {'factor': 1}

        orig_rope_scaling_factor = orig_rope_scaling[
            'factor'] if 'factor' in orig_rope_scaling.keys() else 1
        orig_ctx_len = getattr(llm_cfg, 'max_position_embeddings', None)
        if orig_ctx_len:
            orig_ctx_len *= orig_rope_scaling_factor
            if max_position_embeddings > orig_ctx_len:
                scaling_factor = float(
                    math.ceil(max_position_embeddings / orig_ctx_len))
                llm_cfg.rope_scaling = {
                    'type': 'linear',
                    'factor': scaling_factor
                }

        # hardcode for internlm2
        llm_cfg.attn_implementation = 'flash_attention_2'
        cfg.config = llm_cfg

        return cfg, llm_cfg

    @staticmethod
    def _prepare_for_flash_attn(cfg, llm_cfg):
        cls_name = type(llm_cfg).__name__
        SUPPORT_SDPA_ATTN = ('LlamaConfig', 'GemmaConfig', 'MistralConfig',
                             'MixtralConfig', 'Qwen2Config', 'Qwen2MoeConfig',
                             'Starcoder2Config', 'Starcoder2Config',
                             'Phi3Config')
        SUPPORT_FLASH_ATTN2 = ('InternLM2Config', 'LlamaConfig', 'GemmaConfig',
                               'MistralConfig', 'MixtralConfig', 'Qwen2Config',
                               'Qwen2MoeConfig', 'Starcoder2Config',
                               'Starcoder2Config', 'Phi3Config')

        torch_dtype = torch.bfloat16 if (
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()) \
            else torch.float16

        if getattr(cfg, 'attn_implementation', None) is not None:
            # Flash Attention 2.0 only supports torch.float16 and
            # torch.bfloat16 dtypes
            if cfg.attn_implementation == 'flash_attention_2':
                cfg.torch_dtype = torch_dtype
        elif SUPPORT_FLASH2 and cls_name in SUPPORT_FLASH_ATTN2:
            cfg.torch_dtype = torch_dtype
            cfg.attn_implementation = 'flash_attention_2'
        elif SUPPORT_FLASH1 and cls_name in SUPPORT_SDPA_ATTN:
            cfg.attn_implementation = 'sdpa'

        return cfg, llm_cfg

    @staticmethod
    def _prepare_for_qlora_zero3(cfg):
        if (not is_deepspeed_zero3_enabled()) or (not hasattr(
                cfg, 'quantization_config')):
            return cfg

        torch_dtype = torch.bfloat16 if (
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()) \
            else torch.float16

        cfg.torch_dtype = torch_dtype
        quantization_config = cfg.quantization_config
        quantization_config.bnb_4bit_compute_dtype = torch_dtype
        quantization_config.bnb_4bit_quant_storage = torch_dtype

        return cfg

    def _dispatch_lm_model_cfg(self, cfg, max_position_embeddings=None):
        cfg = self._prepare_for_qlora_zero3(cfg)
        pretrained_model_name_or_path = cfg.pretrained_model_name_or_path
        llm_cfg = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True)
        cfg, llm_cfg = self._prepare_for_flash_attn(cfg, llm_cfg)
        if max_position_embeddings is not None:
            cfg, llm_cfg = self._prepare_for_long_context_training(
                cfg, llm_cfg, max_position_embeddings)
        return cfg

    def _build_from_cfg_or_module(self, cfg_or_mod):
        if isinstance(cfg_or_mod, nn.Module):
            return cfg_or_mod
        elif isinstance(cfg_or_mod, dict):
            traverse_dict(cfg_or_mod)
            return BUILDER.build(cfg_or_mod)
        else:
            raise NotImplementedError

    def forward(self, data, data_samples=None, mode='loss'):
        
        self._update_weight()
            
        if self.is_first_iter:
            # hardcode for qlora DeepSpeed ZeRO3, put buffers and QuantState to
            # device
            # Only required in `LLaVAModel_Selector` .
            # We do not need this in `SupervisedFinetune` .
            self.to(data['input_ids'].device)
            self.is_first_iter = False
        
        sample_id = data.pop('id', None)
        question_embedding = data.pop('question_embedding', None)
        weak_labels = data.pop('weak_labels', None)
        organ_name = data.pop('organ_name', None)
        case_name = data.pop('case_name', None)
        question = data.pop('question', None)
        answer = data.pop('answer', None) # Popped to keep data clean, not used in logic
        idx_to_class = data.pop('idx_to_class', None)
        
        # Initialize the compression loss
        group_compression_loss = torch.tensor(0.0, device=data['input_ids'].device, dtype=self.llm.dtype)
        patch_compression_loss = torch.tensor(0.0, device=data['input_ids'].device, dtype=self.llm.dtype)
        
        # This will be passed to the compute_loss function.
        log_info = {}

        # data_dict['pixel_values']=[[pixel_values of img1], [pixel_values of img2], ...]
        if 'pixel_values' in data:
            feat_to_proj = data['pixel_values'].to(self.llm.dtype) # torch.Size([1, img_num, 512])
            if self.enable_long_net: # to be modified
                long_net_output = self.LongNet_encoder(src_tokens=None, token_embeddings=feat_to_proj.permute(1, 0, 2))["encoder_out"] # shape: (img_num, 1, 1024)
                # wl - output shape (img_num, 1, 512)
                feat_to_proj = long_net_output.permute(1, 0, 2) # permuted shape: [1, img_num, 512]

            patch_features = self.projector(feat_to_proj.to(self.llm.dtype)) # output shape [1, patch_num, 3584]
            
            if self.use_selector and question_embedding is not None and weak_labels is not None:
                B, N, D_feat = patch_features.shape
                K = self.num_groups
                
                projected_q_emb = self.question_prior_proj(question_embedding.to(self.llm.dtype))
                
                # STEP 1: Calculate Group Prototypes
                group_prototypes_sum = torch.zeros(B, K, D_feat, device=patch_features.device, dtype=torch.float32)
                group_counts = torch.zeros(B, K, device=patch_features.device, dtype=torch.float32)
                group_indices_map = weak_labels.unsqueeze(-1).expand(-1, -1, D_feat)
                group_prototypes_sum.scatter_add_(1, group_indices_map, patch_features.to(torch.float32))
                ones = torch.ones_like(patch_features[:, :, 0])
                group_counts.scatter_add_(1, weak_labels, ones.to(torch.float32))
                group_prototypes = group_prototypes_sum / (group_counts.unsqueeze(-1) + 1e-7)

                # STEP 2: Predict Sampling Ratios for All Groups
                group_mask = (group_counts == 0)
                question_mask = torch.zeros(B, 1, dtype=torch.bool, device=group_mask.device)
                attention_mask = torch.cat([question_mask, group_mask], dim=1)
                
                group_selection_probs, group_logits = self.group_selector(
                    group_prototypes.to(self.llm.dtype),
                    projected_q_emb.to(self.llm.dtype),
                    attention_mask=attention_mask
                )
                
                # STEP 3: Calculate Beta Loss (Compression Loss)
                q_emb_norm = F.normalize(projected_q_emb, p=2, dim=-1).unsqueeze(1)
                proto_norm = F.normalize(group_prototypes.to(self.llm.dtype), p=2, dim=-1)

                # 3.1: Calculate raw cosine similarity in float32 for precision
                prior_scores = torch.bmm(
                    q_emb_norm.to(torch.float32),
                    proto_norm.to(torch.float32).transpose(1, 2)
                ).squeeze(1)
                
                # 3.2: Scale the scores using the FIXED logit_scale buffer
                scaled_scores = prior_scores * self.logit_scale.exp()
                
                # 3.3: Calculate target probabilities and convert back to original dtype
                scaled_and_shifted_scores = scaled_scores + self.group_prior_bias
                target_prior = torch.sigmoid(
                    scaled_and_shifted_scores / self.group_prior_temperature
                ).detach().to(self.llm.dtype)

                # 3.4: Calculate the BCE loss
                loss_fn_group = torch.nn.BCEWithLogitsLoss(reduction='none')
                bce_loss = loss_fn_group(group_logits.to(torch.float32), target_prior.to(torch.float32))

                # 3.5: Apply mask to only consider active groups
                active_group_mask = (group_counts > 0)
                masked_loss = bce_loss * active_group_mask
                num_active_groups = active_group_mask.sum()

                if num_active_groups > 0:
                    group_compression_loss = (masked_loss.sum() / num_active_groups)
                    
                # Step 4: Hierarchical Gating, Patch IB Loss, and STE Selection ---
                final_selection_probs = torch.zeros(B, N, device=patch_features.device, dtype=patch_features.dtype)
                hard_selection_mask = torch.zeros(B, N, device=patch_features.device, dtype=patch_features.dtype)
                active_group_patch_losses = []

                for i in range(K):
                    group_count_i = group_counts[0, i]
                    if group_counts[0, i] > 0:
                        group_prob_i = group_selection_probs[:, i]
                        group_mask = (weak_labels == i).squeeze(0)
                        patches_in_group = patch_features.squeeze(0)[group_mask]
                        
                        patch_selection_probs_i, patch_logits_i = self.patch_selector(patches_in_group.to(self.llm.dtype), projected_q_emb.to(self.llm.dtype))
 
                        K_i_float = group_prob_i.detach() * group_count_i.detach()
                        K_i = torch.round(K_i_float).int().item()
                        K_i = min(max(0, K_i), int(group_count_i.item()))
                        
                        if K_i > 0:
                            _, selected_indices_in_group = torch.topk(patch_selection_probs_i, k=K_i)
                            hard_mask_in_group = torch.zeros_like(
                                patch_selection_probs_i, dtype=patch_features.dtype
                            ).scatter_(0, selected_indices_in_group, 1.0)
                        else:
                            hard_mask_in_group = torch.zeros_like(
                                patch_selection_probs_i, dtype=patch_features.dtype)

                        final_probs_i = group_prob_i * patch_selection_probs_i
                        final_selection_probs.squeeze(0)[group_mask] = final_probs_i
                        hard_selection_mask.squeeze(0)[group_mask] = hard_mask_in_group                        

                        # 4.3: Calculate Patch-Level IB Loss for this group
                        if self.training and self.gamma > 0:
                            # Posterior logits are from the selector
                            p_posterior_logits = patch_logits_i
                            
                            q_emb_norm_patch = F.normalize(projected_q_emb.squeeze(0), p=2, dim=-1)
                            patches_norm = F.normalize(patches_in_group, p=2, dim=-1)
                            prior_scores_i = torch.matmul(patches_norm.to(torch.float32), q_emb_norm_patch.to(torch.float32))
                            scaled_prior_scores_i = prior_scores_i * self.logit_scale.exp()
                            scaled_and_shifted_scores_i = scaled_prior_scores_i + self.patch_prior_bias
                            
                            p_prior_probs = torch.sigmoid(
                                scaled_and_shifted_scores_i / self.patch_prior_temperature
                            ).detach().to(self.llm.dtype)

                            # Calculate Binary Cross-Entropy Loss using the stable logit-based function
                            loss_fn_patch = torch.nn.BCEWithLogitsLoss(reduction='mean') 
                            bce_loss_patch = loss_fn_patch(
                                p_posterior_logits.to(torch.float32), 
                                p_prior_probs.to(torch.float32)
                            )
                            active_group_patch_losses.append(bce_loss_patch)
                
                # STEP 5: Aggregate Selected Patches
                if active_group_patch_losses:
                    patch_compression_loss = torch.mean(torch.stack(active_group_patch_losses))

                # 5.1: Perform hard selection using STE
                selection_ste = hard_selection_mask.detach() + final_selection_probs - final_selection_probs.detach()

                # 5.2: Extract selected patches
                selected_indices = torch.nonzero(hard_selection_mask.squeeze(0), as_tuple=True)[0]
                
                if len(selected_indices) > 0:
                    selected_features_pre_ste = patch_features.squeeze(0)[selected_indices]
                    ste_weights_for_selected = selection_ste.squeeze(0)[selected_indices]
                    selected_pixel_values = selected_features_pre_ste * ste_weights_for_selected.unsqueeze(-1)
                    selected_pixel_values = selected_pixel_values.unsqueeze(0)
                else:
                    # To prevent errors, pass at least one token. A zero vector is a safe choice.
                    selected_pixel_values = torch.zeros(B, 1, D_feat, device=patch_features.device, dtype=patch_features.dtype)
                    
                # STEP 6: Prepare data for logging
                if B == 1:
                    log_info['id'] = sample_id[0] if isinstance(sample_id, (list, tuple)) else sample_id
                    log_info['case_name'] = case_name[0] if isinstance(case_name, (list, tuple)) else case_name
                    log_info['organ_name'] = organ_name[0] if isinstance(organ_name, (list, tuple)) else organ_name
                    log_info['question'] = question[0] if isinstance(question, (list, tuple)) else question
                    log_info['answer'] = answer[0] if isinstance(answer, (list, tuple)) else answer
                    
                    num_selected_patches = selected_pixel_values.shape[1]
                    log_info['num_patches_all'] = N
                    log_info['num_selected_patches'] = num_selected_patches
                    log_info['patch_selection_rate'] = num_selected_patches / N if N > 0 else 0.0
                    
                    current_map = idx_to_class[0] if isinstance(idx_to_class, (list, tuple)) else idx_to_class
                    def get_class_name(idx):
                        return current_map.get(idx, f"UnknownIdx_{idx}") if current_map else str(idx)

                    # NEW: Create a detailed, per-group log that shows the hierarchical selection results.
                    selection_details_log = {}
                    # Get the labels of only the patches that were actually selected
                    selected_labels = weak_labels.squeeze(0)[selected_indices]

                    for i in range(K):
                        if group_counts[0, i] > 0:
                            class_name = get_class_name(i)
                            total_in_group = int(group_counts[0, i].item())
                            # Count how many of the selected patches belong to this group
                            selected_in_group = int((selected_labels == i).sum().item())
                            
                            group_prob = group_selection_probs[0, i].item()
                            prior_prob = target_prior[0, i].item() # From group prior calculation
                            
                            # Create a rich log string for each group
                            selection_details_log[class_name] = (
                                f"GroupProb: {group_prob:.3f} (Prior: {prior_prob:.3f}) | "
                                f"Selected: {selected_in_group} / {total_in_group}"
                            )
                    
                    log_info['selection_details'] = selection_details_log
                
                # The data preparation part remains the same
                data['pixel_values'] = selected_pixel_values
            
            else: # Fallback if no selector is used or required data is missing
                data['pixel_values'] = patch_features
            
            # Prepare final inputs for the language model
            data = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)
        
        if self.training:
            self._iter_counter.add_(1)

        if mode == 'loss':
            # Pass the prepared log_info dictionary to the compute_loss function.
            return self.compute_loss(data, data_samples, group_compression_loss=group_compression_loss, patch_compression_loss=patch_compression_loss, log_info=log_info)
        elif mode == 'predict':
            return self.predict(data, data_samples)
        elif mode == 'tensor':
            return self._forward(data, data_samples)
        else:
            raise NotImplementedError

    def _forward(self, data, data_samples=None):

        outputs = self.llm(**data)

        return outputs

    def predict(self, data, data_samples=None):
        outputs = self.llm(**data)
        logits_dict = [{'logits': logits} for logits in outputs.logits]
        return logits_dict

    def compute_loss(self, data, data_samples=None, group_compression_loss=0.0, patch_compression_loss=0.0, log_info=None):
        """
        Computes the final loss, including VQA and hierarchical IB losses.
        """
        outputs = self.llm(**data)
        vqa_loss = outputs.loss

        # Apply independent weights to each compression loss
        weighted_group_loss = self.beta * group_compression_loss
        weighted_patch_loss = self.gamma * patch_compression_loss
        total_loss = vqa_loss + weighted_group_loss + weighted_patch_loss
        
        loss_dict = {
            'loss': total_loss,
            'loss_vqa': vqa_loss,
            'loss_group_ib': group_compression_loss,
            'loss_patch_ib': patch_compression_loss,
        }

        # The log will now only be printed during training to avoid clutter during validation/testing.
        if log_info and self.training:
            # Console Logging
            print("\n" + "="*80)
            print(f"🔬 HIERARCHICAL SELECTION LOG (beta={self.beta:.4f}, gamma={self.gamma:.4f}")
            print(f"  - ID: {log_info.get('id', 'N/A')}")
            print(f"  - Case Name: {log_info.get('case_name', 'N/A')}")
            print(f"  - Question: {log_info.get('question', 'N/A')}")
            print(f"  - GT Answer: {log_info.get('answer', 'N/A')}")
            print("-" * 40)
            print("  - Selection Details per Group:")
            selection_details = log_info.get('selection_details', {})
            if selection_details:
                for class_name, details in selection_details.items():
                    print(f"    - {class_name:<20}: {details}")
            else:
                print("    - N/A")
            print("-" * 40)
            print(f"  - Patch Counts (Selected/Total): {int(log_info.get('num_selected_patches', 0))} / {log_info.get('num_patches_all', 0)}")
            print(f"  - Final Patch Selection Rate: {log_info.get('patch_selection_rate', 0):.2%}")
            print("-" * 40)
            print(f"  - VQA Loss: {vqa_loss.item():.6f}")
            print(f"  - Group IB Loss (raw): {group_compression_loss.item():.6f} (weighted: {weighted_group_loss.item():.6f})")
            print(f"  - Patch IB Loss (raw): {patch_compression_loss.item():.6f} (weighted: {weighted_patch_loss.item():.6f})")
            print(f"  - Total Loss: {total_loss.item():.6f}")
            print("="*80 + "\n")

            # File Logging
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            selection_details_str = "\n".join([f"    - {name}: {details}" for name, details in log_info.get('selection_details', {}).items()])
            log_content = (
                f"-------------------------- [{timestamp}] --------------------------\n"
                f"[Info]\n"
                f"    - ID:         {log_info.get('id', 'N/A')}\n"
                f"    - Case Name:  {log_info.get('case_name', 'N/A')}\n"
                f"    - Question:   {log_info.get('question', 'N/A')}\n"
                f"    - GT Answer:  {log_info.get('answer', 'N/A')}\n"
                f"    - Hyperparams: beta={self.beta:.4f}, gamma={self.gamma:.4f}\n\n"
                f"[Selection Details]\n{selection_details_str}\n\n"
                f"[Patch Selection Results]\n"
                f"    - Total:    {log_info.get('num_patches_all', 0)}\n"
                f"    - Selected: {int(log_info.get('num_selected_patches', 0))}\n"
                f"    - Rate:     {log_info.get('patch_selection_rate', 0):.2%}\n\n"
                f"[Loss]\n"
                f"    - VQA Loss:          {vqa_loss.item():.6f}\n"
                f"    - Group IB Loss:     {group_compression_loss.item():.6f}\n"
                f"    - Patch IB Loss:     {patch_compression_loss.item():.6f}\n"
                f"    - Weighted Group IB: {weighted_group_loss.item():.6f}\n"
                f"    - Weighted Patch IB: {weighted_patch_loss.item():.6f}\n"
                f"    - Total Loss:        {total_loss.item():.6f}\n\n\n"
            )
            self._log(log_content)

        return loss_dict

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)

    def to_hf(self,
              cfg,
              save_dir,
              fp32=False,
              save_pretrained_kwargs={},
              save_format='xtuner',
              **kwargs):
        if save_format == 'xtuner':
            self.to_xtuner_llava(cfg, save_dir, fp32, save_pretrained_kwargs)
        elif save_format == 'huggingface':
            self.to_huggingface_llava(cfg, save_dir, fp32,
                                      save_pretrained_kwargs)
        elif save_format == 'official':
            self.to_official_llava(cfg, save_dir, fp32, save_pretrained_kwargs)
        else:
            raise NotImplementedError

    def to_xtuner_llava(self,
                        cfg,
                        save_dir,
                        fp32=False,
                        save_pretrained_kwargs={}):
        # LLM
        self.llm.config.use_cache = True
        if not fp32:
            print_log('Convert LLM to float16', 'current')
            self.llm.half()
        if self.use_llm_lora:
            llm_path = osp.join(save_dir, 'llm_adapter')
            print_log(f'Saving LLM adapter to {llm_path}', 'current')
            self.llm.save_pretrained(llm_path, **save_pretrained_kwargs)
        elif not self.freeze_llm:
            llm_path = save_dir
            print_log(f'Saving LLM tokenizer to {llm_path}', 'current')
            tokenizer = BUILDER.build(cfg.tokenizer)
            tokenizer.save_pretrained(llm_path, **save_pretrained_kwargs)
            print_log(f'Saving LLM to {llm_path}', 'current')
            self.llm.save_pretrained(llm_path, **save_pretrained_kwargs)
        self.llm.config.use_cache = False

        # Visual Encoder
        if self.use_visual_encoder_lora:
            visual_encoder_path = osp.join(save_dir, 'visual_encoder_adapter')
            print_log(
                f'Saving visual_encoder adapter to {visual_encoder_path}',
                'current')
            self.visual_encoder.save_pretrained(visual_encoder_path,
                                                **save_pretrained_kwargs)
        elif not self.freeze_visual_encoder:
            visual_encoder_path = osp.join(save_dir, 'visual_encoder')
            print_log(
                'Saving visual_encoder image_processor to'
                f'{visual_encoder_path}', 'current')
            image_processor = BUILDER.build(cfg.image_processor)
            image_processor.save_pretrained(visual_encoder_path,
                                            **save_pretrained_kwargs)
            print_log(f'Saving visual_encoder to {visual_encoder_path}',
                      'current')
            self.visual_encoder.save_pretrained(visual_encoder_path,
                                                **save_pretrained_kwargs)

        # Projector
        projector_path = osp.join(save_dir, 'projector')
        print_log(f'Saving projector to {projector_path}', 'current')
        self.projector.save_pretrained(projector_path,
                                       **save_pretrained_kwargs)

        # LongNet_encoder
        LongNet_encoder_path = osp.join(save_dir, 'LongNet_encoder')
        print_log(f'Saving LongNet_encoder to {LongNet_encoder_path}', 'current')
        self.LongNet_encoder.save_pretrained(LongNet_encoder_path,
                                       **save_pretrained_kwargs)
        
        
        
        

    def to_huggingface_llava(self,
                             cfg,
                             save_dir,
                             fp32=False,
                             save_pretrained_kwargs={}):

        LLM_MAPPING = {
            'model': 'language_model.model',
            'lm_head': 'language_model.lm_head',
        }
        VIT_MAPPING = {
            'vision_model': 'vision_tower.vision_model',
        }
        PROJECTOR_MAPPING = {
            'model.0': 'multi_modal_projector.linear_1',
            'model.2': 'multi_modal_projector.linear_2',
        }
        LONGNET_MAPPING = {
            'layers.0': 'LongNet_encoder.layers.0',
            'layers.1': 'LongNet_encoder.layers.1',
            'layer_norm': 'LongNet_encoder.layer_norm'
        }

        assert getattr(self.llm, 'hf_quantizer', None) is None, \
            'This conversion format does not support quantized LLM.'

        # get state_dict
        llm = self.llm
        if self.use_llm_lora:
            llm = self.llm.merge_and_unload()
        llm.config.use_cache = True
        if not fp32:
            print_log('Convert LLM to float16', 'current')
            llm.half()

        assert isinstance(llm, LlamaForCausalLM), \
            'This conversion format only supports LlamaForCausalLM.'
        llm_state_dict = llm.state_dict()
        llm_state_dict = convert_state_dict_to_hf(llm_state_dict, LLM_MAPPING)

        need_visual_encoder = (not self.freeze_visual_encoder
                               or self.use_visual_encoder_lora)
        visual_encoder = self.visual_encoder
        if self.use_visual_encoder_lora:
            visual_encoder = self.visual_encoder.merge_and_unload()
        assert isinstance(visual_encoder, CLIPVisionModel),\
            'This conversion format only supports CLIPVisionModel.'
        if need_visual_encoder:
            visual_encoder_state_dict = visual_encoder.state_dict()
            visual_encoder_state_dict = convert_state_dict_to_hf(
                visual_encoder_state_dict, VIT_MAPPING)
        else:
            visual_encoder_state_dict = {}

        projector_state_dict = self.projector.state_dict()
        projector_state_dict = convert_state_dict_to_hf(
            projector_state_dict, PROJECTOR_MAPPING)

        LongNet_encoder_state_dict = self.LongNet_encoder.state_dict()
        LongNet_encoder_state_dict = convert_state_dict_to_hf(
            LongNet_encoder_state_dict, LONGNET_MAPPING)

        state_dict = {
            **projector_state_dict,
            **llm_state_dict,
            **visual_encoder_state_dict,
            **LongNet_encoder_state_dict
        }

        # init model
        text_config = llm.config
        vision_config = visual_encoder.config
        config = LlavaConfig(
            text_config=text_config,
            vision_config=vision_config,
            attn_implementation='eager')

        with init_empty_weights():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', message='.*non-meta.*', category=UserWarning)
                model = LlavaForConditionalGeneration(config)
        model.load_state_dict(state_dict, strict=True, assign=True)

        # processor
        cfg.tokenizer.type = LlamaTokenizerFast.from_pretrained
        tokenizer = BUILDER.build(cfg.tokenizer)

        tokenizer.add_tokens(
            AddedToken(DEFAULT_IMAGE_TOKEN, special=True, normalized=False),
            special_tokens=True)
        tokenizer.add_special_tokens({'pad_token': '<pad>'})

        image_processor = BUILDER.build(cfg.image_processor)
        assert isinstance(image_processor, CLIPImageProcessor),\
            'This conversion format only supports CLIPImageProcessor.'

        processor = LlavaProcessor(
            tokenizer=tokenizer, image_processor=image_processor)

        # Pad to 64 for performance reasons
        pad_shape = 64

        pre_expansion_embeddings = \
            model.language_model.model.embed_tokens.weight.data
        mu = torch.mean(pre_expansion_embeddings, dim=0).float()
        n = pre_expansion_embeddings.size()[0]
        sigma = ((pre_expansion_embeddings - mu).T
                 @ (pre_expansion_embeddings - mu)) / n
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            mu, covariance_matrix=1e-5 * sigma)

        # We add an image token so we need to resize the model
        ori_vocab_size = config.text_config.vocab_size
        tokenizer_vocab_size = tokenizer.encode('<pad>')[-1]
        added_token = tokenizer_vocab_size - ori_vocab_size

        if added_token > 0:
            model.resize_token_embeddings(ori_vocab_size + added_token,
                                          pad_shape)
            model.language_model.model.embed_tokens.weight.data[
                ori_vocab_size:] = torch.stack(
                    tuple(
                        dist.sample()
                        for _ in range(model.language_model.model.embed_tokens.
                                       weight.data[ori_vocab_size:].shape[0])),
                    dim=0,
                )
            model.language_model.lm_head.weight.data[
                ori_vocab_size:] = torch.stack(
                    tuple(dist.sample()
                          for _ in range(model.language_model.lm_head.weight.
                                         data[ori_vocab_size:].shape[0])),
                    dim=0,
                )
        model.config.image_token_index = tokenizer.encode(
            DEFAULT_IMAGE_TOKEN)[-1]
        model.config.pad_token_id = tokenizer.encode('<pad>')[-1]

        # save
        print_log(f'Saving to {save_dir}', 'current')
        model.save_pretrained(save_dir, **save_pretrained_kwargs)
        processor.save_pretrained(save_dir, **save_pretrained_kwargs)

    def to_official_llava(self,
                          cfg,
                          save_dir,
                          fp32=False,
                          save_pretrained_kwargs={}):

        VIT_MAPPING = {
            'vision_model': 'model.vision_tower.vision_tower.vision_model',
        }
        PROJECTOR_MAPPING = {
            'model.0': 'model.mm_projector.0',
            'model.2': 'model.mm_projector.2',
        }
        LONGNET_MAPPING = {
            'layers.0': 'LongNet_encoder.layers.0',
            'layers.1': 'LongNet_encoder.layers.1',
            'layer_norm': 'LongNet_encoder.layer_norm'
        }

        try:
            from llava.model import LlavaConfig, LlavaLlamaForCausalLM
        except ImportError:
            raise ImportError(
                'Please install llava with '
                '`pip install git+https://github.com/haotian-liu/LLaVA.git '
                '--no-deps`.')

        assert getattr(self.llm, 'hf_quantizer', None) is None, \
            'This conversion format does not support quantized LLM.'

        # get state_dict
        llm = self.llm
        if self.use_llm_lora:
            llm = self.llm.merge_and_unload()
        llm.config.use_cache = True
        if not fp32:
            print_log('Convert LLM to float16', 'current')
            llm.half()

        assert isinstance(llm, LlamaForCausalLM), \
            'This conversion format only supports LlamaForCausalLM.'
        llm_state_dict = llm.state_dict()

        need_visual_encoder = (not self.freeze_visual_encoder
                               or self.use_visual_encoder_lora)
        visual_encoder = self.visual_encoder
        if self.use_visual_encoder_lora:
            visual_encoder = self.visual_encoder.merge_and_unload()
        assert isinstance(visual_encoder, CLIPVisionModel),\
            'This conversion format only supports CLIPVisionModel.'
        if need_visual_encoder:
            visual_encoder_state_dict = visual_encoder.state_dict()
            visual_encoder_state_dict = convert_state_dict_to_hf(
                visual_encoder_state_dict, VIT_MAPPING)
        else:
            visual_encoder_state_dict = {}

        projector_state_dict = self.projector.state_dict()
        projector_state_dict = convert_state_dict_to_hf(
            projector_state_dict, PROJECTOR_MAPPING)

        LongNet_encoder_state_dict = self.LongNet_encoder.state_dict()
        LongNet_encoder_state_dict = convert_state_dict_to_hf(
            LongNet_encoder_state_dict, LONGNET_MAPPING)

        state_dict = {
            **projector_state_dict,
            **llm_state_dict,
            **visual_encoder_state_dict,
            **LongNet_encoder_state_dict
        }

        # init model
        tokenizer = BUILDER.build(cfg.tokenizer)
        image_processor = BUILDER.build(cfg.image_processor)
        assert isinstance(image_processor, CLIPImageProcessor),\
            'This conversion format only supports CLIPImageProcessor.'

        llava_config_dict = llm.config.__dict__.copy()
        llava_config_dict.update(
            dict(
                image_aspect_ratio='pad',
                mm_hidden_size=visual_encoder.config.hidden_size,
                mm_projector_type=f'mlp{self.projector_depth}x_gelu',
                mm_use_im_patch_token=False,
                mm_use_im_start_end=False,
                mm_vision_select_feature='patch',
                mm_vision_select_layer=self.visual_select_layer,
                mm_vision_tower=visual_encoder.config.name_or_path,
                unfreeze_mm_vision_tower=need_visual_encoder,
                model_type='llava',
                use_cache=True,
                use_mm_proj=True))

        llava_config = LlavaConfig(**llava_config_dict)

        with init_empty_weights():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', message='.*non-meta.*', category=UserWarning)
                model = LlavaLlamaForCausalLM(llava_config)

        model.load_state_dict(state_dict, strict=True, assign=True)

        # save
        print_log(f'Saving to {save_dir}', 'current')

        model.save_pretrained(save_dir, **save_pretrained_kwargs)
        image_processor.save_pretrained(save_dir, **save_pretrained_kwargs)
        tokenizer.save_pretrained(save_dir, **save_pretrained_kwargs)
