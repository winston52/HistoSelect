# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, Sequence
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from xtuner.parallel.sequence import (get_sequence_parallel_world_size,
                                      pad_for_sequence_parallel)
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX


def default_collate_fn(instances: Sequence[Dict],
                       pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
                       return_hf_format: bool = False,
                       use_varlen_attn: bool = False):
    seq_parallel_world_size = get_sequence_parallel_world_size()

    input_ids, labels = [], []
    has_image = any(inst.get('pixel_values') is not None for inst in instances)
    has_weak_labels = any(inst.get('weak_labels') is not None for inst in instances)
    has_question_embedding = any(inst.get('question_embedding') is not None for inst in instances)
    
    has_id = any(inst.get('id') is not None for inst in instances)
    has_organ_name = any(inst.get('organ_name') is not None for inst in instances)
    has_case_name = any(inst.get('case_name') is not None for inst in instances)
    has_question = any(inst.get('question') is not None for inst in instances)
    has_answer = any(inst.get('answer') is not None for inst in instances)
    
    if use_varlen_attn:
        position_ids, cumulative_len = [], []
        assert len(instances) == 1, (
            f'If utilizing varlen attention, the batch size should be'
            f' set to 1, but got {len(instances)}')
        assert not has_image, 'Currently, it is not configured to '
        'accommodate the use of varlen Attention in multimodal training'

    if has_image: pixel_values = []
    if has_weak_labels: weak_labels = []
    if has_question_embedding: question_embeddings = []
    if has_organ_name: organ_names = []
    if has_case_name: case_names = []
    if has_question: questions = []
    if has_answer: answers = []
    if has_id: ids = []

    for example in instances:
        input_ids.append(torch.LongTensor(example['input_ids']))
        labels.append(torch.LongTensor(example['labels']))
        
        if has_organ_name:
            organ_names.append(example.get('organ_name', ''))
        if has_case_name:
            case_names.append(example.get('case_name', ''))
        if has_question:
            questions.append(example.get('question', ''))
        if has_answer:
            answers.append(example.get('answer', ''))
        if has_id:
            ids.append(example.get('id', ''))        
            
        if use_varlen_attn:
            cumulative_len.append(torch.IntTensor(example['cumulative_len']))
            position_ids.append(torch.LongTensor(example['position_ids']))

        if has_image:
            if isinstance(example['pixel_values'], list):
                pixel_values.extend(example['pixel_values'])
            else:
                pixel_values.append(example['pixel_values'])
            
        if has_weak_labels and 'weak_labels' in example and example['weak_labels']:
            item_labels = example['weak_labels']
            if isinstance(item_labels, list):
                weak_labels.extend(item_labels)
            else:
                weak_labels.append(item_labels)
                
        if has_question_embedding:
            question_embeddings.append(example['question_embedding'])                      

    ori_length = [len(ids) for ids in input_ids]
    if len(instances) > 1:
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
        if has_weak_labels and weak_labels:
            weak_labels = pad_sequence(
                weak_labels, batch_first=True, padding_value=IGNORE_INDEX)
    else:
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        if has_weak_labels and weak_labels:
            weak_labels = torch.stack(weak_labels)

    if use_varlen_attn:
        assert input_ids.size(1) % seq_parallel_world_size == 0
        attention_mask = None
        position_ids = torch.stack(position_ids, dim=0)
    else:
        attention_mask = torch.zeros_like(input_ids).bool()
        for i, length in enumerate(ori_length):
            attention_mask[i, :length] = True

        bs, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len).unsqueeze(0).long().repeat(bs, 1)

    if seq_parallel_world_size > 1:
        input_ids = pad_for_sequence_parallel(input_ids, pad_index)
        labels = pad_for_sequence_parallel(labels, IGNORE_INDEX)
        position_ids = pad_for_sequence_parallel(position_ids, 0)
        if attention_mask is not None:
            attention_mask = pad_for_sequence_parallel(attention_mask, 0)
        if has_weak_labels and weak_labels:
            weak_labels = pad_for_sequence_parallel(weak_labels, IGNORE_INDEX)
            
    if use_varlen_attn:
        max_seqlen = (
            cumulative_len[0][1:] -
            cumulative_len[0][:-1]).max().item()
        data_dict = {
            'input_ids': input_ids,
            'cumulative_len': cumulative_len,
            'position_ids': position_ids,
            'labels': labels,
            'max_seqlen': max_seqlen
        }
    else:
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'labels': labels
        }

    if has_image:
        pixel_values = torch.stack(pixel_values)
        data_dict['pixel_values'] = pixel_values

    if has_id:
        data_dict['id'] = ids
    if has_organ_name:
        data_dict['organ_name'] = organ_names
    if has_case_name:
        data_dict['case_name'] = case_names
    if has_question:
        data_dict['question'] = questions
    if has_answer:
        data_dict['answer'] = answers
    if has_weak_labels:
        data_dict['weak_labels'] = weak_labels
        
    if has_question_embedding:
        data_dict['question_embedding'] = torch.stack(question_embeddings)
        
    if instances and 'idx_to_class' in instances[0]:
            data_dict['idx_to_class'] = instances[0]['idx_to_class']

    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': None}
