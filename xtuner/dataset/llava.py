# Copyright (c) OpenMMLab. All rights reserved.
import json
import logging
import os

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from PIL import Image
from torch.utils.data import Dataset

from xtuner.registry import BUILDER
from .huggingface import process_hf_dataset
from .utils import expand2square

import pandas as pd
import numpy as np
import h5py

def load_jsonl(json_file):
    with open(json_file) as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data


class LLaVADataset(Dataset):

    def __init__(self,
                 image_folder,
                 image_path_list,
                 per_image_length,
                 data_path=None,
                 tokenizer=None,
                 offline_processed_text_folder=None,
                 max_dataset_length=None,
                 dataset_map_fn=None,
                 template_map_fn=None,
                 max_length=2048,
                 pad_image_to_square=False,
                 sample_num=10240, 
                 identifier = '',
                 image_feature_prefix='', 
                 image_feature_suffix='.h5',
                 weak_label_prefix='',
                 weak_label_suffix='.h5',
                 question_embedding_path=''):
        super().__init__()

        self.sample_num = sample_num
        self.per_image_length = per_image_length
        assert offline_processed_text_folder or (data_path and tokenizer)
        if offline_processed_text_folder and data_path:
            print_log(
                'Both `offline_processed_text_folder` and '
                '`data_path` are set, and we load dataset from'
                '`offline_processed_text_folder` '
                f'({offline_processed_text_folder})',
                logger='current',
                level=logging.WARNING)

        if offline_processed_text_folder is not None:
            self.text_data = load_from_disk(offline_processed_text_folder)
        else:
            if data_path.endswith('.json'):
                json_data = json.load(open(data_path))
            elif data_path.endswith('.jsonl'):
                json_data = load_jsonl(data_path)
            else:
                raise NotImplementedError

            for idx in range(len(json_data)):
                if isinstance(json_data[idx]['id'], int):
                    json_data[idx]['id'] = str(json_data[idx]['id'])
            json_data = DatasetDict({'train': HFDataset.from_list(json_data)})
            self.text_data = process_hf_dataset(
                dataset=json_data,
                tokenizer=tokenizer,
                max_length=max_length,
                dataset_map_fn=dataset_map_fn,
                template_map_fn=template_map_fn,
                split='train',
                max_dataset_length=max_dataset_length,
                remove_unused_columns=False,
                pack_to_max_length=False,
                with_image_token=True,
                per_image_length=self.per_image_length)

        self.image_folder = image_folder
        self.image_path_list = image_path_list
        self.pad_image_to_square = pad_image_to_square
        self.image_feature_prefix = image_feature_prefix
        
        image_feature_suffix = image_feature_suffix
        if image_feature_suffix not in ['.csv', '.pt', '.h5']:
            raise ValueError(
                f'Unsupported image feature suffix: {image_feature_suffix}. '
                'Supported suffixes are: .csv, .pt, .h5')
        self.image_feature_suffix = image_feature_suffix
        
        self.identifier = identifier
        # Store weak label attributes.
        self.weak_label_prefix = weak_label_prefix
        self.weak_label_suffix = weak_label_suffix
        
        # Load question embeddings if the path is provided
        self.question_embedding_path = question_embedding_path
        if self.question_embedding_path and os.path.exists(self.question_embedding_path):
            print_log(
                f'Loading question embeddings from {self.question_embedding_path}...',
                logger='current')
            self.question_embeddings = torch.load(self.question_embedding_path, map_location='cpu')
            print_log('Successfully loaded question embeddings.', logger='current')
        else:
            self.question_embeddings = None
            print_log(
                'Question embedding path not provided or file does not exist. Skipping.',
                logger='current',
                level=logging.WARNING)
        
        self.classes = [
            "malignant tissue", "benign tissue", "stroma", "lymphocytes",
            "necrosis", "adipose tissue", "tissue artifact", "blood vessel",
            "extracellular mucin", "nerve", "hemorrhage", "smooth muscle", "plasma cells"
        ]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.classes)}
        
    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.text_data:
            cur_len = len(data_dict['input_ids'])
            image = data_dict.get('image', None)
            if image is None:
                cur_len = -cur_len
            else:
                if isinstance(image, str):
                    n_images = 1
                else:
                    n_images = len(image)
                cur_len = cur_len - n_images + self.per_image_length * n_images
            length_list.append(cur_len)
        return length_list

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, index):
        
        data_dict = self.text_data[index]
        
        # --- Part 1: Get the question and answer ---
        conversations = data_dict.get('conversations', [])
        question = ''
        answer = ''
        for turn in conversations:
            if turn.get('from') == 'human':
                question = turn.get('value', '').replace('<image>\n', '').strip()
            elif turn.get('from') == 'gpt':
                answer = turn.get('value', '').strip()
        
        data_dict['question'] = question
        data_dict['answer'] = answer
        
        # --- Process image-related data if it exists ---
        if data_dict.get('image', None) is not None:
            image_list = data_dict['image']
            if isinstance(image_list, str):
                image_list = [image_list]
            
            images = []
            if self.image_feature_suffix == '.h5':
                weak_labels = []

            for image_file in image_list:
                # --- Part 2: Get the case name and organ name ---
                organ_name = image_file.split('/')[0].split('-')[1].lower()
                case_name = os.path.splitext(os.path.basename(image_file))[0]
                data_dict['organ_name'] = organ_name
                data_dict['case_name'] = case_name
                
                # --- Part 3: Get the image features ---
                # Construct the full path to the feature file
                if self.image_feature_suffix == ".pt":
                    train_image_file = os.path.join(self.image_feature_prefix, str(organ_name) + self.identifier, 
                                                    "pt_files", case_name + self.image_feature_suffix)
                elif self.image_feature_suffix == ".csv":
                    train_image_file = os.path.join(self.image_feature_prefix, str(organ_name) + self.identifier, 
                                                    "csv_files", case_name + self.image_feature_suffix)
                elif self.image_feature_suffix == '.h5':
                     train_image_file = os.path.join(self.image_feature_prefix, str(organ_name) + self.identifier,
                                                    "h5_files", case_name + self.image_feature_suffix)

                # Load and sample features based on file type
                if train_image_file.endswith('.csv'):
                    image = pd.read_csv(train_image_file).iloc[:, :512]
                    total_rows = image.shape[0]
                    if total_rows > self.sample_num:
                        indices = np.linspace(0, total_rows - 1, self.sample_num, dtype=int)
                        image = image.iloc[indices]
                    image = torch.from_numpy(image.to_numpy())
                
                elif train_image_file.endswith('.pt'):
                    image = torch.load(train_image_file, map_location='cpu')
                    total_rows = image.shape[0]
                    if total_rows > self.sample_num:
                        indices = np.linspace(0, total_rows - 1, self.sample_num, dtype=int)
                        image = image[indices]

                elif train_image_file.endswith('.h5'):
                    with h5py.File(train_image_file, 'r') as h5_file:
                        features_list = [h5_file[key][()] for key in h5_file.keys()]
                    image = features_list[1] # Assuming features are at index 1

                    # --- Part 4: Get weak labels if weak_label_prefix is provided ---
                    if self.weak_label_prefix: # Check if prefix is not an empty string
                        weak_label_file = os.path.join(
                            self.weak_label_prefix, str(organ_name),
                            "h5_files", case_name + self.weak_label_suffix)
                        
                        with h5py.File(weak_label_file, 'r') as h5_label_file:
                            label_list = [h5_label_file[key][()] for key in h5_label_file.keys()]
                        
                        assert image.shape[0] == label_list[1].shape[0], \
                            f"Mismatch in counts for {case_name}: Features {image.shape[0]}, Labels {label_list[1].shape[0]}"

                        byte_labels = label_list[1] # Assuming labels are at index 1
                        string_labels = [label.decode('utf-8') for label in byte_labels]
                        weak_label_indices = [self.class_to_idx[label] for label in string_labels]

                        # Sample both features and labels together
                        total_rows = image.shape[0]
                        if total_rows > self.sample_num:
                            indices = np.linspace(0, total_rows - 1, self.sample_num, dtype=int)
                            image = image[indices]
                            weak_label_indices = [weak_label_indices[i] for i in indices]
                        
                        weak_label = torch.tensor(weak_label_indices, dtype=torch.long)
                        weak_labels.append(weak_label)
                        data_dict['idx_to_class'] = self.idx_to_class
                    
                    else: # If weak_label_prefix is empty, just sample the image features
                        total_rows = image.shape[0]
                        if total_rows > self.sample_num:
                            indices = np.linspace(0, total_rows - 1, self.sample_num, dtype=int)
                            image = image[indices]
                    
                    image = torch.from_numpy(image)
                
                images.append(image)

            # Assign the final lists to the data dictionary
            data_dict['pixel_values'] = images
            if self.image_feature_suffix == '.h5':
                data_dict['weak_labels'] = weak_labels
        
        # --- Part 5: Get the question embedding ---
        if self.question_embeddings is not None:
            sample_id = data_dict['id']
            question_embedding = self.question_embeddings.get(sample_id)
            if question_embedding is not None:
                data_dict['question_embedding'] = question_embedding
            else:
                print_log(
                    f"Warning: Question embedding not found for id: {sample_id}",
                    logger='current',
                    level=logging.WARNING)
                        
        return data_dict
