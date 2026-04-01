import time
import os
import argparse
import pdb
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
from tqdm import tqdm

import numpy as np

from utils_clam.file_utils import save_hdf5
# from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from dataset_modules.dataset_patch import Whole_Slide_Bag, Whole_Slide_Bag_FP
from models import get_encoder



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(output_path, loader, model, verbose = 0):
	"""
	args:
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		verbose: level of feedback
	"""
	if verbose > 0:
		print(f'processing a total of {len(loader)} batches'.format(len(loader)))

	mode = 'w'
	for count, data in enumerate(tqdm(loader)):
		with torch.inference_mode():	
			batch = data['img']
			# coords = data['coord'].numpy().astype(np.int32)
			x_coords = data['coord'][0].cpu().numpy().astype(np.int32)
			y_coords = data['coord'][1].cpu().numpy().astype(np.int32)
			coords = np.stack((x_coords, y_coords), axis=1)

			batch = batch.to(device, non_blocking=True)
			features = model(batch)
			features = features.cpu().numpy().astype(np.float32)

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--patch_dir', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--model_name', type=str, default='resnet50_trunc', choices=['resnet50_trunc', 'uni_v1', 'conch_v1', 'dino_gbm', 'dino_brca','dino_default', 'uni'])
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
parser.add_argument('--target_magnification', type=int, default=20)
parser.add_argument('--original_magnification', type=int, default=40)
args = parser.parse_args()


if __name__ == '__main__':
	print('initializing dataset')

	bags_dataset = os.listdir(args.patch_dir)

	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))
	print('dest_files', len(dest_files), dest_files)
	print(os.path.join(args.feat_dir, 'pt_files'))

	model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
			
	_ = model.eval()
	model = model.to(device)
	total = len(bags_dataset)

	loader_kwargs = {'num_workers': 32, 'pin_memory': True} if device.type == "cuda" else {}
 
	print("num_workers: ", loader_kwargs['num_workers'])

	# skip those already generated
	feat_path = os.path.join(args.feat_dir, 'pt_files')

	for bag_candidate_idx in tqdm(range(total)):
		
     
		slide_id = bags_dataset[bag_candidate_idx]
		patch_path = os.path.join(args.patch_dir, slide_id)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		# if slide_id already in the feature folder, then skip it
		check_exist_feat_path = os.path.join(feat_path, slide_id+'.pt')
		# Check if the slide_id already exists in the 'pt_files' folder
		if os.path.exists(check_exist_feat_path):
			print(f"Skipping {slide_id} as it already exists in the folder.")
			continue  # Skip this iteration


		# skip those already generated
		# if not args.no_auto_skip and slide_id+'.pt' in dest_files:
		# print(slide_id+'.pt')
		if slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		output_path = os.path.join(args.feat_dir, 'h5_files', slide_id + '.h5')
		time_start = time.time()

		scale_factor = int(args.original_magnification / args.target_magnification)
  
		dataset = Whole_Slide_Bag_FP(file_path=patch_path,
									 scale_factor=scale_factor,
									 img_transforms=img_transforms)

		loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
		output_file_path = compute_w_loader(output_path, loader = loader, model = model, verbose = 1)

		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))

		# in case some WSI has no patches, we will skip those patches
		if not os.path.exists(output_file_path):
			continue

		with h5py.File(output_file_path, "r") as file:
			features = file['features'][:]
			print('features size: ', features.shape)
			print('coordinates size: ', file['coords'].shape)

		features = torch.from_numpy(features)
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', slide_id + '.pt'))