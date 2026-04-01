import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import h5py

class Whole_Slide_Bag(Dataset):
	def __init__(self,
		file_path,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			roi_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.roi_transforms = img_transforms
		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['imgs']
			self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		with h5py.File(self.file_path, "r") as hdf5_file:
			dset = hdf5_file['imgs']
			for name, value in dset.attrs.items():
				print(name, value)

		print('transformations:', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			img = hdf5_file['imgs'][idx]
			coord = hdf5_file['coords'][idx]
		
		img = Image.fromarray(img)
		img = self.roi_transforms(img)
		return {'img': img, 'coord': coord}


class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		scale_factor,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			img_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.roi_transforms = img_transforms
		self.file_path = file_path
		self.scale_factor = scale_factor
		self.patches_sum = os.listdir(file_path)
			
		self.summary()
			
	def __len__(self):
		return len(self.patches_sum)

	def summary(self):

		print('\nfeature extraction settings')
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		
		patch_name = self.patches_sum[idx].split('/')[-1]
		patch_path = os.path.join(self.file_path, self.patches_sum[idx])
		img = Image.open(patch_path).convert('RGB')
		img = self.roi_transforms(img)
		if len(patch_name.split('_')) == 3:
			x, y, _ = patch_name.split('_')
		elif len(patch_name.split('_')) == 2:
			# remove the extension
			patch_name = patch_name.split('.')[0]
			x, y = patch_name.split('_')
		# x, y, _ = patch_name.split('_')
		coord = (int(x) * self.scale_factor, int(y) * self.scale_factor)
  
		return {'img': img, 'coord': coord}


class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]
