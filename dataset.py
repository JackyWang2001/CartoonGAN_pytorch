import os
import glob
import PIL.Image as Image

import imgaug as ia
import imgaug.augmenters as iaa
import torch.utils.data
import torchvision.transforms as transforms


class MyDataset(torch.utils.data.Dataset):
	"""
	folder structure is:
	-- Dataset folder
		-- real
			-- train
			-- valid
			-- test
		-- STYLE_NAME1
			-- train
			-- valid
		-- STYLE_NAME2
		...
	"""
	def __init__(self, root, style, mode, transform=None):
		"""
		Dataset for CartoonGAN,
		:param root: root path of the dataset
		:param style: real or violet
		:param mode: train / valid / test
		:param transform: if None, apply Augment as default
		"""
		super(MyDataset, self).__init__()
		# list images
		self.root = root
		self.dir = os.path.join(self.root, style, mode)
		self.path_list = glob.glob(os.path.join(self.dir, '*'))
		# get transform
		self.transform = transform
		if self.transform is None:
			self.transform = Augment()

	def __len__(self):
		return len(self.path_list)

	def __getitem__(self, item):
		img = self.dir[item]
		img = Image.open(img)
		self.transform(img)
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
		])
		img = transform(img)
		return img


class Augment(object):
	"""
	Data augmentation for GAN, using Imgaug
	"""
	def __init__(self):
		sometimes = lambda aug: iaa.Sometimes(0.5, aug)
		# Define our sequence of augmentation steps that will be applied to every image
		self.seq = iaa.Sequential(
			[
				# apply the following augmenters to most images
				iaa.Fliplr(0.5),  # horizontally flip 50% of all images
				iaa.Flipud(0.2),  # vertically flip 20% of all images
				# crop images by -5% to 10% of their height/width
				sometimes(iaa.Crop(px=(1, 16), keep_size=False)),
				sometimes(iaa.Affine(
					scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
					# scale images to 80-120% of their size, individually per axis
					translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
					# translate by -20 to +20 percent (per axis)
					rotate=(-45, 45),  # rotate by -45 to +45 degrees
					shear=(-16, 16),  # shear by -16 to +16 degrees
					order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
					cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
					mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
				)),
				iaa.Resize({"height": 256, "width": 256})  # resize
			],
			random_order=False
		)

	def __call__(self, img):
		return self.seq(images=img)