# -*- coding: utf-8 -*-
"""
Title: 
    recxml_to_nii.py

Agenda: 
    A .py script to convert .rec/.xml files to .nii format.

Author: 
    Joseph Plummer - joseph.plummer@cchmc.org 

Creation date: 
    2023-03-13
    
Modification date: 
    
    
"""

# %% Import packages

# Base
from readphilips.file_io import io
import readphilips as rp
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
from glob import glob
plt.style.use('dark_background')

# %% Select folder

# Data folder location
# location = io(
#     initial_dir="I:\\Woods_CPIR_Images\\IRC740H_CF&Non-CF_Bronchiectasis\\IRC740H-001\\").selectDirectory()
location = initial_dir="I:\\Woods_CPIR_Images\\IRC740H_CF&Non-CF_Bronchiectasis\\"

# List to store paths of all *.rec files found
rec_files = []

# Traverse through all directories and subdirectories
for root, dirs, files in os.walk(location):
    # Check if any *.rec files exist in the current directory
    rec_files.extend([os.path.join(root, file) for file in files if file.endswith(".rec")])

print("Number of *.rec files found inside folder: " + str(len(rec_files)))
# %% Load data

for i in range(len(rec_files)):
    try:
        # Load data
        # filename = io(initial_dir=location, f_type="rec files",
        #             ext_type="*.rec").selectFile()
        filename = rec_files[i]
        rec = rp.PhilipsData(filename)
        rec.compute()
        rec_convert = np.array(rec.data)
        rec_convert = np.squeeze(rec_convert)
        print("Data shape = " + str(np.shape(rec_convert)))

        # Estimate affine information
        # TODO: use raw Philips data to get exact values of xyz offsets

        x_res = float(rec.header.get('Resolution X')[0])
        y_res = float(rec.header.get('Resolution Y')[0])
        z_res = np.max(np.array(rec.header.get('Slice'), dtype=float))

        plt.figure(figsize=(12,12))
        plt.imshow(rec_convert[int(z_res/2), ...], cmap="gray")
        plt.title(filename, fontsize=9)
        plt.axis("off")
        plt.show()

        print("x_res = " + str(x_res))
        print("y_res = " + str(y_res))
        print("z_res = " + str(z_res))

        x_voxel = np.array([list(map(float, item.split())) for item in rec.header.get('Pixel Spacing')])[0][0]
        y_voxel = np.array([list(map(float, item.split())) for item in rec.header.get('Pixel Spacing')])[0][0]
        z_voxel = float(rec.header.get('Slice Thickness')[0])

        print("x_voxel = " + str(x_voxel))
        print("y_voxel = " + str(y_voxel))
        print("z_voxel = " + str(z_voxel))

        if x_res == z_res:
            x_scale = x_voxel
            y_scale = y_voxel
            z_scale = x_voxel # force isotropic
        else:
            x_scale = x_voxel
            y_scale = y_voxel
            z_scale = z_voxel

        # Save images as Nifti files

        if np.ndim(rec_convert) == 4:
            rec_convert_temp = np.array(rec_convert)
            rec_convert = np.rot90(rec_convert_temp, k=1,  axes=[1, 3])
            rec_convert = np.flip(rec_convert, axis=1)
            rec_convert = np.flip(rec_convert, axis=3)
        elif np.ndim(rec_convert) == 3:
            rec_convert_temp = np.array(rec_convert)
            rec_convert = np.rot90(rec_convert_temp, k=1,  axes=[0, 2])
            rec_convert = np.flip(rec_convert, axis=0)
            rec_convert = np.flip(rec_convert, axis=2)

        # Build an array using matrix multiplication
        scaling_affine = np.array([[-x_scale, 0, 0, 0],
                                [0, -y_scale, 0, 0],
                                [0, 0, -z_scale, 0],
                                [0, 0, 0, -1]])

        # Rotate gamma radians about axis i
        cos_gamma = np.cos(0)
        sin_gamma = np.sin(0)
        rotation_affine_1 = np.array([[1, 0, 0, 0],
                                    [0, cos_gamma, -sin_gamma,  0],
                                    [0, sin_gamma, cos_gamma, 0],
                                    [0, 0, 0, 1]])
        cos_gamma = np.cos(0)
        sin_gamma = np.sin(0)
        rotation_affine_2 = np.array([[cos_gamma, 0, sin_gamma, 0],
                                    [0, 1, 0, 0],
                                    [-sin_gamma, 0, cos_gamma, 0],
                                    [0, 0, 0, 1]])
        cos_gamma = np.cos(0)
        sin_gamma = np.sin(0)
        rotation_affine_3 = np.array([[cos_gamma, -sin_gamma, 0, 0],
                                    [sin_gamma, cos_gamma, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        rotation_affine = rotation_affine_1.dot(
            rotation_affine_2.dot(rotation_affine_3))

        # Apply translation
        translation_affine = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])

        # Multiply matrices together
        aff = translation_affine.dot(rotation_affine.dot(scaling_affine))

        # Make path to save image results

        save_filename = filename[:-4] + ".nii.gz"

        img = rec_convert
        ni_img = nib.Nifti1Image(abs(img), affine=aff)
        nib.save(ni_img, save_filename)
    except:
        print("ERROR in filename: " + filename)
        print("Could not utilize the *.rec image inside " + filename)
        

# %%
