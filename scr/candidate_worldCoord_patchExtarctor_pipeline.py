from cvseg_utils import*
import warnings
import os
import logging
import random
import cv2
import json
import torch
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk
from monai.transforms import Compose, ScaleIntensityRanged
random.seed(200)
np.random.seed(200)
from datetime import datetime

def create_folder_if_not_exists(folder_path):
    import os
    # Check if the folder exists
    if not os.path.exists(folder_path):
        # If the folder doesn't exist, create it
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")

def nifti_patche_extractor_for_worldCoord_main():
    parser = argparse.ArgumentParser(description='Nodule segmentation and feature extraction from CT images.')
    parser.add_argument('--raw_data_path', type=str, required=True, help='Path to raw CT images')
    parser.add_argument('--csv_save_path', type=str, required=True, help='Path to save the CSV files')
    parser.add_argument('--dataset_csv',   type=str, required=True, help='Path to the dataset CSV')
    parser.add_argument('--dataset_name',  type=str, default='DLCS24', help='Dataset to use')
    # Allow multiple column names as input arguments
    parser.add_argument('--nifti_clm_name',   type=str, required=True, help='name to the nifti column name')
    parser.add_argument('--unique_Annotation_id', type=str, help='Column for unique annotation ID')
    parser.add_argument('--Malignant_lbl', type=str, help='Column name for malignancy labels')
    parser.add_argument('--coordX', type=str, required=True, help='Column name for X coordinate')
    parser.add_argument('--coordY', type=str, required=True, help='Column name for Y coordinate')
    parser.add_argument('--coordZ', type=str, required=True, help='Column name for Z coordinate')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[64, 64, 64], help="Patch size as three integers, e.g., --patch_size 64 64 64")
    # Normalization (4 values)
    parser.add_argument('--normalization', type=float, nargs=4, default=[-1000, 500.0, 0.0, 1.0],help="Normalization values as four floats: A_min A_max B_min B_max")
    # Clip (Boolean from string input)
    parser.add_argument('--clip', type=str, choices=["True", "False"], default="False",help="Enable or disable clipping (True/False). Default is False.")
    parser.add_argument('--save_nifti_path', type=str, help='Path to save the nifti files')

    args           = parser.parse_args()
    raw_data_path  = args.raw_data_path
    csv_save_path  = args.csv_save_path
    dataset_csv    = args.dataset_csv

    create_folder_if_not_exists(csv_save_path)
    create_folder_if_not_exists(args.save_nifti_path)
    # Extract normalization values
    A_min, A_max, B_min, B_max = args.normalization
    # Convert clip argument to boolean
    CLIP = args.clip == "True"
    output_csv = csv_save_path + f'CandidateSeg_{args.dataset_name}_patch{args.patch_size[0]}x{args.patch_size[1]}y{args.patch_size[2]}z.csv'
    Erroroutput_csv = csv_save_path + f'CandidateSeg_{args.dataset_name}_patch{args.patch_size[0]}x{args.patch_size[1]}y{args.patch_size[2]}z_Error.csv'

    # Derive the log file name from the output CSV file
    log_file = output_csv.replace('.csv', '.log')

    # Configure logging
    logging.basicConfig(
        filename=log_file, 
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logging.info(f"Output CSV File: {output_csv}")
    logging.info(f"Error CSV File: {Erroroutput_csv}")
    logging.info(f"Log File Created: {log_file}")
    logging.info("File names generated successfully.")


    ###----input CSV
    df                      = pd.read_csv(dataset_csv)
    final_dect              = df[args.nifti_clm_name].unique()
    output_df               = pd.DataFrame()
    Error_ids = []
    for dictonary_list_i in tqdm(range(0,len(final_dect)), desc='Processing CTs'):
        try:
            logging.info(f"---Loading---: {dictonary_list_i+1}")
            #print(make_bold('|' + '-'*30 + ' No={} '.format(dictonary_list_i+1) + '-'*30 + '|'))
            #print('\n')

            desired_value      = final_dect[dictonary_list_i]
            filtered_df        = df[df[args.nifti_clm_name] == desired_value]
            example_dictionary = filtered_df.reset_index()

            logging.info(f"Loading the Image:{example_dictionary[args.nifti_clm_name][0]}")
            logging.info(f"Number of Annotations:{len(example_dictionary)}")
            #print('Loading the Image: {}'.format(example_dictionary[args.nifti_clm_name][0]))
            #print('Number of Annotations: {}'.format(len(example_dictionary)))
            ct_nifti_path = raw_data_path + example_dictionary[args.nifti_clm_name][0]
            ct_image      = sitk.ReadImage(ct_nifti_path)
            ct_array      = sitk.GetArrayFromImage(ct_image)


            torch_image         = torch.from_numpy(ct_array)
            temp_torch_image    = {"image": torch_image}
            intensity_transform = Compose([ScaleIntensityRanged(keys=["image"], a_min=A_min, a_max=A_max, b_min=B_min, b_max=B_max, clip=CLIP),])
            transformed_image   = intensity_transform(temp_torch_image)
            numpyImage          = transformed_image["image"].numpy()

            for Which_box_to_use in range(0,len(example_dictionary)):

                #print('-----------------------------------------------------------------------------------------------')
                if args.unique_Annotation_id in example_dictionary.columns:
                    annotation_id = example_dictionary[args.unique_Annotation_id][Which_box_to_use]
                else:
                    # Generate an ID using the image name (without extension) and an index
                    image_name = example_dictionary[args.nifti_clm_name][0].split('.nii')[0]
                    annotation_id = f"{image_name}_candidate_{Which_box_to_use+1}"

                
                
                worldCoord = np.asarray([float(example_dictionary[args.coordX][Which_box_to_use]), float(example_dictionary[args.coordY][Which_box_to_use]), float(example_dictionary[args.coordZ][Which_box_to_use])])
                voxelCoord = ct_image.TransformPhysicalPointToIndex(worldCoord)
                # Access individual values
                w = args.patch_size[0]
                h = args.patch_size[1]
                d = args.patch_size[2]
                start_x, end_x = int(voxelCoord[0] - w/2), int(voxelCoord[0] + w/2)
                start_y, end_y = int(voxelCoord[1] - h/2), int(voxelCoord[1] + h/2)
                start_z, end_z = int(voxelCoord[2] - d/2), int(voxelCoord[2] + d/2)
                X, Y, Z = int(voxelCoord[0]), int(voxelCoord[1]), int(voxelCoord[2])
                numpy_to_save_np = numpyImage[max(start_z,0):end_z, max(start_y,0):end_y, max(start_x,0):end_x]

                # Pad if necessary
                if np.any(numpy_to_save_np.shape != (d, h, w)):
                    dZ, dY, dX = numpyImage.shape
                    numpy_to_save_np = np.pad(numpy_to_save_np, ((max(d // 2 - Z, 0), d // 2 - min(dZ - Z, d // 2)),
                                                                 (max(h // 2 - Y, 0), h // 2 - min(dY - Y, h // 2)),
                                                                 (max(w // 2 - X, 0), w // 2 - min(dX - X, w // 2))), mode="constant", constant_values=0.)                

                #--- Segmentation---#
                patch_image       = sitk.GetImageFromArray(numpy_to_save_np)
                patch_image.SetSpacing(ct_image.GetSpacing())
                patch_image.SetDirection(ct_image.GetDirection())
                patch_image.SetOrigin(ct_image.GetOrigin())
                if args.Malignant_lbl in example_dictionary.columns:
                    feature_row = pd.DataFrame({args.nifti_clm_name: [example_dictionary[args.nifti_clm_name][0]],'candidateID': [annotation_id],args.Malignant_lbl: [example_dictionary[args.Malignant_lbl][Which_box_to_use]]})
                else:
                    feature_row = pd.DataFrame({args.nifti_clm_name: [example_dictionary[args.nifti_clm_name][0]],'candidateID': [annotation_id]})
                feature_row[args.coordX] = example_dictionary[args.coordX][Which_box_to_use]
                feature_row[args.coordY] = example_dictionary[args.coordY][Which_box_to_use]
                feature_row[args.coordZ] = example_dictionary[args.coordZ][Which_box_to_use]
                # Save 
                output_nifti_path = os.path.join(args.save_nifti_path, f"{annotation_id}.nii.gz")
                # Ensure the directory exists before writing the file
                output_nifti_dir = os.path.dirname(output_nifti_path)
                os.makedirs(output_nifti_dir, exist_ok=True)         # Creates the directory if it doesn't exist
                sitk.WriteImage(patch_image, output_nifti_path)
                # Append the row to the output DataFrame
                output_df = pd.concat([output_df, feature_row], ignore_index=True)
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            print(f" Error occured: {e}")
            Error_ids.append(final_dect[dictonary_list_i])
            pass
    
    # Save the output DataFrame to a CSV file
    output_df.to_csv(output_csv, index=False,encoding='utf-8')
    print("completed and saved to {}".format(output_csv))
    Erroroutput_df = pd.DataFrame(list(Error_ids),columns=[args.nifti_clm_name])
    Erroroutput_df.to_csv(Erroroutput_csv, index=False,encoding='utf-8')
    print("completed and saved Error to {}".format(Erroroutput_csv))


if __name__ == "__main__":
    nifti_patche_extractor_for_worldCoord_main()
