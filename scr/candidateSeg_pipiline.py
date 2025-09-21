from cvseg_utils import *
import warnings
warnings.filterwarnings("ignore", message="GLCM is symmetrical, therefore Sum Average = 2 * Joint Average")
import os
import logging
from datetime import datetime

def seg_main():
    parser = argparse.ArgumentParser(description='Nodule segmentation and feature extraction from CT images.')
    parser.add_argument('--raw_data_path', type=str, required=True, help='Path to raw CT images')
    parser.add_argument('--csv_save_path', type=str, required=True, help='Path to save the CSV files')
    parser.add_argument('--dataset_csv', type=str, required=True, help='Path to the dataset CSV')
    parser.add_argument('--nifti_clm_name', type=str, required=True, help='name to the nifti column name')
    parser.add_argument('--unique_Annotation_id', type=str, help='Column for unique annotation ID')
    parser.add_argument('--Malignant_lbl', type=str, required=True, help='Column name for malignancy labels')
    parser.add_argument('--coordX', type=str, required=True, help='Column name for X coordinate')
    parser.add_argument('--coordY', type=str, required=True, help='Column name for Y coordinate')
    parser.add_argument('--coordZ', type=str, required=True, help='Column name for Z coordinate')
    parser.add_argument('--w', type=str, required=True, help='Column name for width')
    parser.add_argument('--h', type=str, required=True, help='Column name for height')
    parser.add_argument('--d', type=str, required=True, help='Column name for depth')
    parser.add_argument('--seg_alg', type=str, default='gmm', choices=['gmm', 'knn', 'fcm', 'otsu'], help='Segmentation algorithm to use')
    parser.add_argument('--dataset_name', type=str, default='DLCS24', help='Dataset to use')
    parser.add_argument('--expansion_mm', type=float, default=1.0, help='Expansion in mm')
    parser.add_argument('--use_expand', action='store_true', help='Use expansion if set')
    parser.add_argument('--params_json', type=str, required=True, help="Path to JSON file with radiomics parameters")
    parser.add_argument('--save_the_generated_mask', action='store_true', help='Save generated segmentation mask')
    parser.add_argument('--save_nifti_path', type=str, help='Path to save the nifti files')

    args = parser.parse_args()

    df = pd.read_csv(args.dataset_csv)
    final_dect = df[args.nifti_clm_name].unique()

    for dictonary_list_i, ct_filename in enumerate(final_dect):
        try:
            filtered_df = df[df[args.nifti_clm_name] == ct_filename].reset_index()
            ct_nifti_path = os.path.join(args.raw_data_path, ct_filename)
            ct_image = sitk.ReadImage(ct_nifti_path)
            ct_array = sitk.GetArrayFromImage(ct_image)
            spacing = ct_image.GetSpacing()

            full_mask_array = np.zeros_like(ct_array, dtype=np.uint8)

            for idx, row in filtered_df.iterrows():
                worldCoord = np.asarray([row[args.coordX], row[args.coordY], row[args.coordZ]])
                voxelCoord = ct_image.TransformPhysicalPointToIndex(worldCoord.tolist())
                spacing = ct_image.GetSpacing()
                w = int(row[args.w] / spacing[0])
                h = int(row[args.h] / spacing[1])
                d = int(row[args.d] / spacing[2])
                bbox_center = [voxelCoord[2], voxelCoord[1], voxelCoord[0]]
                bbox_whd = [d, h, w]

                if args.seg_alg == 'gmm':
                    mask_image_array = segment_nodule_gmm(ct_array, bbox_center, bbox_whd)
                elif args.seg_alg == 'knn':
                    mask_image_array = segment_nodule_kmeans(ct_array, bbox_center, bbox_whd)
                elif args.seg_alg == 'fcm':
                    mask_image_array = segment_nodule_fcm(ct_array, bbox_center, bbox_whd)
                elif args.seg_alg == 'otsu':
                    mask_image_array = segment_nodule_otsu(ct_array, bbox_center, bbox_whd)

                if args.use_expand:
                    mask_image_array = expand_mask_by_distance(mask_image_array, spacing=spacing, expansion_mm=args.expansion_mm)
                    #print("Segmented mask sum:", np.sum(mask_image_array))


                full_mask_array[mask_image_array==1] = 1
                


            if args.save_the_generated_mask:
                print("Segmented mask sum:", np.sum(full_mask_array))
                combined_mask_image = sitk.GetImageFromArray(full_mask_array)
                combined_mask_image.SetSpacing(ct_image.GetSpacing())
                combined_mask_image.SetDirection(ct_image.GetDirection())
                combined_mask_image.SetOrigin(ct_image.GetOrigin())
                mask_filename = ct_filename.split('.nii')[0]+"_mask.nii.gz"
                sitk.WriteImage(combined_mask_image, os.path.join(args.save_nifti_path, mask_filename))
                print(f"Saved {mask_filename}")
        except Exception as e:
            print(f"Error processing {ct_filename}: {e}")

if __name__ == "__main__":
    seg_main()
