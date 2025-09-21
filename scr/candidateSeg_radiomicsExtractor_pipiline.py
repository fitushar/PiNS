from cvseg_utils import*
import warnings
warnings.filterwarnings("ignore", message="GLCM is symmetrical, therefore Sum Average = 2 * Joint Average")
import os
import logging
from datetime import datetime


def seg_pyradiomics_main():

    parser = argparse.ArgumentParser(description='Nodule segmentation and feature extraction from CT images.')
    parser.add_argument('--raw_data_path', type=str, required=True, help='Path to raw CT images')
    parser.add_argument('--csv_save_path', type=str, required=True, help='Path to save the CSV files')
    parser.add_argument('--dataset_csv',   type=str, required=True, help='Path to the dataset CSV')

    # Allow multiple column names as input arguments
    parser.add_argument('--nifti_clm_name',   type=str, required=True, help='name to the nifti column name')
    parser.add_argument('--unique_Annotation_id', type=str, help='Column for unique annotation ID')
    parser.add_argument('--Malignant_lbl', type=str, help='Column name for malignancy labels')
    parser.add_argument('--coordX', type=str, required=True, help='Column name for X coordinate')
    parser.add_argument('--coordY', type=str, required=True, help='Column name for Y coordinate')
    parser.add_argument('--coordZ', type=str, required=True, help='Column name for Z coordinate')
    parser.add_argument('--w', type=str, required=True, help='Column name for width')
    parser.add_argument('--h', type=str, required=True, help='Column name for height')
    parser.add_argument('--d', type=str, required=True, help='Column name for depth')

    parser.add_argument('--seg_alg',       type=str, default='gmm',    choices=['gmm', 'knn', 'fcm', 'otsu'], help='Segmentation algorithm to use')
    parser.add_argument('--dataset_name',  type=str, default='DLCS24', help='Dataset to use')
    parser.add_argument('--expansion_mm',  type=float, default=1.0, help='Expansion in mm')
    parser.add_argument('--use_expand',           action='store_true', help='Use expansion if set')
    parser.add_argument('--extract_radiomics',    action='store_true', help='extarct Radiomics if set')
    parser.add_argument('--params_json',          type=str, required=True, help="Path to JSON file with radiomics parameters")
    parser.add_argument('--save_the_generated_mask',  action='store_true', help='Use expansion if set')
    parser.add_argument('--save_nifti_path', type=str, help='Path to save the nifti files')

    args           = parser.parse_args()
    raw_data_path  = args.raw_data_path
    csv_save_path  = args.csv_save_path
    dataset_csv    = args.dataset_csv
    seg_alg        = args.seg_alg


    if args.use_expand:
        if args.extract_radiomics:
            output_csv = csv_save_path + f'PyRadiomics_CandidateSeg_{args.dataset_name}_{seg_alg}_expand_{args.expansion_mm}mm.csv'
            Erroroutput_csv = csv_save_path + f'PyRadiomics_CandidateSeg_{args.dataset_name}_{seg_alg}_expand_{args.expansion_mm}mm_Error.csv'
        else:
            output_csv = csv_save_path + f'CandidateSeg_{args.dataset_name}_{seg_alg}_expand_{args.expansion_mm}mm.csv'
            Erroroutput_csv = csv_save_path + f'CandidateSeg_{args.dataset_name}_{seg_alg}_expand_{args.expansion_mm}mm_Error.csv'
    else:
        if args.extract_radiomics:
            output_csv = csv_save_path + f'PyRadiomics_CandidateSeg_{args.dataset_name}_{seg_alg}.csv'
            Erroroutput_csv = csv_save_path + f'PyRadiomics_CandidateSeg_{args.dataset_name}_{seg_alg}_Error.csv'
        else:
            output_csv = csv_save_path + f'CandidateSeg_{args.dataset_name}_{seg_alg}.csv'
            Erroroutput_csv = csv_save_path + f'CandidateSeg_{args.dataset_name}_{seg_alg}_Error.csv'



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
    # Initialize the feature extractor
    with open(args.params_json, 'r') as f:
        params = json.load(f)

    interpolator_map = {"sitkBSpline": sitk.sitkBSpline,"sitkNearestNeighbor": sitk.sitkNearestNeighbor}
    params["interpolator"] = interpolator_map.get(params["interpolator"], sitk.sitkBSpline)
    params["labelInterpolator"] = interpolator_map.get(params["labelInterpolator"], sitk.sitkNearestNeighbor)

    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    # Prepare the output CSV
    output_df               = pd.DataFrame()
    Error_ids = []
    for dictonary_list_i in range(0,len(final_dect)):
        try:
            logging.info(f"---Loading---: {dictonary_list_i+1}")
            print(make_bold('|' + '-'*30 + ' No={} '.format(dictonary_list_i+1) + '-'*30 + '|'))
            print('\n')

            desired_value      = final_dect[dictonary_list_i]
            filtered_df        = df[df[args.nifti_clm_name] == desired_value]
            example_dictionary = filtered_df.reset_index()

            logging.info(f"Loading the Image:{example_dictionary[args.nifti_clm_name][0]}")
            logging.info(f"Number of Annotations:{len(example_dictionary)}")
            print('Loading the Image: {}'.format(example_dictionary[args.nifti_clm_name][0]))
            print('Number of Annotations: {}'.format(len(example_dictionary)))
            ct_nifti_path = raw_data_path + example_dictionary[args.nifti_clm_name][0]
            ct_image      = sitk.ReadImage(ct_nifti_path)
            ct_array      = sitk.GetArrayFromImage(ct_image)

            for Which_box_to_use in range(0,len(example_dictionary)):

                print('-----------------------------------------------------------------------------------------------')

                if args.unique_Annotation_id in example_dictionary.columns:
                    annotation_id = example_dictionary[args.unique_Annotation_id][Which_box_to_use]
                else:
                    # Generate an ID using the image name (without extension) and an index
                    image_name = example_dictionary[args.nifti_clm_name][0].split('.nii')[0]
                    annotation_id = f"{image_name}_annotation_{Which_box_to_use+1}"
                if args.Malignant_lbl in example_dictionary.columns:
                   annotation_lbl       = example_dictionary[args.Malignant_lbl][Which_box_to_use]
                print('Annotation-ID = {}'.format(annotation_id))
                worldCoord = np.asarray([float(example_dictionary[args.coordX][Which_box_to_use]), float(example_dictionary[args.coordY][Which_box_to_use]), float(example_dictionary[args.coordZ][Which_box_to_use])])
                voxelCoord = ct_image.TransformPhysicalPointToIndex(worldCoord)
                voxel_coords = voxelCoord
                print('WorldCoord  CCC (x,y,z) = {}'.format(worldCoord))
                print('VoxelCoord CCC (x,y,z) = {}'.format(voxelCoord))

                whd_worldCoord = np.asarray([float(example_dictionary[args.w][Which_box_to_use]), float(example_dictionary[args.h][Which_box_to_use]), float(example_dictionary[args.d][Which_box_to_use])])
                spacing        = ct_image.GetSpacing()
                w = int(whd_worldCoord[0] / spacing[0])
                h = int(whd_worldCoord[1] / spacing[1])
                d = int(whd_worldCoord[2] / spacing[2])
                whd_voxelCoord = [w, h, d]
                print('WorldCoord (w,h,d) = {}'.format(whd_worldCoord))
                print('VoxelCoord (w,h,d) = {}'.format(whd_voxelCoord))

                # Define bounding box
                center_index   = voxelCoord
                size_voxel     = np.array(whd_voxelCoord) / 2
                bbox_center    = [voxel_coords[2],voxel_coords[1],voxel_coords[0]]
                bbox_whd       = [d,h,w]


                #--Image-processing algorithms for segmentations
                if seg_alg=='gmm':
                    mask_image_array = segment_nodule_gmm(ct_array, bbox_center, bbox_whd)
                if seg_alg=='knn':
                    mask_image_array = segment_nodule_kmeans(ct_array, bbox_center, bbox_whd)
                if seg_alg=='fcm':
                    mask_image_array = segment_nodule_fcm(ct_array, bbox_center, bbox_whd)
                if seg_alg=='otsu':
                    mask_image_array = segment_nodule_otsu(ct_array, bbox_center, bbox_whd)

                if args.use_expand:
                    mask_image_array = expand_mask_by_distance(mask_image_array, spacing=spacing, expansion_mm=args.expansion_mm)

                #--- Segmentation---#
                mask_image       = sitk.GetImageFromArray(mask_image_array)
                mask_image.SetSpacing(ct_image.GetSpacing())
                mask_image.SetDirection(ct_image.GetDirection())
                mask_image.SetOrigin(ct_image.GetOrigin())


                if args.extract_radiomics:
                    # Extract features
                    features = extractor.execute(ct_image, mask_image)
                    # Convert the features to a pandas DataFrame row
                    feature_row = pd.DataFrame([features])
                    feature_row[args.nifti_clm_name]   = example_dictionary[args.nifti_clm_name][0]
                    feature_row['candidateID']         = annotation_id
                    if args.Malignant_lbl in example_dictionary.columns:
                       feature_row[args.Malignant_lbl]    = annotation_lbl
                else:
                    # Convert the features to a pandas DataFrame row
                    if args.Malignant_lbl in example_dictionary.columns:
                       feature_row = pd.DataFrame({args.nifti_clm_name: [example_dictionary[args.nifti_clm_name][0]],'candidateID': [annotation_id],args.Malignant_lbl: [annotation_lbl]})
                    else:
                       feature_row = pd.DataFrame({args.nifti_clm_name: [example_dictionary[args.nifti_clm_name][0]],'candidateID': [annotation_id]})
                    
                    print(feature_row)


                feature_row[args.coordX] = example_dictionary[args.coordX][Which_box_to_use]
                feature_row[args.coordY] = example_dictionary[args.coordY][Which_box_to_use]
                feature_row[args.coordZ] = example_dictionary[args.coordZ][Which_box_to_use]
                feature_row[args.w]      = example_dictionary[args.w][Which_box_to_use]
                feature_row[args.h]      = example_dictionary[args.h][Which_box_to_use]
                feature_row[args.d]      = example_dictionary[args.d][Which_box_to_use]

                # Save mask if needed
                if args.save_the_generated_mask:
                    output_nifti_path = os.path.join(args.save_nifti_path, f"{annotation_id}.nii.gz")
                    sitk.WriteImage(mask_image, output_nifti_path)
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
    seg_pyradiomics_main()
