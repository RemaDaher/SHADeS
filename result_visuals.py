import numpy as np
import cv2

def process_images(image_files, pred_depth_files, gt_depth_files= None):
    # List to store the cropped images
    cropped_images = []

    for image_file in image_files:
        # Read the image
        img = cv2.imread(image_file)
        # Add the image to the list
        cropped_images.append(img)

    for pred_depth_file in pred_depth_files:
        # Read the image
        img = cv2.imread(pred_depth_file, cv2.IMREAD_GRAYSCALE)
        # Apply the colormap
        img = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)

        # Calculate the coordinates of the middle third
        height, width = img.shape[:2]
        start_width = width // 3
        end_width = start_width * 2

        # Crop the image
        cropped_img = img[:, start_width:end_width]

        # Add the cropped image to the list
        cropped_images.append(cropped_img)

    if gt_depth_files is not None:
        for gt_depth_file in gt_depth_files:
            # Read the image
            img = cv2.imread(gt_depth_file, cv2.IMREAD_GRAYSCALE)
            # Apply the colormap
            img = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
            # Add the image to the list
            cropped_images.append(img)

    # Stack the images horizontally
    return np.hstack(cropped_images)


# List of image paths
IID_pretrained = False
aug_list = ['', '_pseudo_dsms','_pseudo_dsms_automasking','_pseudo_dsms_automasking_noadjust']  #['', '_pseudo', '_lightinput', '_pseudo_lightinput']#['', '_add', '_rem', '_addrem']
# seq_list = ['sigmoid_t3_b', 'trans_t2_a', 'trans_t2_b', 'trans_t3_a', 'trans_t4_a', 'trans_t4_b']
# idx_list = ['0000', '0000', '0007', '0000', '0000', '0000']
seq_list = ["sigmoid_t3_a", "trans_t2_a", "trans_t2_b", "trans_t4_b"]
idx_list = ["0000", "0000", "0000", "0000"]
model_list = ['IID']#['monodepth2', 'monovit', 'IID']

if IID_pretrained:
    for model in model_list:
        rows = []
        for seq, idx in zip(seq_list, idx_list):
            image_files = [f"/media/rema/data/C3VD/Undistorted/Dataset/{seq}/{idx}_color.png"]
            gt_depth_files = None
            pred_depth_files = [f"/media/rema/outputs/undisttrain/undist/IID/IID_depth_model/{seq}/{idx}_color_triplet.png"]
            # Process the images for each row
            rows.append(process_images(image_files, pred_depth_files, gt_depth_files))

        # Stack the rows vertically
        result = np.vstack(rows)

        # Save the result
        cv2.imwrite(f'/media/rema/outputs/undisttrain/undist/visualresultsIIDpretrained.png', result)
else:
    for model in model_list:
        rows = []
        for seq, idx in zip(seq_list, idx_list):
            image_files = [f"/raid/rema/data/C3VD/Undistorted/Dataset/{seq}/{idx}_color.png"]
            gt_depth_files = [f"/raid/rema/data/C3VD/Undistorted/Dataset/{seq}/{idx}_depth.tiff"]
            pred_depth_files = [f"/raid/rema/outputs/undisttrain/undist/{model}/finetuned_mono_hkfull_288{aug}/models/weights_19/{seq}/{idx}_color_triplet.png" for aug in aug_list]
            # Process the images for each row
            rows.append(process_images(image_files, pred_depth_files, gt_depth_files))

        # Stack the rows vertically
        result = np.vstack(rows)

        # Save the result
        cv2.imwrite(f'/raid/rema/outputs/undisttrain/undist/visualresults{model}{str(aug_list)}.png', result)