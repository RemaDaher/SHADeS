import numpy as np
import cv2

def process_images(image_files, pred_depth_files, gt_depth_files=None, colormap = True):
    # List to store the cropped images
    cropped_images = []
    if colormap: 
        gray = cv2.IMREAD_GRAYSCALE
    else:
        gray = cv2.IMREAD_COLOR
        
    for image_file in image_files:
        # Read the image
        img = cv2.imread(image_file)
        # Add the image to the list
        cropped_images.append(img)

    for pred_depth_file in pred_depth_files:
        # Read the image
        img = cv2.imread(pred_depth_file, gray)
        # Apply the colormap
        if colormap:
            img = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)

        # Add the cropped image to the list
        cropped_images.append(img)

    if gt_depth_files is not None:
        for gt_depth_file in gt_depth_files:
            # Read the image
            img = cv2.imread(gt_depth_file, gray)
            # Apply the colormap
            if colormap:
                img = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
            # Add the image to the list
            cropped_images.append(img)

    # Stack the images horizontally
    return np.hstack(cropped_images)


# List of image paths
IID_pretrained = False
decompose = False
colormap = True
color = "" if colormap else "raw"
prefix = ["/decomposed","light"] if decompose else ["",""] # reflect or light
aug_list = ['', '_pseudo', '_lightinput', '_pseudo_lightinput']#['', '_add', '_rem', '_addrem'] 
#['', '_inp_pseudo', '_flip', '_rot']#'_ds3', '_alb05', '_rep5', '_rec05','_lr5']#, '_add', '_rem', '_addrem']
seq_list = ["07bbedb7-1d45-4655-9ccd-aeb28a2c4bab", 
            "da5d2629-3a74-4ec0-9ace-57dbc6ebddad", 
            "baa6b87a-ff86-4306-9be2-c518956ae4ee", 
            "da5d2629-3a74-4ec0-9ace-57dbc6ebddad"]
idx_list = ["00000", "00107", "00019", "00266"]

# seq_list = ["07bbedb7-1d45-4655-9ccd-aeb28a2c4bab", 
#             "07bbedb7-1d45-4655-9ccd-aeb28a2c4bab", 
#             "07bbedb7-1d45-4655-9ccd-aeb28a2c4bab", 
#             "07bbedb7-1d45-4655-9ccd-aeb28a2c4bab"]
# idx_list = ["00000", "00016", "00078", "00087"]

model_list =  ['IID'] #['monodepth2', 'monovit', 'IID'] #

if IID_pretrained:
    for model in model_list:
        rows = []
        for seq, idx in zip(seq_list, idx_list):
            image_files = [f"/raid/rema/data/DataHKGab/Undistorted/{seq}/{idx}.png"]
            gt_depth_files = None #[f"/raid/rema/outputs/undisttrain/undist/IID/IID_depth_model/{seq}inpainted/{idx}.png"]
            pred_depth_files = [f"/raid/rema/outputs/undisttrain/undist_masked/IID/IID_depth_model/{seq}/{idx}.png"]
            # Process the images for each row
            rows.append(process_images(image_files, pred_depth_files, gt_depth_files, colormap))

        # Stack the rows vertically
        result = np.vstack(rows)

        # Save the result
        cv2.imwrite(f'/raid/rema/outputs/undisttrain/undist_masked/visualresultsIIDpretrainedHK.png', result)
else:
    for model in model_list:
        rows = []
        for seq, idx in zip(seq_list, idx_list):
            image_files = [f"/raid/rema/data/BBPS-2-3Frames/Undistorted/Frames/{seq}/{idx}.png"]
            gt_depth_files = [f"/raid/rema/outputs/undisttrain/undist/{model}/finetuned_mono_hkfull_288{aug_list[0]}/models/weights_19/hkinpainted/{seq}{prefix[0]}/{prefix[1]}{idx}.png"]
            pred_depth_files = [f"/raid/rema/outputs/undisttrain/undist/{model}/finetuned_mono_hkfull_288{aug}/models/weights_19/hk/{seq}{prefix[0]}/{prefix[1]}{idx}.png" for aug in aug_list]
            # pred_depth_files.extend([f"/raid/rema/outputs/undisttrain/undist/{model}/finetuned_mono_hk_288{aug_list[-1]}/models/weights_29/{seq}{prefix[0]}/{prefix[1]}{idx}.png"])

            # Process the images for each row
            rows.append(process_images(image_files, pred_depth_files, gt_depth_files, colormap))

        # Stack the rows vertically
        result = np.vstack(rows)

        # Save the result
        cv2.imwrite(f'/raid/rema/outputs/undisttrain/undist/visualresults{color}{prefix[1]}{model}HK2{str(aug_list)}.png', result)