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
decompose = True
colormap = False
color = "" if colormap else "raw"
prefix = ["/decomposed","reflect"] if decompose else ["",""]
aug_list = ['']#, '_add', '_rem', '_addrem']
seq_list = ["Test1", "Test1", "Test1", "Test1"]
idx_list = ["00000", "00020", "00050", "00060"]
model_list = ['IID'] #['monodepth2', 'monovit', 'IID']

if IID_pretrained:
    for model in model_list:
        rows = []
        for seq, idx in zip(seq_list, idx_list):
            image_files = [f"/media/rema/data/DataHKGab/Undistorted/{seq}/{idx}.png"]
            gt_depth_files = None #[f"/media/rema/outputs/undisttrain/undist/IID/IID_depth_model/{seq}inpainted/{idx}.png"]
            pred_depth_files = [f"/media/rema/outputs/undisttrain/undist_masked/IID/IID_depth_model/{seq}/{idx}.png"]
            # Process the images for each row
            rows.append(process_images(image_files, pred_depth_files, gt_depth_files, colormap))

        # Stack the rows vertically
        result = np.vstack(rows)

        # Save the result
        cv2.imwrite(f'/media/rema/outputs/undisttrain/undist_masked/visualresultsIIDpretrainedHK.png', result)
else:
    for model in model_list:
        rows = []
        for seq, idx in zip(seq_list, idx_list):
            image_files = [f"/media/rema/data/DataHKGab/Undistorted/{seq}/{idx}.png"]
            gt_depth_files = [f"/media/rema/outputs/undisttrain/undist_masked/{model}/finetuned{aug_list[0]}_mono_hk_288/models/weights_19/{seq}inpainted{prefix[0]}/{prefix[1]}{idx}.png"]
            pred_depth_files = [f"/media/rema/outputs/undisttrain/undist_masked/{model}/finetuned{aug}_mono_hk_288/models/weights_19/{seq}{prefix[0]}/{prefix[1]}{idx}.png" for aug in aug_list]
            # Process the images for each row
            rows.append(process_images(image_files, pred_depth_files, gt_depth_files, colormap))

        # Stack the rows vertically
        result = np.vstack(rows)

        # Save the result
        cv2.imwrite(f'/media/rema/outputs/undisttrain/undist_masked/visualresults{color}{prefix[1]}{model}HK.png', result)