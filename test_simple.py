# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import glob
import argparse
import numpy as np
import PIL.Image as pil

import torch
from torchvision import transforms

from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
import datetime
import csv
import cv2
import pandas as pd


def compute_errors(gt, pred, image_path):
    """Computation of error metrics between predicted and ground truth depths
    """
    gt_mm = gt / 655.35
    pred_mm = pred / 655.35
    
    thresh = np.maximum((gt_mm / pred_mm), (pred_mm / gt_mm))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    # compute MAE and MedAE
    mAE = np.mean(np.abs(gt_mm - pred_mm))
    medAE = np.median(np.abs(gt_mm - pred_mm))
    rmse = (gt_mm - pred_mm) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt_mm) - np.log(pred_mm)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt_mm - pred_mm) / gt_mm)

    sq_rel = np.mean(((gt_mm - pred_mm) ** 2) / gt_mm)

    return image_path, mAE, medAE, rmse, rmse_log, abs_rel, sq_rel, a1, a2, a3



def load_monodepth2_model(method, decompose):
    if method == "monodepth2":
        import networksMonoDepth2 as networks
    else:
        import networks
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))
    if method == "IID" and decompose:
        decompose_encoder = networks.ResnetEncoder(18, False)   
        decompose = networks.decompose_decoder(
            decompose_encoder.num_ch_enc, scales=range(4))
        return encoder, depth_decoder, decompose_encoder, decompose
    return encoder, depth_decoder, None, None

def load_monovit_model_hr(depth_decoder_path, device):
    import networksMonoVIT as networks
    depth_dict = torch.load(depth_decoder_path, map_location=device)
    feed_height = depth_dict['height']
    feed_width = depth_dict['width']
    #new_dict = depth_dict
    new_dict = {}
    for k,v in depth_dict.items():
        name = k[7:]
        new_dict[name]=v
    depth_decoder = networks.DeepNet('mpvitnet')
    depth_decoder.load_state_dict({k: v for k, v in new_dict.items() if k in depth_decoder.state_dict()})
    return None, depth_decoder, feed_height, feed_width

def load_monovit_model_lr():
    import networksMonoVIT as networks
    print("   Loading pretrained encoder")
    encoder = networks.mpvit_small() #networks.ResnetEncoder(opt.num_layers, False)
    encoder.num_ch_enc = [64,128,216,288,288]  # = networks.ResnetEncoder(opt.num_layers, False)
    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder()
    return encoder, depth_decoder, None, None

def load_model(depth_decoder_path, method, model_name, device, decompose=False):
    if method == "monodepth2" or method == "IID":
        return load_monodepth2_model(method, decompose)
    elif method == "monovit":
        if "640" not in model_name and "288" not in model_name:
            return load_monovit_model_hr(depth_decoder_path, device)
        else:
            return load_monovit_model_lr()

def spec_score_func(image, binary_mask, image_path):
    # Ensure binary_mask is of type uint8
    binary_mask = binary_mask.astype(np.uint8)
    
    # Find connected components (blobs) in the binary mask
    num_labels, labels_im = cv2.connectedComponents(binary_mask)

    kernel_size = 5  # Define the size of the surrounding region
    half_k = kernel_size // 2

    # Pad the image to handle edge cases when extracting surroundings
    padded_image = np.pad(image, pad_width=half_k, mode='edge')

    # List to store differences between blobs and surroundings
    blob_differences = []

    for label in range(1, num_labels):  # Start from 1 to skip background
        # Create a mask for the current blob
        blob_mask = (labels_im == label).astype(np.uint8)

        # Get the pixel values of the blob
        blob_pixels = image[blob_mask > 0]

        # Find the bounding box of the blob
        y_coords, x_coords = np.where(blob_mask > 0)
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        min_x, max_x = np.min(x_coords), np.max(x_coords)

        # Expand the bounding box to include the surrounding area
        expanded_min_y = max(min_y - half_k, 0)
        expanded_max_y = min(max_y + half_k, image.shape[0] - 1)
        expanded_min_x = max(min_x - half_k, 0)
        expanded_max_x = min(max_x + half_k, image.shape[1] - 1)

        # Extract the surrounding region (excluding the blob itself)
        surrounding_region = padded_image[expanded_min_y:expanded_max_y + 1, expanded_min_x:expanded_max_x + 1]

        # Create a mask for the surrounding region
        surrounding_mask = np.zeros_like(padded_image, dtype=np.uint8)
        surrounding_mask[expanded_min_y:expanded_max_y + 1, expanded_min_x:expanded_max_x + 1] = 1

        # Remove blob pixels from surrounding region
        surrounding_mask[expanded_min_y + (y_coords - min_y), expanded_min_x + (x_coords - min_x)] = 0
        surrounding_region_no_blob = surrounding_region[surrounding_mask[expanded_min_y:expanded_max_y + 1, expanded_min_x:expanded_max_x + 1] == 1]


        # Calculate the mean value of the blob and the surrounding region
        blob_mean = np.mean(blob_pixels)
        surrounding_mean = np.mean(surrounding_region_no_blob)

        # Compute the absolute difference between blob and surrounding
        diff = abs(blob_mean - surrounding_mean)/surrounding_mean
        blob_differences.append(diff)
        # print(f"Blob {label}: Mean Blob Value = {blob_mean}, Mean Surrounding Value = {surrounding_mean}, Difference = {diff}")
    
    # Set a threshold to consider if a blob is "close" to its surroundings
    threshold = 0.01  # Example threshold

    # Check if the blob differences are within the threshold
    close_blobs = [diff < threshold for diff in blob_differences]

    # Report the results
    if len(close_blobs) == 0:
        return image_path, 0
    else:
        percentage_close = sum(close_blobs) / len(close_blobs) * 100
        # print(f"Percentage of blobs close to their surroundings: {percentage_close:.2f}%")
        return image_path, percentage_close



def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for MonoVIT/Monodepth2/IID models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use')
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="png")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--pred_metric_depth",
                        help='if set, predicts metric depth instead of disparity. (This only '
                             'makes sense for stereo-trained KITTI models).',
                        action='store_true')
    parser.add_argument("--method", type=str,
                        help='method to use for depth prediction',
                        choices=["monodepth2", "monovit", "IID"],
                        default="monovit")
    parser.add_argument("--output_path", type=str,
                        help='output path for saving predictions in a folder',
                        default="output")
    parser.add_argument("--model_basepath", type=str,
                        help='base path for model files',
                        default="models")
    parser.add_argument("--config", type=str,
                        help='config file with training parameters',
                        default=None)
    parser.add_argument("--save_depth", action='store_true',
                        help='save predicted depth maps')
    parser.add_argument("--eval", action='store_true',
                        help='evaluate predicted depth maps')
    parser.add_argument("--min_depth", type=float, default=0.1,
                        help='minimum depth for evaluation')
    parser.add_argument("--max_depth", type=float, default=150, 
                        help='maximum depth for evaluation')
    parser.add_argument("--seq", type=str, nargs='+', default=[""],
                        help='sequence(s) to evaluate')
    parser.add_argument("--save_triplet", action='store_true',
                        help='save image depth and ground truth')
    parser.add_argument("--disable_median_scaling", action='store_true',
                        help='disable median scaling')
    parser.add_argument("--pred_depth_scale_factor", type=float, default=1.0,
                        help='depth prediction scaling factor')
    parser.add_argument("--median_scaling_specular", action='store_true',
                        help='use median scaling only on specular pixels')
    parser.add_argument("--notclipped", action='store_true',
                        help='skip clipping depth values')
    parser.add_argument("--input_mask", type=str, default=None,) #this is only for visualization and not for masking image before putting in encoder
    parser.add_argument("--decompose", action='store_true',
                        help='use decompose model')
    parser.add_argument("--addedspec", action='store_true',
                        help='use added specularity c3vd dataset')
    parser.add_argument("--maxing", action='store_true',
                        help='use maxing value for depth values')
    parser.add_argument("--maxing_value", type=float, default=0.8,
                        help='maxing value for depth values')
    parser.add_argument("--split_path", type=str, default=None,
                        help='path to a split file with image paths for evaluation')
    return parser.parse_args()


def test_simple(args, seq):
    # the c3vd dataset follows the format discribed bellow
    # depthmaps are stored as uint_16 .tiff
    # values between 0-(2**16-1) are map to 0-100 mm
    MIN_DEPTH = 0
    MAX_DEPTH = (2**16-1)
    
    """Function to predict for a single image or folder of images
    """
    errors = []
    errors_masked = []
    ratios = []
    scores = []
    
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.pred_metric_depth and "stereo" not in args.model_name:
        print("Warning: The --pred_metric_depth flag only makes sense for stereo-trained KITTI "
              "models. For mono-trained models, output depths will not in metric space.")

    download_model_if_doesnt_exist(args.model_basepath, args.model_name, args.method)
    model_path = os.path.join(args.model_basepath, args.method , args.model_name)


    print("-> Loading model from ", model_path)

    depth_decoder_path = os.path.join(model_path, "depth.pth")
    
    # LOADING PRETRAINED MODEL

    
    if args.method == "IID" and args.decompose:
        encoder, depth_decoder, decompose_encoder, decompose = load_model(
            depth_decoder_path, args.method, args.model_name, device, args.decompose)
    else:
        encoder, depth_decoder, feed_height, feed_width = load_model(
            depth_decoder_path, args.method, args.model_name, device)
        
    
    if encoder is not None:
        encoder_path = os.path.join(model_path, "encoder.pth")
        loaded_dict_enc = torch.load(encoder_path, map_location=device)

        # extract the height and width of image that this model was trained with
        feed_height = loaded_dict_enc['height']
        feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)
        encoder.to(device)
        encoder.eval()
        
        loaded_dict = torch.load(depth_decoder_path, map_location=device)
        depth_decoder.load_state_dict(loaded_dict)
        
        if args.method == "IID" and args.decompose:
            decompose_encoder_path = os.path.join(model_path, "decompose_encoder.pth")
            decompose_path = os.path.join(model_path, "decompose.pth")
            decompose_encoder.load_state_dict(torch.load(decompose_encoder_path, map_location=device))
            decompose_encoder.to(device)
            decompose_encoder.eval()
            
            decompose.load_state_dict(torch.load(decompose_path, map_location=device))
            decompose.to(device)
            decompose.eval()
        

    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        # print(os.path.join(args.image_path, seq, '*.{}'.format(args.ext)))
        paths = sorted(glob.glob(os.path.join(args.image_path, seq, '*.{}'.format(args.ext))))
        output_directory = os.path.join(args.output_path, args.method, args.model_name, args.type_data, seq)
        os.makedirs(output_directory, exist_ok=True)
        if args.method == "IID" and args.decompose:
            os.makedirs(os.path.join(output_directory, "decomposed"), exist_ok=True)
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.{}".format(args.ext)):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            # print(image_path)
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            if args.input_mask is not None:
                input_mask = pil.open(args.input_mask).convert('1')
                from PIL import ImageFilter
                for _ in range(10):
                    input_mask = input_mask.filter(ImageFilter.MinFilter(3))
                input_mask_pil = input_mask.resize((feed_width, feed_height), pil.LANCZOS)
                # dilate slightly
                
                input_mask = transforms.ToTensor()(input_mask_pil).unsqueeze(0)
                input_mask = torch.cat([input_mask]*3, dim=1)
                # input_image[input_mask==0] = 0
                # transforms.functional.to_pil_image(input_image.squeeze(0)).save("inputimage_masked.png")

            input_image = input_image.to(device)
            # transforms.functional.to_pil_image(input_image.squeeze(0)).save("inputimage.png")
            # exit(0)
            
            
            if not (args.method == "monovit" and "640" not in args.model_name and "288" not in args.model_name):
                features = encoder(input_image)
                outputs = depth_decoder(features)
                if args.method == "IID" and args.decompose:
                    decompose_features = decompose_encoder(input_image)
                    outputs[("reflectance",0)],outputs[("light",0)] = decompose(decompose_features)
                
            else:
                outputs = depth_decoder(input_image)

            
            pred_disp, _ = disp_to_depth(outputs[("disp", 0)], args.min_depth, args.max_depth)

            disp_resized = torch.nn.functional.interpolate(
                pred_disp, (original_height, original_width), mode="bilinear", align_corners=False)
            

            pred_depth = (1/disp_resized).squeeze().cpu().numpy()

            output_name_trip = os.path.splitext(os.path.basename(image_path))[0]
            name_dest_im_trip = os.path.join(output_directory, "{}.{}".format(output_name_trip, args.ext))
            # print(name_dest_im_trip)      
                
            if args.save_depth:
                if args.input_mask is not None:
                    input_mask_np = input_mask[0, 0, :, :].numpy()
                    pred_depth[input_mask_np == 0] = 0      
                pred_depth_raw = pred_depth.copy()      
                if args.maxing and pred_depth[pred_depth <= args.maxing_value].size > 0: # threshold 3 for sploss model
                    pred_depth[pred_depth > args.maxing_value] = np.max(pred_depth[pred_depth <= args.maxing_value]) #IMPORTANT:remove in some cases!
                max_value = np.max(pred_depth)
                trip_im = pil.fromarray(np.stack((pred_depth*255/max_value,)*3, axis=-1).astype(np.uint8))
                trip_im.save(name_dest_im_trip)
                if args.eval is not None:
                    # load gt 
                    if args.addedspec:
                        adjusted_image_path = image_path.replace("AddedSpec", "Dataset")
                    else:
                        adjusted_image_path = image_path
                    # spec mask
                    spec_mask = pil.open(adjusted_image_path.replace("Dataset", "Annotations_Dilated").replace("Inpainted_HKgen9","Annotations_Dilated"))
                    spec_mask = spec_mask.convert('L')
                    spec_mask = spec_mask.point( lambda p: 255 if p > 200 else 0 )
                    spec_mask = np.array(spec_mask.convert('1'))
                    scores.append(spec_score_func(pred_depth_raw, spec_mask, image_path))
                    
                    
            if args.method == "IID" and args.decompose:
                name_dest_im_reflec = os.path.join(output_directory, "decomposed", "{}{}.{}".format("reflect",output_name_trip, args.ext))
                reflec = outputs[("reflectance",0)].squeeze().cpu().numpy().transpose(1,2,0)
                reflec_pil = pil.fromarray((reflec*255).astype(np.uint8))
                reflec_pil.save(name_dest_im_reflec)
                
                name_dest_im_light = os.path.join(output_directory, "decomposed", "{}{}.{}".format("light",output_name_trip, args.ext))
                light = outputs[("light",0)].squeeze().cpu().numpy()
                light_pil = pil.fromarray((light*255).astype(np.uint8))
                light_pil.save(name_dest_im_light)

                        
            if args.eval and not args.save_depth:
                # load gt 
                if args.addedspec:
                    adjusted_image_path = image_path.replace("AddedSpec", "Dataset")
                else:
                    adjusted_image_path = image_path
                tiff_gt_depth = pil.open(adjusted_image_path.replace("color", "depth").replace("Inpainted_HKgen9","Dataset").replace("png", "tiff"))
                gt_depth = np.array(tiff_gt_depth, dtype=np.float32)
                
                
                # spec mask
                spec_mask = pil.open(adjusted_image_path.replace("Dataset", "Annotations_Dilated").replace("Inpainted_HKgen9","Annotations_Dilated"))
                spec_mask = spec_mask.convert('L')
                spec_mask = spec_mask.point( lambda p: 255 if p > 200 else 0 )
                spec_mask = np.array(spec_mask.convert('1'))


                # c3vd mask
                mask = gt_depth > 0



                if not args.disable_median_scaling:
                    if args.median_scaling_specular and spec_mask.sum() > 0:
                        ratio = np.median(gt_depth[spec_mask*mask]) / np.median(pred_depth[spec_mask*mask])
                    else:
                        ratio = np.median(gt_depth[mask]) / np.median(pred_depth[mask])
                    ratios.append(ratio)
                    pred_depth *= ratio
                else:
                    # to get this value monodepth runs this once and gets the median of the ratios as the single scale factor
                    pred_depth *= args.pred_depth_scale_factor

                # save triplet                
                if args.save_triplet:
                    max_value = np.max([np.max(gt_depth), np.max(pred_depth)])
                    trip_im = pil.fromarray(np.hstack([np.array(pil.open(image_path).convert('RGB')),
                                                       np.stack((pred_depth*mask*255/max_value,)*3, axis=-1).astype(np.uint8), 
                                                       np.stack((gt_depth*mask*255/max_value,)*3, axis=-1).astype(np.uint8)
                                                       ]))
                    output_name_trip = os.path.splitext(os.path.basename(image_path))[0]
                    name_dest_im_trip = os.path.join(output_directory, "{}_triplet.{}".format(output_name_trip, args.ext))
                    trip_im.save(name_dest_im_trip)
                    # print(name_dest_im_trip)
                    
                    
                pred_depth_masked = pred_depth[mask]
                gt_depth_masked = gt_depth[mask]
                
                if not args.notclipped:
                    pred_depth_masked[pred_depth_masked < MIN_DEPTH] = MIN_DEPTH
                    pred_depth_masked[pred_depth_masked > MAX_DEPTH] = MAX_DEPTH
                
                errors.append(compute_errors(gt_depth_masked, pred_depth_masked, image_path))
                
                if np.unique(spec_mask*mask).shape[0] == 2:
                    pred_depth_spec_masked = pred_depth[spec_mask*mask]
                    gt_depth_spec_masked = gt_depth[spec_mask*mask]
                    
                    if not args.notclipped:
                        pred_depth_spec_masked[pred_depth_spec_masked < MIN_DEPTH] = MIN_DEPTH
                        pred_depth_spec_masked[pred_depth_spec_masked > MAX_DEPTH] = MAX_DEPTH
                    
                    
                    errors_masked.append(compute_errors(gt_depth_spec_masked, pred_depth_spec_masked, image_path))
                else:
                    print("No valid pixels in spec_mask")
                    errors_masked.append([image_path, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")])
    
    if args.eval and not args.save_depth:
        if not args.disable_median_scaling:
            ratios = np.array(ratios)
            med = np.median(ratios)
            print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
    
        # mean_errors = np.array(errors).mean(0)
        # mean_errors_masked = np.array(errors_masked).mean(0)
        
        return errors, errors_masked
    elif args.eval and args.save_depth:
        # mean_scores = np.array(scores).mean(0)
        return scores, None
    else:
        return None, None
        


if __name__ == '__main__':
    args = parse_args()

    if args.config is not None:
        # Load args from the configuration file
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Update default args with args from the configuration file
        for key, value in config.items():
            setattr(args, key, value)

    if args.eval:
        date = datetime.datetime.now().strftime("_%Y-%m-%d")
        out = os.path.join(args.output_path, args.method, args.model_name)
        os.makedirs(out, exist_ok=True)
        if args.notclipped:
            notclipped = "_notclipped"
        else: 
            notclipped = ""
        
        if args.eval and args.save_depth:
            score = "specscore_"
            cols = ['video', 'path', 'SSM', '__']
        else:
            score = ""
            cols = ['video', 'path', "mAE", "medAE", "rmse", "rmse_log", "abs_rel", "sq_rel", "a1", "a2", "a3",  "mAE_masked", "medAE_masked", "rmse_masked", "rmse_log_masked", "abs_rel_masked", "sq_rel_masked", "a1_masked", "a2_masked", "a3_masked"]
    
    args.type_data = ""
    if "Inpainted_HKgen9" in args.image_path:
        args.type_data = "inpainted"
    # Check if args.seq is set to 'all'
    if args.split_path is not None:
        if args.split_path.endswith('inpainted.txt'):
            args.type_data = "hkinpainted"
        else:
            args.type_data = "hk"
        # Read list of image directories:
        with open(args.split_path) as f:
            images = f.read().splitlines()
        # Extract unique last folder names directly
        sequences = sorted(list({os.path.basename(os.path.normpath(os.path.split(path)[0])) for path in images}))
    elif os.path.isdir(args.image_path) and args.seq == ['all']:
        # List all folders in args.image_path
        sequences = sorted([folder for folder in os.listdir(args.image_path) if os.path.isdir(os.path.join(args.image_path, folder))])
    elif os.path.isdir(args.image_path):
        sequences = args.seq

    if args.eval:
        file =  open(f"{out}{score}{args.type_data}results{notclipped}{date}.csv", mode='w')
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(cols)
        
    for seq in sequences:
        errors, errors_masked = test_simple(args, seq)
        if args.eval:
            seq_list = [seq]*(len(errors))
            # save results to csv using unique_dirs
            

            # Create DataFrames from the errors and errors_masked lists
            df_errors = pd.DataFrame(errors)
            df_errors_masked = pd.DataFrame(errors_masked)

            # Merge the DataFrames on the image_path column
            if errors_masked != None:
                df_merged = pd.merge(df_errors, df_errors_masked,  on=df_errors.columns[0])
            else:
                df_merged = df_errors

            # Insert the sequence name column at the first position
            df_merged.insert(0, 'sequence_name', seq_list)

            writer.writerows(df_merged.itertuples(index=False, name=None))

    if args.eval:
        file.close()