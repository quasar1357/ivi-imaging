from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from cellpose import core, denoise, io, utils
from skimage import morphology
from skimage.filters import threshold_otsu
import pandas as pd
import seaborn as sns


use_GPU = core.use_gpu()
yn = ['NO', 'YES']
print(f'>>> GPU activated? {yn[use_GPU]}')

# Define the model globally
model = denoise.CellposeDenoiseModel(gpu=use_GPU, model_type="cyto3",
                                     restore_type="denoise_cyto3")


def segment(input, diameter=100, channels=[0,0], log=False):
    """
    Segments the input image(s) into separate cells using the Cellpose model.
    If a list of images is given, each output will be a list containing the results for the images.
    
    Parameters:
        input : np.array or list of np.arrays
            The image(s) to segment.
        diameter : int
            The expected diameter of the cells in the image(s).
        channels : list of int
            The channels to use for the segmentation. Details see below.
        log : bool
            Whether to log the output of the Cellpose model.

    Returns:
        masks : np.array or list of np.arrays
            The masks of the segmented cells.
        flows : np.array or list of np.arrays
            The flows of the segmented cells.
        styles : np.array or list of np.arrays
            The styles of the segmented cells.
        imgs_dn : np.array or list of np.arrays
            The denoised images of the segmented cells.

    Channels:
        define CHANNELS to run segementation on
        grayscale=0, R=1, G=2, B=3
        channels = [cytoplasm, nucleus]
        if NUCLEUS channel does not exist, set the second channel to 0

        IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
        channels = [0,0] # IF YOU HAVE GRAYSCALE
        channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
        channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

        or if you have different types of channels in each image
        channels = [[0,0], [2,3], [0,0]]

        if diameter is set to None, the size of the cells is estimated on a per image basis
        you can set the average cell `diameter` in pixels yourself (recommended) 
        diameter can be a list or a single number for all images
    """

    if log:
        io.logger_setup()

    # DEFINE CELLPOSE MODEL
    # model_type="cyto3" or "nuclei", or other model
    # restore_type: "denoise_cyto3", "deblur_cyto3", "upsample_cyto3", "denoise_nuclei", "deblur_nuclei", "upsample_nuclei"

    # model = denoise.CellposeDenoiseModel(gpu=False, model_type="cyto3",
    #                                  restore_type="denoise_cyto3")
    
    if not isinstance(input, list):
        img_list = [input]
        diam_list = [diameter]
    else:
        img_list = input
        diam_list = [diameter]*len(img_list)

    img_list = [check_make_single_ch(img) for img in img_list]

    masks, flows, styles, imgs_dn = model.eval(img_list, diameter=diam_list, channels=channels)
    outlines = [utils.masks_to_outlines(m) for m in masks]

    # Make cell IDs unique (continuing from previous image)
    prev_max = 0
    for mask in masks:
        mask += prev_max
        mask[mask == prev_max] = 0
        prev_max = mask.max()

    # Convert back to single image if only one image was given
    if not isinstance(input, list):
        masks = masks[0]
        flows = flows[0]
        styles = styles[0]
        imgs_dn = imgs_dn[0]
        outlines = outlines[0]

    return masks, flows, styles, imgs_dn, outlines


def get_means(input, masks, dilate=0):
    """
    Extracts the mean signal of each cell in the input image(s) based on the masks.
    If a list of images and a list of masks are given, each output will be a list containing the results for the images.

    Parameters:
        input : np.array or list of np.arrays
            The image(s) to extract the signal from.
        masks : np.array or list of np.arrays
            The masks of the cells in the image(s).
        dilate : int
            The amount of dilation to apply to the masks before calculating the mean signal.
            If negative, erosion is applied instead of dilation.

    Returns:
        signal_means_dicts_out : dict or list of dicts
            The mean signal of each cell in the image(s) as a dictionary.
        signal_means_lists_out : list or list of lists
            The mean signal of each cell in the image(s) as a list. (Loses cell IDs.)
        signal_means_masks_out : np.array or list of np.arrays
            An image in which, for each cell, the mean signal is assigned to its pixels.
    """

    if not isinstance(input, list):
        img_list = [input]
        mask_list = [masks]
    else:
        img_list = input
        mask_list = masks

    if not len(img_list) == len(mask_list):
        raise ValueError('Length of input and masks must be the same.')

    signal_means_dicts_out = []
    signal_means_lists_out = []
    signal_means_masks_out = []
    for img, mask in zip(img_list, mask_list):

        img = check_make_single_ch(img)
        
        img_signal_means_dict = {} #{cell_id: np.mean(img[cell_mask_for_mean]) for cell_id in range(1, mask.max()+1)}
        img_signal_means_list = [] #[val for k, val in img_signal_means_dict.items()]
        img_signal_means_mask = np.zeros_like(img, dtype=np.float32)
        # add the mean to the dict and mask
        lowest_non_zero = mask[mask != 0].min()
        for cell_id in range(lowest_non_zero, mask.max()+1):
            cell_mask = mask == cell_id
            cell_mask_for_mean = cell_mask.copy()
            if dilate > 0:
                cell_mask_for_mean = morphology.binary_dilation(cell_mask_for_mean, morphology.disk(dilate))
            elif dilate < 0:
                cell_mask_for_mean = morphology.binary_erosion(cell_mask_for_mean, morphology.disk(-dilate))
            cell_mean = np.mean(img[cell_mask_for_mean])
            img_signal_means_dict[cell_id] = cell_mean
            img_signal_means_list.append(cell_mean)
            img_signal_means_mask += cell_mean * cell_mask

        signal_means_dicts_out.append(img_signal_means_dict)
        signal_means_lists_out.append(img_signal_means_list)
        signal_means_masks_out.append(img_signal_means_mask)

    if not isinstance(input, list):
        signal_means_dicts_out = signal_means_dicts_out[0]
        signal_means_lists_out = signal_means_lists_out[0]
        signal_means_masks_out = signal_means_masks_out[0]
    
    return signal_means_dicts_out, signal_means_lists_out, signal_means_masks_out


def get_bins(means_dicts_in, thresh=None, cell_masks_in=None):
    """
    Puts the mean signal of each cell in bins based on a threshold, assigning 1 (below threshold) or 2 (above threshold).
    If a list of dictionaries (and optionally cell_masks) is given, each output will be a list containing the results for the dictionaries.

    Parameters:
        means_dicts_in : dict or list of dicts
            The mean signal of each cell in the image(s) as a dictionary.
        thresh : float
            The threshold to use for the binning.
        cell_masks_in : np.array or list of np.arrays, optional
            An image containing the masks of the cells in the image(s).
            If given, the bins will be applied to the masks.

    Returns:
        bins_dicts_out : dict or list of dicts
            The bins of the mean signal of each cell in the image(s) as a dictionary.
        bins_lists_out : list or list of lists
            The bins of the mean signal of each cell in the image(s) as a list. (Loses cell IDs.)
        bins_masks_out : np.array or list of np.arrays
            An image in which, for each cell, the bin is assigned to its pixels.
            None if cell_masks_in is None.
    """

    if not isinstance(means_dicts_in, list):
        means_dicts = [means_dicts_in]
        cell_masks = [cell_masks_in]
    else:
        means_dicts = means_dicts_in
        cell_masks = cell_masks_in
    
    if cell_masks_in is not None and not len(means_dicts) == len(cell_masks):
        raise ValueError('Length of means_dicts and cell_masks must be the same.')

    bins_dicts_out = []
    bins_lists_out = []
    bins_masks_out = []
    for means_dict, cell_mask in zip(means_dicts, cell_masks):
        # If there is no threshold given from higher levels, use otsu on single sample as default
        if thresh is None:
            sample_thresh = threshold_otsu(np.array([v for v in means_dict.values()]))
            print("sample_thresh", sample_thresh)
        else:
            sample_thresh = thresh
        # Create bins using the threshold
        bins_dict = {cell_id: 2 if val > sample_thresh else 1 for cell_id, val in means_dict.items()}
        bins_list = [val for k, val in bins_dict.items()]

        bins_dicts_out.append(bins_dict)
        bins_lists_out.append(bins_list)

        if cell_masks_in is None:
            bins_masks_out.append(None)
            continue
        
        # create mask
        bins_mask = np.zeros_like(cell_mask, dtype=np.float32)
        # add each mean
        for cell_id in bins_dict.keys():
            single_cell_mask = cell_mask == cell_id
            cell_bin = bins_dict[cell_id]
            bins_mask += cell_bin * single_cell_mask
        bins_mask = bins_mask.astype(np.uint8)
        bins_masks_out.append(bins_mask)

    # return single values if only one image
    if not isinstance(means_dicts_in, list):
        bins_dicts_out = bins_dicts_out[0]
        bins_lists_out = bins_lists_out[0]
        bins_masks_out = bins_masks_out[0]

    return bins_dicts_out, bins_lists_out, bins_masks_out


def get_pop(bins_1_in, bins_2_in, bin1_name=None, bin2_name=None):
    """
    Takes two sets of bins and creates a population matrix based on the unique combinations of the bins.
    The populations are assigned in the order of the unique combinations.
    If lists of bins are given, each output will be a list containing the results for each pair.

    Parameters:
        bins_1_in : dict or list of dicts
            The bins of the first variable.
        bins_2_in : dict or list of dicts
            The bins of the second variable.
        bin1_name : str, optional
            The name of the first variable.
        bin2_name : str, optional
            The name of the second variable.

    Returns:
        cell_pop_dict_out : dict
            A dictionary with the population of each cell.
        pop_counts_out : dict
            A dictionary with the count of each population.
        pop_counts_df_out : pd.DataFrame
            A DataFrame with one row, and the populations as columns.
        pop_counts_matrix_dfs_out : pd.DataFrame
            A DataFrame with the populations as rows and counts as columns.
            If bin1_name and bin2_name are given, they will be used as the index and column names, respectively.
    """

    if not isinstance(bins_1_in, list):
        bins_1_list = [bins_1_in]
        bins_2_list = [bins_2_in]
    else:
        bins_1_list = bins_1_in
        bins_2_list = bins_2_in
    
    if not len(bins_1_list) == len(bins_2_list):
        raise ValueError('Length of both bins dicts must be the same.')
    
    cell_pop_dicts_out = []
    pop_counts_out = []
    pop_counts_df_out = []
    pop_counts_matrix_dfs_out = []

    for bins_1, bins_2 in zip(bins_1_list, bins_2_list):
        # Get the unique values from bins_1 and bins_2
        bins_1_vals = set(bins_1.values())
        bins_2_vals = set(bins_2.values())
        # Initialize the dictionary to store the population for each cell
        cell_pop_dict = {}
        # Populate the cell_pop_dict based on bin values
        for cell_id in bins_1.keys():
            pop_bin = 1
            for bin1_val in bins_1_vals:
                for bin2_val in bins_2_vals:
                    if bins_1[cell_id] == bin1_val and bins_2[cell_id] == bin2_val:            
                        cell_pop_dict[cell_id] = pop_bin
                    pop_bin += 1

        # populations = set(cell_pop_dict.values())
        populations = range(1, len(bins_1_vals)*len(bins_2_vals)+1)
        pop_counts = {pop: len([id for id in cell_pop_dict if cell_pop_dict[id]==pop]) for pop in populations}
        pop_counts_df = pd.DataFrame(pop_counts, index=[0])
        # NOTE: This naming is not flexible for bin numbers other than 2
        if bin1_name is not None and bin2_name is not None:
            pop_counts_df.columns = ["non-" + bin1_name + " & " + "non-" + bin2_name,
                                        "non-" + bin1_name + " & " + bin2_name,
                                        bin1_name + " & " + "non-" + bin2_name,
                                        bin1_name + " & " + bin2_name
                                        ]

        # Create a df with the populations and counts, putting bins_1 in rows and bins_2 in columns
        dict_for_df = {}
        pop_bin = 1
        for bin1_val in bins_1_vals:
            row = {}
            for bin2_val in bins_2_vals:
                if pop_bin in pop_counts:
                    row[bin2_val] = pop_counts[pop_bin]
                pop_bin += 1
            dict_for_df[bin1_val] = row
        # Create the df, make sure to fill by row
        pop_counts_matrix_df = pd.DataFrame(dict_for_df).T
        if bin1_name is not None:
            pop_counts_matrix_df.index.name = bin1_name
        if bin2_name is not None:
            pop_counts_matrix_df.columns.name = bin2_name
        
        cell_pop_dicts_out.append(cell_pop_dict)
        pop_counts_out.append(pop_counts)
        pop_counts_df_out.append(pop_counts_df)
        pop_counts_matrix_dfs_out.append(pop_counts_matrix_df)

    # return single values if only one image
    if not isinstance(bins_1_in, list):
        cell_pop_dicts_out = cell_pop_dicts_out[0]
        pop_counts_out = pop_counts_out[0]
        pop_counts_df_out = pop_counts_df_out[0]
        pop_counts_matrix_dfs_out = pop_counts_matrix_dfs_out[0]
    
    return cell_pop_dicts_out, pop_counts_out, pop_counts_df_out, pop_counts_matrix_dfs_out


def get_sep_rel_pop_counts_df(pop_counts_df, bin1_name=None, bin2_name=None):
    """
    Takes a DataFrame with the population counts and returns the counts of the two bins
    separately and as relative values.
    If bin2_name is given, the columns will be renamed accordingly.

    Parameters:
        pop_counts_df : pd.DataFrame
            A DataFrame with one row, and the populations as columns.
        bin2_name : str, optional
            The name of the second variable.

    Returns:
        df_1 : pd.DataFrame
            A DataFrame with the counts of the first bin.
        df_2 : pd.DataFrame
            A DataFrame with the counts of the second bin.
        df_1_rel : pd.DataFrame
            A DataFrame with the relative counts of the first bin.
        df_2_rel : pd.DataFrame
            A DataFrame with the relative counts of the second bin.
    """
    # Split the DataFrame into two parts
    df_1 = pop_counts_df.copy().iloc[:,:2]
    df_2 = pop_counts_df.copy().iloc[:,2:]

    # Calculate the relative values
    df_1_rel = df_1.div(df_1.sum(axis=1), axis=0)
    df_2_rel = df_2.div(df_2.sum(axis=1), axis=0)

    # Rename the columns if bin2_name is given
    # NOTE: This naming is not flexible for bin numbers other than 2
    for df in [df_1, df_2, df_1_rel, df_2_rel]:
        if bin2_name is not None:
            df.columns = ["non-" + bin2_name, bin2_name]

    # Save in dictionaries
    # NOTE: This naming is not flexible for bin numbers other than 2
    if bin1_name is not None:
        keys = ["non-" + bin1_name, bin1_name]
    else:
        keys = ["1", "2"]
    abs = {keys[0]: df_1, keys[1]: df_2}
    rel = {keys[0]: df_1_rel, keys[1]: df_2_rel}

    return abs, rel


def get_pop_mask(cell_pop_dicts_in, cell_masks_in):
    """
    Assign the population of each cell to the pixels in the cell mask.
    If lists of dictionaries and masks are given, the output will be a list containing the results for each pair.

    Parameters:
        cell_pop_dict : dict or list of dicts
            A dictionary with the population of each cell.
        cell_mask : np.array or list of np.arrays
            An image containing the masks of the cells in the image.

    Returns:
        pop_masks : np.array
            An image in which, for each cell, the population is assigned to its pixels
    """

    if not isinstance(cell_pop_dicts_in, list):
        cell_pop_dicts = [cell_pop_dicts_in]
        cell_masks = [cell_masks_in]
    else:
        cell_pop_dicts = cell_pop_dicts_in
        cell_masks = cell_masks_in

    if not len(cell_pop_dicts) == len(cell_masks):
        raise ValueError('Length of cell_pop_dicts and cell_masks must be the same.')

    pop_masks = []
    for cell_pop_dict, cell_mask in zip(cell_pop_dicts, cell_masks):
        
        pop_mask = np.zeros_like(cell_mask, dtype=np.int32)

        for cell_id in cell_pop_dict.keys():
            single_cell_mask = cell_mask == cell_id
            pop = cell_pop_dict[cell_id]
            pop_mask += pop * single_cell_mask
        pop_mask = pop_mask.astype(np.uint8)
        pop_masks.append(pop_mask)

    if not isinstance(cell_pop_dicts_in, list):
        pop_masks = pop_masks[0]

    return pop_masks
    

### HELPER FUNCTIONS ###


def read_image(path):
    img = Image.open(path)
    img = np.array(img)
    return img

def check_make_single_ch(input):

    if not isinstance(input, list):
        imgs = [input]
    else:
        imgs = input
    
    imgs_out = []
    for img in imgs:
        if img.ndim == 3:
            # See if RGB and only one channel has data
            if img.shape[2] == 3:
                if np.max(img[:,:,0]) == 0 and np.max(img[:,:,1]) == 0:
                    img = img[:,:,2]
                elif np.max(img[:,:,0]) == 0 and np.max(img[:,:,2]) == 0:
                    img = img[:,:,1]
                elif np.max(img[:,:,1]) == 0 and np.max(img[:,:,2]) == 0:
                    img = img[:,:,0]
                else:
                    raise ValueError('Image is RGB but not single channel.')
            else:
                raise ValueError('Image is not RGB.')
        imgs_out.append(img)
    
    if not isinstance(input, list):
        imgs_out = imgs_out[0]

    return imgs_out

def normalize(input):

    if not isinstance(input, list):
        images = [input]
    else:
        images = input

    # Step 1: Normalize each image to the range [0, 1]
    normalized_images = []

    for image in images:
        # Normalize the image to [0, 1]
        norm_image = (image - np.min(image)) / (np.max(image) - np.min(image))
        # Rescale to [0, 255]
        norm_image = norm_image * 255
        normalized_images.append(norm_image.astype(np.uint8))

    # Step 2: Compute global mean and std (or use a reference image)
    global_mean = np.mean([np.mean(image) for image in normalized_images])
    global_std = np.std([np.std(image) for image in normalized_images])

    # Step 3: Normalize the mean and std of each image to match the global mean and std
    final_normalized_images = []

    for image in normalized_images:
        # Normalize the image to have the global mean and std
        image_mean = np.mean(image)
        image_std = np.std(image)
        standardized_image = (image - image_mean) / image_std
        standardized_image = standardized_image * global_std + global_mean
        # Clip the values to be between 0 and 255
        standardized_image = np.clip(standardized_image, 0, 255)
        final_normalized_images.append(standardized_image.astype(np.uint8))
    
    if not isinstance(input, list):
        final_normalized_images = final_normalized_images[0]
    
    return final_normalized_images


### WRAPPER & PLOTTING FUNCTIONS ###

def seg_mean_bin_pop(seg_input, signal1_input, signal2_input, masks=None, norm=True,
                     diameter=100, dilate=0, signal1_thresh= 'otsu_overall', signal2_thresh= 'otsu_overall',
                     signal1_name="signal 1", signal2_name="signal 2", sample_names=None,
                     plt_res=False):
    
    # Segment the images
    if masks is None:
        masks, flows, styles, imgs_dn = segment(seg_input, diameter=diameter)

    # Pre-process
    signal1_input = check_make_single_ch(signal1_input)
    signal2_input = check_make_single_ch(signal2_input)
    if norm:
        signal1_input = normalize(signal1_input)
        signal2_input = normalize(signal2_input)

    # Get the means and bins
    signal1_means, signal1_means_list, signal1_means_mask = get_means(signal1_input, masks, dilate=dilate)
    signal2_means, signal2_means_list, signal2_means_mask = get_means(signal2_input, masks, dilate=dilate)
    means = {signal1_name: signal1_means, signal2_name: signal2_means}

    # Get the bins
    # IMPORTANT: THIS DEFAULT THRESHOLD MIGHT NEED TO BE ADJUSTED
    if signal1_thresh == 'otsu_overall':
        # signal1_thresh = np.mean(signal1_means_list)
        signal1_thresh = threshold_otsu(np.concatenate(signal1_means_list))
        print("signal1_thresh", signal1_thresh)
    elif signal1_thresh == 'otsu-per-sample':
        signal1_thresh = None
    elif type(signal1_thresh) != int or type(signal1_thresh) != float:
        raise ValueError('signal1_thresh must be an integer or float.')
    if signal2_thresh ==  'otsu_overall':
        # signal2_thresh = np.mean(signal2_means_list)
        signal2_thresh = threshold_otsu(np.concatenate(signal2_means_list))
        print("signal2_thresh", signal2_thresh)
    elif signal2_thresh == 'otsu-per-sample':
        signal2_thresh = None
    elif type(signal2_thresh) != int or type(signal2_thresh) != float:
        raise ValueError('signal2_thresh must be an integer or float.')
    signal1_bins, signal1_bins_list, signal1_bins_mask = get_bins(signal1_means, signal1_thresh, masks)
    signal2_bins, signal2_bins_list, signal2_bins_mask = get_bins(signal2_means, signal2_thresh, masks)
    bins = {signal1_name: signal1_bins, signal2_name: signal2_bins}

    # Get the populations
    cell_pop_dicts, pop_counts, pop_counts_dfs, pop_counts_matrix_dfs = get_pop(signal1_bins, signal2_bins,
                                                                                signal1_name, signal2_name)
    pop_masks = get_pop_mask(cell_pop_dicts, masks)

    # Combine samples
    overall_count_df = pd.concat(pop_counts_dfs)
    if sample_names is not None:
        overall_count_df.index = sample_names

    # Get the separate, relative populations
    abs, rel = get_sep_rel_pop_counts_df(overall_count_df, signal1_name, signal2_name)

    # Get the overall population matrix
    overall_count_matrix_df = sum(pop_counts_matrix_dfs)
    overall_perc_matrix_df = overall_count_matrix_df / overall_count_matrix_df.sum().sum()

    if plt_res:
        plot_bin2_in_bin1(rel)

    return masks, means, bins, cell_pop_dicts, pop_counts, overall_count_df, rel


def plot_bin2_in_bin1(rel_in):
    """
    Plots the relative populations of bin2 in bin1.
    Creates a Barplot with a bar for each bin1 value (negative and positive),
    and each bar represents the relative amount of bin2 values in the bin1.
    Only works with two bins for now.
    
    Parameters:
        rel_in : Dictionary
            The relative populations. The keys are the bin1 values, and the values are DataFrames.
            Each DataFrame has the bin2 values as columns and the relative populations as rows.

    Returns:
        None
    """
    inf_in_cil = pd.concat([df.iloc[:, 1] for df in rel_in.values()],
                           axis=1)
    inf_in_cil.columns = rel_in.keys()

    plt.figure(figsize=(2, 3), dpi=200)
    sns.barplot(data=inf_in_cil, errorbar='sd', color='skyblue', capsize=0.1)
    plt.title(list(rel_in.values())[0].columns[1])
    # plt.ylim(0, 1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()


def plot_bins(rel_in):
    
    fig, ax = plt.subplots(1, len(rel_in), figsize=(3*len(rel_in), 5))
    plot_num = 0
    for key, val in rel_in.items():
        sns.barplot(data=val, errorbar='sd', color='skyblue', capsize=0.1, ax=ax[plot_num])
        ax[plot_num].set_title(key)
        ax[plot_num].set_ylim(0, 1)
        ax[plot_num].spines['top'].set_visible(False)
        ax[plot_num].spines['right'].set_visible(False)
        plot_num += 1

    plt.tight_layout()
    plt.show()