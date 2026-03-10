from PIL import Image, ImageDraw, ImageFont
import numpy as np
from matplotlib import pyplot as plt
from cellpose import core, denoise, io, utils
from skimage import morphology
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
import pandas as pd
import seaborn as sns
from pathlib import Path
import itertools
import re
from aicsimageio import AICSImage
import pickle


class CellAnalyzer:
    
    def __init__(self, path):

        # Initialize the class with the path to the images and the model
        self.path = Path(path)
        self.samples_df = None
        self.img_arrays = None
        self.projections = None
        self.projections_types = None
        self.seg_channels = None
        self.seg_diameter = None
        self.masks = None
        self.flows = None
        self.styles = None
        self.imgs_dn = None
        self.outlines = None
        self.signal_dicts = {}
        self.signal_lists = {}
        self.signal_masks = {}
        self.cells_df = None
        self.signal_mode = {}
        self.bin_masks = {}
        self.bins = {}
        self.cfg_df = None

        # Load cellpose model
        self.load_cellpose_model()

    def save(self, folder_name=None, overwrite=False):
        """
        Saves the data frame in a csv and the object in a pickle file.
        
        Parameters:
            name : str
                The name of the file to save to.

        Returns:
            None
        """
        if folder_name is None:
            folder_name = "CellAnalyzer"
        # Check if the file already exists
        if (self.path / folder_name).exists() and not overwrite:
            raise ValueError(f"Folder {folder_name} already exists. Please choose a different name.")

        # Create the folder if it doesn't exist
        output_path = self.path / folder_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Save the DataFrames
        if self.samples_df is not None:
            self.samples_df.to_csv(output_path / "metadata.csv", index=False)
        if self.cells_df is not None:
            self.cells_df.to_csv(output_path / "metadata_cells.csv", index=False)
        if self.cfg_df is not None:
            self.cfg_df.to_csv(output_path / "metadata_cfg.csv", index=False)

        # Save the object
        data_to_save = {
            'path': self.path,
            'samples_df': self.samples_df,
            'projections': self.projections,
            'projections_types': self.projections_types,
            'seg_channels': self.seg_channels,
            'seg_diameter': self.seg_diameter,
            'masks': self.masks,
            'flows': self.flows,
            'styles': self.styles,
            # 'imgs_dn': self.imgs_dn,
            'outlines': self.outlines,
            'signal_means_dicts': self.signal_dicts,
            'signal_lists': self.signal_lists,
            'signal_masks': self.signal_masks,
            'cells_df': self.cells_df,
            'signal_mode': self.signal_mode,
            'bin_masks': self.bin_masks,
            'bins': self.bins,
            'cfg_df': self.cfg_df
        }

        with open(output_path / "CellAnalyzer.pkl", "wb") as f:
            pickle.dump(data_to_save, f)

    @staticmethod
    def load(pkl_name=None, use_GPU=True, load_images=False):
        """
        Loads the object from a pickle file.
        
        Parameters:
            name : str
                The name of the file to load from.

        Returns:
            CellAnalyzer instance
                The loaded CellAnalyzer instance.
        """
        # Automatically find the pickle file if not given
        if pkl_name[-4:] != ".pkl": # If a folder given instead of file, try find the pickle file; if it's not in the current folder, look if there is a folder CellAnalyzer and look in there
            folder_path = Path(pkl_name)
            if (folder_path / "CellAnalyzer.pkl").exists():
                pkl_name = folder_path / "CellAnalyzer.pkl"
            elif (folder_path / "CellAnalyzer" / "CellAnalyzer.pkl").exists():
                pkl_name = folder_path / "CellAnalyzer" / "CellAnalyzer.pkl"
            else:
                raise ValueError(f"Could not find CellAnalyzer.pkl in {folder_path} or {folder_path / 'CellAnalyzer'}. Please provide the full path to the pickle file.")

        # Load the object
        with open(pkl_name, "rb") as f:
            data = pickle.load(f)
        # Create a new instance of the class
        loaded_instance = CellAnalyzer(data['path'])
        # Update the instance with the loaded data
        loaded_instance.__dict__.update(data)

        if load_images:
            # Load the images
            loaded_instance.img_arrays = [AICSImage(loaded_instance.samples_df["filepath"][i]) for i in range(len(loaded_instance.samples_df))]
            # Convert to numpy array
            loaded_instance.img_arrays = [img.get_image_data("CZYX", T=0) for img in loaded_instance.img_arrays]

        # Cellpose model
        loaded_instance.load_cellpose_model()

        # Return the loaded instance
        return loaded_instance
    
    def load_cellpose_model(self):
        # Initializations for Cellpose
        use_GPU = core.use_gpu()
        yn = ['NO', 'YES']
        print(f'>>> GPU activated? {yn[use_GPU]}')

        # Define the model globally
        self.cellpose_model = denoise.CellposeDenoiseModel(gpu=use_GPU, model_type="cyto3",
                                            restore_type="denoise_cyto3")
        
    def read_data(self, parsing_settings="ALI"):
        """
        Parses structured microscopy .dv filenames from a given folder and returns a DataFrame
        with extracted metadata.
        Takes the path from the class initialization. Saves the DataFrame and image arrays in the object.
        
        Parameters:
            parsing_settings : str, optional
                The parsing settings to use. Options are "ALI" (default) or "jinglecells".
                "ALI" expects filenames in the format:
                <prefix>_<condition>_<temp>_<host>_<donor>_<mag>_<time>_<date>_<sample>.nd2
                "jinglecells" expects filenames in the format:
                <condition>_<donor>_<time>_<date>.<sample>_<mode1>_<mode2>.dv
        
        Returns:
            pd.DataFrame, np.array:
                DataFrame containing extracted metadata from filenames
                and a list of loaded image arrays.
        """
        input_path = self.path

        # Regex pattern to extract components
        if parsing_settings=="jinglecells":
            file_extension = ".dv"
            date_format = "%y.%m.%d"
            pattern = re.compile(
            r'(?P<condition>[a-zA-Z0-9]+)_'
            r'(?P<donor>BEC\d+)_'
            r'(?P<time>\d+h)_'
            r'(?P<date>\d{2}\.\d{2}\.\d{2})'
            r'(?:\.(?P<sample>\d+))?_'
            r'(?P<mode1>[A-Z0-9]+)_'
            r'(?P<mode2>[A-Z0-9]+)\.dv$'
            )
        else:
            file_extension = ".nd2"
            date_format = "%Y%m%d"
            pattern = re.compile(
                r'(?P<prefix>[a-zA-Z0-9]+)_'
                r'(?P<condition>[a-zA-Z0-9]+)_'
                r'(?P<temp>[0-9]+)_'
                r'(?P<host>[a-zA-Z]+)_'
                r'(?P<donor>D\d+)_'
                r'(?P<mag>\d+x)_'
                r'(?P<time>\d+hpi)_'
                r'(?P<date>\d{8})_'              # YYYYMMDD format
                r'(?P<sample>\d+)\.nd2$'
            )

        filenames = list(input_path.glob(f"*{file_extension}"))

        records = []
        for file in filenames:
            match = pattern.match(file.name)
            if match:
                data = match.groupdict()
                data["filename"] = file.name
                data["filepath"] = str(file.resolve()) # Full path for loading
                records.append(data)

        # Check if any data found
        if not records:
            raise ValueError("No suited files found.")

        # Create DataFrame
        df = pd.DataFrame(records)

        # Replace None as sample with 00
        df['sample'] = df['sample'].fillna('00')

        # Sort the DataFrame by condition, donor, time, date, and sample
        df.sort_values(by=['condition', 'donor', 'time', 'date', 'sample'], inplace=True)

        # Create a new column for "replicate", which is a unique number within each condition-donor group
        df['replicate'] = df.groupby(['condition', 'donor']).cumcount() + 1
        # Put it right after "sample"
        sample_index = df.columns.get_loc('sample') + 1
        df.insert(sample_index, 'replicate', df.pop('replicate'))
        # Also create a column for a unique sample ID
        df['sample_id'] = df["donor"] + "_" + df["replicate"].astype(str)
        # Put it right after "replicate"
        replicate_index = df.columns.get_loc('replicate') + 1
        df.insert(replicate_index, 'sample_id', df.pop('sample_id'))

        # Reset index
        df.reset_index(drop=True, inplace=True)

        # Convert date column to datetime
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'], format=date_format)

        # Load the image
        imgs = [AICSImage(df["filepath"][i]) for i in range(len(df))]
        # Convert to numpy array
        img_arrays = [img.get_image_data("CZYX", T=0) for img in imgs] # 3D image stack, all channels
        print(len(img_arrays), "images loaded")

        # Save in the object
        self.samples_df = df
        self.img_arrays = img_arrays

        return df, img_arrays

    def create_projections(self, types=["max","max","max","max"], c_axis=0, z_axis=1):
        """
        Creates projections of all channels of the images in the image list.
        
        Parameters:
            types : list of str
                The type of projection to create for each channel. Options are "max", "min", "mean", "median", "sum", "perc_X".
                "perc_X" means percentile, which picks the value at the X-th percentile; e.g. using 99 is similar to max but less sensitive to outliers.
            c_axis : int
                The axis of the channels in the image arrays. Default is 0 (CZYX).
            z_axis : int
                The axis of the z-dimension in the image arrays. Default is 1 (CZYX).

        Returns:
            projections : np.array or list of np.arrays
                The projections of the images.
        """
        # Check if the axis indices are valid
        if c_axis < 0 or c_axis > 3 or z_axis < 0 or z_axis > 3 or c_axis == z_axis:
            raise ValueError("Axis indices must be between 0 and 3 and different from each other.")

        # Test number channels
        num_channels = self.img_arrays[0].shape[c_axis]
        if len(types) != num_channels:
            raise ValueError(f"Number of types ({len(types)}) does not match number of channels ({num_channels}).")

        # Create projections
        projections = []
        for img in self.img_arrays:
            img_projections = []
            for i in range(img.shape[c_axis]):
                # Get the projection type for the current channel
                proj_type = types[i]
                # Get channel
                img_channel = np.take(img, indices=i, axis=c_axis)
                # If the z_axis was behind the c_axis, it was moved one forward when extracting the channel
                if z_axis > c_axis:
                    z_axis -= 1
                # Create the projection
                if proj_type == "max":
                    proj = np.max(img_channel, axis=z_axis)
                elif proj_type == "min":
                    proj = np.min(img_channel, axis=z_axis)
                elif proj_type == "mean":
                    proj = np.mean(img_channel, axis=z_axis)
                elif proj_type == "median":
                    proj = np.median(img_channel, axis=z_axis)
                elif proj_type == "sum":
                    proj = np.sum(img_channel, axis=z_axis)
                elif "perc_" in proj_type:
                    perc = int(proj_type.split("_")[-1])
                    proj = np.percentile(img_channel, perc, axis=z_axis)
                else:
                    raise ValueError(f"Projection type '{proj_type}' not recognized. Use 'sum', 'max', 'min', or 'mean'.")  
                # Append the projection to the list
                img_projections.append(proj)
                
            # Stack the projections along the channel axis
            img_projections = np.stack(img_projections, axis=c_axis)
            # Append the projections to the list
            projections.append(img_projections)

        # Save in the object
        self.projections = projections
        self.projections_types = types

        # Save the projection types in the DataFrame
        # self.samples_df["projection_types"] = [types for _ in range(len(self.samples_df))]
        if self.cfg_df is None:
            self.cfg_df = pd.DataFrame({"projection": types})
        else:
            self.cfg_df["projection"] = types
        
        return projections
        
    def segment_cells(self, diameter=100, channels=[0,0], log=False, calculate_neighbours=True):
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
                The masks of the segmented cells. Also saved in the object as self.masks.
            flows : np.array or list of np.arrays
                The flows of the segmented cells. Also saved in the object as self.flows.
            styles : np.array or list of np.arrays
                The styles of the segmented cells. Also saved in the object as self.styles.
            imgs_dn : np.array or list of np.arrays
                The denoised images of the segmented cells. Also saved in the object as self.imgs_dn.
            outlines : np.array or list of np.arrays
                The outlines of the segmented cells. Also saved in the object as self.outlines.

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

        img_list = self.projections
        diam_list = [diameter]*len(img_list)

        if log:
            io.logger_setup()
            print("Starting segmentation with Cellpose...")

        masks, flows, styles, imgs_dn = self.cellpose_model.eval(img_list, diameter=diam_list, channels=channels)
        outlines = [utils.masks_to_outlines(m) for m in masks]

        if log:
            print("Segmentation done. Post-processing results...")
            num_masks = len(masks)
            print(f"Number of masks: {num_masks}")

        # Make cell IDs unique (continuing from previous image)
        prev_max = 0
        new_masks = []
        max_val = sum([m.max() for m in masks]) # Maximum index is the sum of all cell ids > 0
        int_type = "int16" if max_val < 32767 else "int32"
        for i, mask in enumerate(masks):
            if log:
                print(f"Processing mask {i+1}/{num_masks} (int_type={int_type})...")
            new_mask = mask.copy().astype(int_type)
            # Add the number of cells to the DataFrame (as int)
            self.samples_df.at[i, "num_cells"] = new_mask.max()
            # Make the cell IDs unique
            new_mask += prev_max
            new_mask[new_mask == prev_max] = 0

            # Save the cell IDs in the DataFrame
            self.samples_df.at[i, "cell_id_min"] = prev_max + 1
            self.samples_df.at[i, "cell_id_max"] = new_mask.max()

            # Set the previous max to the current max
            prev_max = new_mask.max()

            # Append the new mask to the list
            new_masks.append(new_mask)

        # Make sure the columns are ints
        self.samples_df["cell_id_min"] = self.samples_df["cell_id_min"].astype(int)
        self.samples_df["cell_id_max"] = self.samples_df["cell_id_max"].astype(int)
        self.samples_df["num_cells"] = self.samples_df["num_cells"].astype(int)

        # Save the masks, flows, styles and denoised images in the object
        self.seg_channels = channels # NOTE: These are 1-indexed
        self.seg_diameter = diameter
        self.masks = new_masks
        self.flows = flows
        self.styles = styles
        self.imgs_dn = imgs_dn
        self.outlines = outlines

        if log:
            print("Post-processing done.")
            print("Creating cells DataFrame...")

        # Start a new df with a row per cell
        self.create_cells_df(log=log, calculate_neighbours=calculate_neighbours)

        return new_masks, flows, styles, imgs_dn, outlines
    
    def create_cells_df(self, log=False, calculate_neighbours=True):
        """
        Creates a DataFrame with a row for each cell in the images.
        The DataFrame contains all columns of the images df, plus specifications for each cell.
        """
        # Create a new DataFrame with a row for each cell
        cells_data = []
        for i, row in self.samples_df.iterrows():
            if log:
                print(f"Processing image {i+1}/{len(self.samples_df)} for cells DataFrame...")
            # Get the cell ID range for this image
            cell_id_min = row["cell_id_min"]
            cell_id_max = row["cell_id_max"]
            # Create a new row for each cell
            mask = self.masks[i]
            for cell_id in range(cell_id_min, cell_id_max + 1):
                new_row = row.copy()
                new_row["cell_id"] = cell_id
                # Calculate the area of the cell
                cell_mask = mask == cell_id
                cell_area = np.sum(cell_mask)
                new_row["cell_area_px"] = cell_area
                # Calculate the neighbours of the cell
                if calculate_neighbours:
                    num_neighbours = self.count_surrounding_cells(mask, cell_id, expected_diameter=self.seg_diameter)
                    new_row["num_neighbours"] = int(num_neighbours)
                # Append the new row to the list
                cells_data.append(new_row)

        # Save the DataFrame in the object
        self.cells_df = pd.DataFrame(cells_data)
        # Drop the columns that are not needed on cell level
        self.cells_df.drop(columns=["cell_id_min", "cell_id_max", "num_cells"], inplace=True)
        # Reset the index and set it to the cell_id
        self.cells_df.reset_index(drop=True, inplace=True)
        self.cells_df.set_index("cell_id", inplace=True)

    def save_segmentation_imgs(self, folder_name="segmentations", background_channels=None, overwrite=False, norm_per_img=False, norm_perc=1, scale_bar_px=150, scale_bar_um=20):
        """
        Saves the segmentation results to image files.

        Parameters:
            folder_name : str
                The name of the (sub-)folder to save the segmentation results to.
            background_channels : list of int, optional
                The channels to use for the background of the outlines. If None, uses segmentation channels.
            overwrite : bool
                Whether to overwrite existing files. Default is False.
            norm_per_img : bool
                Whether to normalize the background channels for each image separately. Default is False (normalize over all images).
            norm_perc : int
                The percentile to use for normalization of the background channels. Default is 1 (1st and 99th percentile).
                These percentiles are used as min/max, meaning values outside are clipped.
            scale_bar_px : int
                Scale bar length in pixels. Default is 100.
            scale_bar_um : int or float
                Label value in micrometers for the scale bar. Default is 20.
        """
        # Save the masks, flows, styles and denoised images
        out_folder = self.path / folder_name
        # Create the folder if it doesn't exist
        out_folder.mkdir(parents=True, exist_ok=True)

        # OUTLINES WITH CHOSEN BACKGOUND CHANNELS

        if background_channels is None:
            bg_channels = [n-1 for n in self.seg_channels]  # Decrease by 1 to make it 0-indexed
        else:
            bg_channels = [n-1 for n in background_channels] # Decrease by 1 to make it 0-indexed
        if len(bg_channels) > 3:
            raise ValueError("Number of background channels must be 3 or less (RGB channels together with outlines).")
        overall_mins, overall_maxs = {}, {}
        for bg_channel in bg_channels:
            if bg_channel < 0 or bg_channel >= len(self.projections[0]):
                raise ValueError(f"Channel {bg_channel+1} is out of bounds for the projections. Available channels: {len(self.projections[0])}.")
            # overall_mins[bg_channel] = min([img[bg_channel].min() for img in self.projections])
            # overall_maxs[bg_channel] = max([img[bg_channel].max() for img in self.projections])
            all_values = np.concatenate([img[bg_channel].ravel() for img in self.projections])
            overall_mins[bg_channel] = np.percentile(all_values, norm_perc)
            overall_maxs[bg_channel] = np.percentile(all_values, 100-norm_perc)

        # Take empty images and add channels such that it's an RGB image
        for img_num, img in enumerate(self.projections):
            outline = self.outlines[img_num]
            # Create a new image with 3 channels, to overlay the outlines
            _, h, w = img.shape
            img_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for rgb_channel, bg_channel in enumerate(bg_channels):
                channel = img[bg_channel, :, :]
                # Normalize the channel to 0-255 over all images (or over the current image if norm_per_img is True)
                used_min = overall_mins[bg_channel] if not norm_per_img else np.percentile(channel, norm_perc)
                used_max = overall_maxs[bg_channel] if not norm_per_img else np.percentile(channel, 100-norm_perc)
                channel = (channel - used_min) / (used_max - used_min) * 255
                channel = np.clip(channel, 0, 255)
                channel = channel.astype(np.uint8)
                img_rgb[:, :, rgb_channel] = channel
            # Add white outlines
            img_rgb[outline > 0] = [150, 150, 150]  # Set the outline channel to white
            img_rgb = self._add_scale_bar(img_rgb, scale_bar_px=scale_bar_px, scale_bar_um=scale_bar_um)

            # Save the image
            img_rgb = Image.fromarray(img_rgb)
            img_dir = out_folder / f"{self.samples_df['filename'][img_num]}_outlines.png"
            # Check if the file already exists
            if img_dir.exists() and not overwrite:
                print(f"File {img_dir} already exists. Saving this file was skipped.")
            else:
                img_rgb.save(img_dir)

        print(img_num+1, "outlines saved.")
        
        # MASKS

        for img_num, mask in enumerate(self.masks):
            new_mask = self.masks[img_num].copy()
            # Subtract the minimum value, but only where it is not 0
            min_val = mask[mask>0].min()
            new_mask[mask > 0] -= (min_val -1)
            # Normalize to 0-1
            new_mask = (new_mask - new_mask.min()) / (new_mask.max() - new_mask.min())
            # Map to cmap
            mapped = plt.cm.viridis(new_mask)
            mapped = (mapped[:, :, :3] * 255).astype(np.uint8)
            mapped = self._add_scale_bar(mapped, scale_bar_px=scale_bar_px, scale_bar_um=scale_bar_um)
            
            # Save the image
            mapped = Image.fromarray(mapped)
            img_dir = out_folder / f"{self.samples_df['filename'][img_num]}_masks.png"
            # Check if the file already exists
            if img_dir.exists() and not overwrite:
                print(f"File {img_dir} already exists. Saving this file was skipped.")
            else:
                mapped.save(out_folder / f"{self.samples_df['filename'][img_num]}_masks.png")

        print(img_num+1, "masks saved.")

    def calculate_single_cell_signal(self, channel_name, channel_num, dilate=None, mode="mean"):
        """
        Extracts the mean signal of each cell in the input image(s) based on the masks.
        Will populate the signal_means_dicts, signal_means_lists and signal_means_masks attributes.

        Parameters:
            channel_name : str
                Name of the channel to use for the signal calculation.
            channel_num : int
                Position of the channel to use for the signal calculation.
            dilate : int
                The amount of dilation to apply to the masks before calculating the mean signal.
                If negative, erosion is applied instead of dilation.
            mode: str
                The mode used to calculate the representative signal for each cell
                Default = "mean"; "perc_X" means X-th percentile

        Returns:
            cells_df : pd.DataFrame
                The cells DataFrame with the calculated representative signal of each cell in the image(s).
            signal_masks : list of np.array
                The masks of the signals in each image, with the same shape as the input images.
        """
        # Perform checks
        if self.cells_df is None:
            raise ValueError("cells_df is empty. Please run previous methods in the pipeline first.")
        if self.masks is None:
            raise ValueError("masks are empty. Please run segment_cells() first.")
        
        if channel_num < 1 or channel_num >= len(self.projections[0])+1:
            raise ValueError(f"Channel number {channel_num} for channel {channel_name} is out of bounds for the projections." +
                             f"Available channels: {len(self.projections[0])}.")
        if channel_num == 0:
            raise ValueError(f"You chose 0 as a channel for {channel_name}. This must be an accident." +
                             "Note that the input channels are 1-indexed, and with the input 0, you would be using -1 as index.")
        channel_num -= 1 # Decrease by 1 to make it 0-indexed

        # Register the signal mode
        self.signal_mode[channel_name] = mode

        # Make sure dilate is a valid input
        if dilate is None:
            dilate = 0

        # Perform the calculation
        cells_df = self.cells_df.copy()
        signal_dicts_out = []
        signal_lists_out = []
        signal_masks_out = []
        for img, mask in zip(self.projections, self.masks):

            # Extract the channel from the image
            img = img[channel_num]
            # Prepare the empty containers
            img_signal_dict = {} #{cell_id: np.mean(img[cell_mask_for_mean]) for cell_id in range(1, mask.max()+1)}
            img_signal_list = [] #[val for k, val in img_signal_means_dict.items()]
            img_signal_mask = np.zeros_like(img, dtype=np.float32)
            # add the signal to the dict and mask
            lowest_non_zero = mask[mask != 0].min()

            for cell_id in range(lowest_non_zero, mask.max()+1):
                cell_mask = mask == cell_id
                cell_mask_for_signal = cell_mask.copy()
                # Dilate or erode if needed
                if dilate > 0:
                    cell_mask_for_signal = morphology.binary_dilation(cell_mask_for_signal, morphology.disk(dilate))
                elif dilate < 0:
                    cell_mask_for_signal = morphology.binary_erosion(cell_mask_for_signal, morphology.disk(-dilate))

                # Calculate the mean or median signal for the cell
                if mode == "mean":
                    cell_signal = np.mean(img[cell_mask_for_signal])
                elif mode == "median":
                    cell_signal = np.median(img[cell_mask_for_signal])
                elif "perc_" in mode:
                    perc = int(mode.split("_")[-1])
                    cell_signal = np.percentile(img[cell_mask_for_signal], perc)
                else:
                    raise ValueError(f"Mode '{mode}' not recognized for channel {channel_name}. Check docstring for options.")

                # Assign the signal to the cell ID in the dict and mask
                if np.isnan(cell_signal):
                    cell_signal = 0
                img_signal_dict[cell_id] = cell_signal
                img_signal_list.append(cell_signal)
                img_signal_mask += cell_signal * cell_mask # NOTE: use un-altered mask here to have no overlaps between cells, even though for the calculation of the signal, the dilated/eroded mask was used

                # Add the signal to the cells_df
                # cells_df.loc[cell_id, channel_name+"_"+mode] = cell_signal
                cells_df.loc[cell_id, channel_name+"_signal"] = cell_signal

                # Also add the log10 of the signal
                cells_df.loc[cell_id, channel_name+"_signal_log10"] = np.log10(cell_signal) if cell_signal > 0 else 0

            signal_dicts_out.append(img_signal_dict)
            signal_lists_out.append(img_signal_list)
            signal_masks_out.append(img_signal_mask)

        self.signal_dicts[channel_name] = signal_dicts_out
        self.signal_lists[channel_name] = signal_lists_out
        self.signal_masks[channel_name] = signal_masks_out

        # Save the cells_df in the object
        self.cells_df = cells_df
        # Add the signal dilation and mode to the cfg_df
        if self.cfg_df is not None:
            self.cfg_df.loc[channel_num, "channel_name"] = channel_name
            self.cfg_df.loc[channel_num, "signal_dilate"] = dilate
            self.cfg_df.loc[channel_num, "signal_mode"] = mode
            # Move name to first column
            cols = self.cfg_df.columns.tolist()
            cols = ["channel_name"] + [col for col in cols if col != "channel_name"]
            self.cfg_df = self.cfg_df[cols]
        else:
            self.cfg_df = pd.DataFrame({channel_num: {"channel_name": channel_name, "signal_dilate": dilate, "signal_mode": mode}}).T
        # Ensure int, since with NaN values, it becomes float
        self.cfg_df["signal_dilate"] = self.cfg_df["signal_dilate"].astype("Int64")

        return cells_df, signal_masks_out

    def calculate_cell_signals(self, channels, dilate=None, mode="mean"):
        """Extracts the mean signal of each cell in the input image(s) for multiple channels based on the masks.
        Will populate the signal_dicts, signal_lists and signal_masks attributes.

        Parameters:
            channels : dict
                A dictionary with channel names as keys and channel numbers as values, indicating which channels to use for the signal calculation.
            dilate : int or dict
                The amount of dilation to apply to the masks before calculating the mean signal for each channel.
                If negative, erosion is applied instead of dilation.
                If a single int is given, it is applied to all channels.
                If a dict is given, it should have the same keys as channels, with the corresponding dilation values.
            mode: str or dict
                The mode used to calculate the representative signal for each cell. If a single str is given, it is applied to all channels. If a dict is given, it should have the same keys as channels, with the corresponding mode values.
                Default = "mean"; "perc_X" means X-th percentile

        Returns:
            cells_df : pd.DataFrame
                The cells DataFrame with the calculated representative signal of each cell in the image(s) for each channel.
            signal_masks : dict
                A dictionary with channel names as keys and lists of np.arrays as values, where each list contains the masks of the signals in each image for the corresponding channel, with the same shape as the input images.
        """
        # Perform checks
        if isinstance(dilate, int) or dilate is None:
            dilate = {name: dilate for name in channels.keys()}
        elif not all([k in channels.keys() for k in dilate.keys()]):
            raise ValueError('dilate must be a dict with the same keys as channels, or a single int to use for all channels.')
        # Fill in a default dilate in case a signal is given but not a dilate
        for name in channels.keys():
            if name not in dilate.keys():
                print(f"No dilation value given for {name}, applying no dilation/erosion for this channel.")
                dilate[name] = 0
        if isinstance(mode, str):
            mode = {name: mode for name in channels.keys()}
        elif not all([k in channels.keys() for k in mode.keys()]):
            raise ValueError('mode must be a dict with the same keys as channels, or a single str to use for all channels.')
        # Fill in a default mode in case a signal is given but not a mode
        for name in channels.keys():
            if name not in mode.keys():
                print(f"No mode value given for {name}, applying 'mean' for this channel.")
                mode[name] = "mean"

        # Calculate the signals for each channel
        for name in channels.keys():
            self.calculate_single_cell_signal(channel_name=name, channel_num=channels[name], dilate=dilate[name], mode=mode[name])

        return self.cells_df, self.signal_masks

        # cells_df = self.cells_df.copy()
        # for name, num in channels.items():
        #     # Reduce the channel number by one (0-indexed)
        #     num -= 1
        #     # Prepare the empty containers
        #     signal_dicts_out = []
        #     signal_lists_out = []
        #     signal_masks_out = []
        #     for img, mask in zip(self.projections, self.masks):
        #         # Extract the channel from the image
        #         img = img[num]
        #         # Prepare the empty containers
        #         img_signal_dict = {} #{cell_id: np.mean(img[cell_mask_for_mean]) for cell_id in range(1, mask.max()+1)}
        #         img_signal_list = [] #[val for k, val in img_signal_means_dict.items()]
        #         img_signal_mask = np.zeros_like(img, dtype=np.float32)
        #         # add the signal to the dict and mask
        #         lowest_non_zero = mask[mask != 0].min()
        #         for cell_id in range(lowest_non_zero, mask.max()+1):
        #             cell_mask = mask == cell_id
        #             cell_mask_for_signal = cell_mask.copy()
        #             # Dilate or erode if needed
        #             if dilate[name] > 0:
        #                 cell_mask_for_signal = morphology.binary_dilation(cell_mask_for_signal, morphology.disk(dilate[name]))
        #             elif dilate[name] < 0:
        #                 cell_mask_for_signal = morphology.binary_erosion(cell_mask_for_signal, morphology.disk(-dilate[name]))
        #             # Calculate the mean or median signal for the cell
        #             if mode[name] == "mean":
        #                 cell_signal = np.mean(img[cell_mask_for_signal])
        #             elif mode[name] == "median":
        #                 cell_signal = np.median(img[cell_mask_for_signal])
        #             elif "perc_" in mode[name]:
        #                 perc = int(mode[name].split("_")[-1])
        #                 cell_signal = np.percentile(img[cell_mask_for_signal], perc)
        #             else:
        #                 raise ValueError(f"Mode '{mode[name]}' not recognized. Check docstring for options.")
        #             # Assign the signal to the cell ID in the dict and mask
        #             if np.isnan(cell_signal):
        #                 cell_signal = 0
        #             img_signal_dict[cell_id] = cell_signal
        #             img_signal_list.append(cell_signal)
        #             img_signal_mask += cell_signal * cell_mask # NOTE: use un-altered mask here to have no overlaps between cells

        #             # Add the signal to the cell_df
        #             cells_df.loc[cell_id, name+"_"+mode[name]] = cell_signal
        #             # Also add the log10 of the signal
        #             cells_df.loc[cell_id, name+"_"+mode[name]+"_log10"] = np.log10(cell_signal) if cell_signal > 0 else 0

        #         signal_dicts_out.append(img_signal_dict)
        #         signal_lists_out.append(img_signal_list)
        #         signal_masks_out.append(img_signal_mask)
        
        #     self.signal_dicts[name] = signal_dicts_out
        #     self.signal_lists[name] = signal_lists_out
        #     self.signal_masks[name] = signal_masks_out

        # # Save the cells_df in the object
        # self.cells_df = cells_df

        # return self.cells_df, self.signal_masks
    
    def save_signal_masks(self, folder_name="signal_masks", overwrite=False, norm_per_img=False, scale_bar_px=150, scale_bar_um=20):
        """
        Saves the signal masks to a file.

        Parameters:
            folder_name : str
                The name of the (sub-)folder to save the signal masks to.
            overwrite : bool
                Whether to overwrite existing files. Default is False.
            scale_bar_px : int
                Scale bar length in pixels. Default is 100.
            scale_bar_um : int or float
                Label value in micrometers for the scale bar. Default is 20.
        """
        # Checks
        if not self.signal_masks:
            print("No signal masks found. Please run calculate_cell_signals() first.")
            return

        # Create the folder if it doesn't exist
        out_folder = self.path / folder_name
        out_folder.mkdir(parents=True, exist_ok=True)

        for signal_name, masks_list in self.signal_masks.items():
            overall_min = min([mask.min() for mask in masks_list])
            overall_max = max([mask.max() for mask in masks_list])
            for img_num, mask in enumerate(masks_list):
                # Normalize to 0-1
                new_mask = mask.copy()
                used_min = overall_min if not norm_per_img else new_mask.min()
                used_max = overall_max if not norm_per_img else new_mask.max()
                new_mask = (new_mask - used_min) / (used_max - used_min)
                # Map to cmap
                mapped = plt.cm.viridis(new_mask)
                mapped = (mapped[:, :, :3] * 255).astype(np.uint8)

                # Create white outlines
                outline = self.outlines[img_num]
                mapped[outline] = [255, 255, 255]
                mapped = self._add_scale_bar(mapped, scale_bar_px=scale_bar_px, scale_bar_um=scale_bar_um)

                # Save the image
                mapped = Image.fromarray(mapped)
                img_dir = out_folder / f"{self.samples_df['filename'][img_num]}_{signal_name}_mask.png"
                # Check if the file already exists
                if img_dir.exists() and not overwrite:
                    print(f"File {img_dir} already exists. Saving this file was skipped.")
                else:
                    mapped.save(img_dir)

            print(img_num+1, f"masks for signal '{signal_name}' saved.")
        
    def bin_single_cell_signal(self, signal, use_log=True, thresh=None):
        """
        Bins the signal of each cell in the cell_df dataframe based on one or multiple thresholds.
        The bins will be called "negative" and "positive" if only one threshold is given,
        "negative", "partial" and "positive" if three thresholds are given, and will be numbered otherwise.
        Also creates masks with the binning for each cell in the cells_df DataFrame, with the value being the bin number (0="negative" etc.)

        Parameters:
            signal: str
                The name of the signal to bin. Must be same as used for calculate_cell_signals().
            use_log: bool
                Whether to use the log10 of the signal for binning.
            thresh: float, list of floats or None
                The threshold(s) to use for binning the signal.
                If None, Otsu's method is used to determine a single threshold.
                If a single float is given, it is used as the threshold.
                If a list, all values are used to divide the signal int len(thresh)+1 bins.

        Returns:
            cells_df : pd.DataFrame
                The cells DataFrame with the added column for the binned signal.
            bin_masks : list of np.array
                The masks of the binned signals in each image, with the same shape as the input images, where the value of each cell is the bin number.
        """
        # Perform checks
        if self.cells_df is None:
            raise ValueError("cells_df is empty. Please run calculate_cell_signals() first.")
        if self.masks is None:
            raise ValueError("masks are empty. Please run segment_cells() first.")

        column = f"{signal}_signal{'_log10' if use_log else ''}"

        if column not in self.cells_df.columns:
            raise ValueError(f"Column '{column}' not found in cells_df. Please run calculate_cell_signals() first.")

        # Determine threshold if not given
        use_otsu = False
        if thresh is None:
            signals = np.array(self.cells_df[column].dropna())
            if signals.size == 0:
                raise ValueError(f"No non-NA values found in column '{column}' for thresholding.")
            # Use Otsu's method to find the threshold
            thresh = threshold_otsu(signals)
            use_otsu = True
            print(f"Using Otsu's method to find the threshold for {column}: {thresh}")
        else:
            print(f"Using manual threshold(s) for {column}: {thresh}")

        # Use thresholds if given
        if isinstance(thresh, (int, float)):
            thresh = [thresh]
        if len(thresh) == 1:
            bins = ["negative", "positive"]
        elif len(thresh) == 2:
            bins = ["negative", "partial", "positive"]
        else:
            bins = [str(i + 1) for i in range(len(thresh) + 1)]
        bin_nums = {bin_name: i + 1 for i, bin_name in enumerate(bins)}  # 1-indexed, 0 kept for background
        # Sort thresholds if there are multiple given
        thresh = sorted(thresh)
        
        # Create the bin column in the cells_df
        self.bins[signal] = bins
        col_name = f"{signal}_bin" #{'_log10' if use_log else ''}_bin"
        self.cells_df[col_name] = bins[0]  # Initialize the column with the first bin
        for t, bin_name in zip(thresh, bins[1:]):
            # Set the bin for the cells that are above the threshold
            print(f"Thresholding {bin_name} at {t}")
            self.cells_df.loc[self.cells_df[column] > t, col_name] = bin_name

        # Add a column with the thresholds and parameters used to the cfg_df
        if self.cfg_df is not None:
            if "channel_name" in self.cfg_df.columns and signal in self.cfg_df["channel_name"].values:
                channel_num = self.cfg_df[self.cfg_df["channel_name"] == signal].index[0]
                thresh_type = f"{'otsu' if use_otsu else 'manual'}"
                # self.cfg_df.loc[channel_num, f'threshold_{thresh_type}'] = str(thresh)
                self.cfg_df.loc[channel_num, 'bin_use_log'] = str(use_log)
                self.cfg_df.loc[channel_num, 'bin_threshold_type'] = thresh_type
                self.cfg_df.loc[channel_num, f'bin_threshold(s)'] = str(thresh)
            else:
                print(f"Warning: signal '{signal}' not found in cfg_df. Thresholds not saved in cfg_df.")
        else:
            print("Warning: cfg_df is None. Thresholds not saved in cfg_df.")

        # Create masks for the bins
        print(f"Creating bin masks for signal '{signal}'...")
        bin_masks_out = []
        for mask in self.masks:
            bins_mask = np.zeros_like(mask, dtype=np.uint16)
            cell_ids = np.unique(mask)
            cell_ids = cell_ids[cell_ids != 0]
            for cell_id in cell_ids:
                cell_mask = mask == cell_id
                cell_bin = self.cells_df.loc[cell_id, col_name]
                bin_num = bin_nums[cell_bin]
                bins_mask[cell_mask] = bin_num
            bin_masks_out.append(bins_mask)
            print(f"Created bin mask for image {len(bin_masks_out)}")

        self.bin_masks[signal] = bin_masks_out
    
        return self.cells_df, bin_masks_out
    
    def bin_cell_signals(self, signals, use_log=True, thresh=None):
        """
        Bins the signals of each cell in the cell_df dataframe based on one or multiple thresholds for multiple signals.
        The bins will be called "negative" and "positive" if only one threshold is given,
        "negative", "partial" and "positive" if three thresholds are given, and will be numbered otherwise.
        Also creates masks with the binning for each cell in the cells_df DataFrame, with the value being the bin number (0="negative" etc.)

        Parameters:
            signals: list of str or str
                The names of the signals to bin. Must be same as used for calculate_cell_signals().
                (the "log" suffix will be added automatically based on the use_log parameter)
                If a single string is given, it is used as the signal to bin.
            use_log: bool or dict
                Whether to use the log10 of the signal for binning.
                If a single bool is given, it is applied to all signals.
                If a dict is given, it should have the same keys as signals, with the corresponding bool values.
            thresh: float, list of floats or None OR dict with signal names as keys and float, list of floats or None as values
                The threshold(s) to use for binning the signals.
                If only one input is given for thresh, it is applied to all signals.
                If a dict is given, it should have the same keys as signals, with the corresponding threshold(s) values.
                Cases:
                    - None -> Otsu's method is used to determine a single threshold for each signal.
                    - single float (or int) -> is given, it is used as the threshold (into 2 bins).
                    - list -> all values are used to divide the signal int len(thresh)+1 bins.

        Returns:
            cells_df : pd.DataFrame
                The cells DataFrame with the added columns for the binned signals.
            bin_masks : dict
                A dictionary with signal names as keys and lists of np.arrays as values, where each list contains the masks of the binned signals in each image for the corresponding signal, with the same shape as the input images, where the value of each cell is the bin number.
        """
        # Perform checks
        if isinstance(signals, str):
            signals = [signals]
        if isinstance(use_log, bool):
            use_log = {signal: use_log for signal in signals}
        elif not all([s in use_log.keys() for s in signals]):
            raise ValueError('use_log must be a dict with the same keys as signals, or a single bool to use for all signals.')
        if isinstance(thresh, (int, float)) or thresh is None:
            thresh = {signal: thresh for signal in signals}
        elif not all([s in thresh.keys() for s in signals]):
            raise ValueError('thresh must be a dict with the same keys as signals, or a single value to use for all signals.')
        
        # Bin each signal
        for signal in signals:
            self.bin_single_cell_signal(signal=signal, use_log=use_log[signal], thresh=thresh[signal])

        return self.cells_df, self.bin_masks

    def save_bin_masks(self, folder_name="binned_signals", overwrite=False, scale_bar_px=150, scale_bar_um=20):
        """
        Saves the binned signal masks to a file.

        Parameters:
            folder_name : str
                The name of the (sub-)folder to save the binned signal masks to.
            overwrite : bool
                Whether to overwrite existing files. Default is False.
            scale_bar_px : int
                Scale bar length in pixels. Default is 100.
            scale_bar_um : int or float
                Label value in micrometers for the scale bar. Default is 20.
        """
        # Checks
        if not self.bin_masks:
            print("No binned masks found. Please run bin_cell_signal() first.")
            return

        # Create the folder if it doesn't exist
        out_folder = self.path / folder_name
        out_folder.mkdir(parents=True, exist_ok=True)

        for signal_name, masks_list in self.bin_masks.items():
            bins = self.bins[signal_name]
            for img_num, mask in enumerate(masks_list):
                # Normalize to 0-1; according to the number of bins
                new_mask = mask.copy()
                new_mask = new_mask / (len(bins))
                # Map to cmap
                mapped = plt.cm.viridis(new_mask)
                mapped = (mapped[:, :, :3] * 255).astype(np.uint8)

                # Create white outlines
                outline = self.outlines[img_num]
                mapped[outline] = [255, 255, 255]
                mapped = self._add_scale_bar(mapped, scale_bar_px=scale_bar_px, scale_bar_um=scale_bar_um)

                # Save the image
                mapped = Image.fromarray(mapped)
                img_dir = out_folder / f"{self.samples_df['filename'][img_num]}_{signal_name}_bin_mask.png"
                # Check if the file already exists
                if img_dir.exists() and not overwrite:
                    print(f"File {img_dir} already exists. Saving this file was skipped.")
                else:
                    mapped.save(img_dir)

            print(img_num+1, f"bin masks for signal '{signal_name}' saved.")        

    def create_populations(self, signals, signal_tags=None, col_name=None):
        """
        Analyzes the bins in cells_df, creates a column with the combination of any number of signals (= populations).
        By default, the populations are named according to the first three letters of the signal names and the first three letters of the bin names.
        If number of signals <= 3, also creates RGB images for the populations in the cells_df, with RGB in order of the signals given.

        Parameters:
            signals : list of str
                The names of the signals to combine. Must be same as used for bin_cell_signal().
            signal_tags : list of str optional
                The tags to use for each signal in the population name.
                If None, the first three letters of the signal name will be used.
            col_name : str, optional
                The name of the column to create in the cells_df DataFrame.
                If None, the column name will be the combination of the signal names (first three letters each) with "_pop" appended.

        Returns:
            cells_df : pd.DataFrame
                The cells DataFrame with the bins and populations as columns.
        """
        # Check if the signals are in the cells_df
        for i, signal in enumerate(signals):
            if signal[-4:] != "_bin":
                signal = signal+"_bin"
                signals[i] = signal
            if signal not in self.cells_df.columns:
                raise ValueError(f"Bin column for signal '{signal}' not found in cells_df. Please run calculate_cell_signals() and bin_cell_signal() first.")

        # Create a new column for the population, and temp columns
        pop_col_name = col_name if col_name is not None else "_".join([s[:3] for s in signals]) + "_pop"
        # Create the population column by combining the signals; can be overridden by signal_tags input
        signal_tags = signal_tags if signal_tags is not None else [s[:3] for s in signals]
        for i, s in enumerate(signals):
            # Catch potentail NAs
            self.cells_df[s] = self.cells_df[s].fillna("NA")
            # Create a temporary column with the signal tag and the bin name (e.g. "cil-neg"), to then combine into the population column
            self.cells_df["temp_" + s] = signal_tags[i] + "-" + self.cells_df[s].astype(str).str[:3]
        self.cells_df[pop_col_name] = self.cells_df[["temp_"+s for s in signals]].agg("_".join, axis=1)
        # Drop the temp columns
        self.cells_df.drop(columns=["temp_"+s for s in signals], inplace=True)
        # Make sure the column is a string
        self.cells_df[pop_col_name] = self.cells_df[pop_col_name].astype(str)

        return self.cells_df
    
    def save_population_masks(self, signals, folder_name="populations", overwrite=False, rgb_channels=(0, 1, 2), scale_bar_px=150, scale_bar_um=20):

        # Checks
        if not isinstance(signals, (list, tuple)) or len(signals) < 2 or len(signals) > 3:
            raise ValueError("signals must be a list of 2 or 3 signal names")
        if len(signals) > len(rgb_channels):
            raise ValueError("Number of signals exceeds number of RGB channels provided")

        # Create output folder
        pop_name = "_".join([s[:3] for s in signals]) + "_pop"
        out_folder = self.path / folder_name / pop_name
        out_folder.mkdir(parents=True, exist_ok=True)

        # Create RGB images for the populations (e.g. {"negative": 0, "positive": 1}
        signal_bin_nums = {s: {bin_name: i for i, bin_name in enumerate(self.bins[s])} for s in signals}

        i = 0
        for masks in zip(*[self.bin_masks[s] for s in signals], self.outlines):
            *mask_list, outline = masks
            img_rgb = np.zeros((*mask_list[0].shape, 3), dtype=np.uint8)
            # Note: masks are 1-indexed, so 0 is background
            for idx, s in enumerate(signals):
                # Scale up to 255 (excluding 0)
                img_rgb[:, :, rgb_channels[idx]] = mask_list[idx] * 255 // (len(signal_bin_nums[s]))
            # Add white outlines
            img_rgb[outline] = [255, 255, 255]
            img_rgb = self._add_scale_bar(img_rgb, scale_bar_px=scale_bar_px, scale_bar_um=scale_bar_um)

            # Save the image
            img_rgb = Image.fromarray(img_rgb)
            img_dir = out_folder / f"{self.samples_df['filename'][i]}_{pop_name}.png"
            if img_dir.exists() and not overwrite:
                print(f"File {img_dir} already exists. Saving this file was skipped.")
            else:
                img_rgb.save(img_dir)

            i += 1
        print(i, "populations saved.")

        signal_bin_nums = {s: {bin_name: i for i, bin_name in enumerate(sorted(self.bins[s]))} for s in signals}
        signal_levels = {s: [(i + 1) * 255 // len(signal_bin_nums[s]) for i in range(len(signal_bin_nums[s]))] for s in signals}

        # Create legend
        combos = list(itertools.product(*[signal_bin_nums[s].keys() for s in signals]))

        longest_len = 0
        fig, ax = plt.subplots(figsize=(4, len(combos) * 0.3))
        for i_combo, combo in enumerate(combos):
            rgb = [0, 0, 0]
            for idx, s in enumerate(signals):
                rgb[rgb_channels[idx]] = signal_levels[s][signal_bin_nums[s][combo[idx]]] / 255
            label = " | ".join([f"{s}_{combo[idx]}" for idx, s in enumerate(signals)])
            longest_len = max(longest_len, len(label))
            ax.add_patch(plt.Rectangle((0, i_combo), 1, 1, color=rgb))
            ax.text(1.1, i_combo + 0.5, label, va='center')

        ax.set_xlim(0, 1 + longest_len * 0.3)
        ax.set_ylim(0, len(combos))
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(out_folder / f"{pop_name}_legend.png", dpi=150)
        plt.close(fig)
        print(f"Legend saved.")

        print("Folder:", out_folder)

    @staticmethod
    def count_surrounding_cells(mask, cell_id, expected_diameter):
        """
        Count how many other cells are within a circular region around a given cell,
        adjusted for edge effects (partial circle outside image).
        
        Parameters:
            mask : np.ndarray
                2D array where each cell has a unique integer ID (background = 0).
            cell_id : int
                The ID of the cell to analyze.
            expected_diameter : float
            The expected diameter of a cell (in pixels).
        
        Returns:
            float
                Scaled number of unique other cell IDs within the defined circle.
        """
        props = regionprops((mask == cell_id).astype(np.uint8))
        if not props:
            raise ValueError(f"Cell ID {cell_id} not found in mask.")
        region = props[0]

        # Cell centroid (y, x)
        cy, cx = region.centroid
        # Equivalent circular radius
        area = np.sum(mask == cell_id)
        radius = np.sqrt(area / np.pi)
        extended_radius = radius + expected_diameter/2
        # print("Cell ID:", cell_id, "Centroid:", (cy, cx), "Radius:", radius, "Extended radius:", extended_radius)

        y_indices, x_indices = np.indices(mask.shape)
        dist = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)
        circle_mask = dist <= extended_radius

        # Fraction of circle inside the image (edge correction)
        # Theoretical total circle area:
        circle_area = np.pi * extended_radius**2
        # Pixels actually inside image:
        inside_area = np.sum(circle_mask)
        inside_fraction = inside_area / circle_area

        # Get IDs within the circle
        surrounding_ids = np.unique(mask[circle_mask])
        surrounding_ids = surrounding_ids[(surrounding_ids != 0) & (surrounding_ids != cell_id)]

        # Edge-corrected estimate
        corrected_count = len(surrounding_ids) / inside_fraction if inside_fraction > 0 else np.nan
        return corrected_count

    @staticmethod
    def _add_scale_bar(img_rgb, scale_bar_px=150, scale_bar_um=20):
        """
        Adds a semi-transparent scale bar and label to the bottom-right corner of an RGB image.

        Parameters:
            img_rgb : np.ndarray
                RGB image array with shape (H, W, 3).
            scale_bar_px : int
                Bar length in pixels. If <= 0, no scale bar is drawn.
            scale_bar_um : int or float
                Label value in micrometers displayed as '<value> um'.

        Returns:
            np.ndarray
                RGB image with scale bar overlay.
        """
        if scale_bar_px is None or scale_bar_px <= 0:
            return img_rgb
        if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
            raise ValueError("img_rgb must be an RGB image with shape (H, W, 3).")

        # Derive geometry from image size so the bar keeps a similar visual weight
        # across images with different resolutions.
        h, w = img_rgb.shape[:2]
        margin = max(8, int(round(min(h, w) * 0.03)))
        bar_height = max(2, int(round(h * 0.01)))
        bar_len = int(round(scale_bar_px))

        # Clamp bar length so it always fits inside the image with left/right padding.
        bar_len = min(bar_len, w - 2 * margin)
        if bar_len < 2:
            return img_rgb

        # Bottom-right placement with a small inset from the border.
        x2 = w - margin
        x1 = x2 - bar_len
        y2 = h - margin
        y1 = max(0, y2 - bar_height)

        label = f"{scale_bar_um} um"

        # Draw on a separate RGBA overlay so we can use transparency (alpha), then
        # composite once onto the original image.
        base = Image.fromarray(img_rgb).convert("RGBA")
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Semi-transparent white scale bar.
        draw.rectangle([(x1, y1), (x2, y2)], fill=(255, 255, 255, 180))

        # Use a larger font (about 2x the previous default appearance) for readability.
        font_size = max(12, int(round(bar_height * 2.0)))
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except OSError:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except OSError:
                font = ImageFont.load_default()

        # Position text just above the bar and right-aligned to the bar end.
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_left, text_top, text_right, text_bottom = text_bbox
        text_w = text_right - text_left
        text_h = text_bottom - text_top
        text_gap = max(3, bar_height // 1.5)
        text_x = max(margin, x2 - text_w)
        # Place text by aligning the *visible* text bottom above the bar.
        # This avoids vertical drift from font-specific bbox offsets.
        text_y = max(0, y1 - text_gap - text_bottom)

        # Add a subtle dark backing box so the label stays readable on bright images.
        box_x1 = max(0, text_x + text_left - 2)
        box_y1 = max(0, text_y + text_top - 1)
        box_x2 = min(w, text_x + text_right + 2)
        box_y2 = min(h, text_y + text_bottom + 1)
        draw.rectangle([(box_x1, box_y1), (box_x2, box_y2)], fill=(0, 0, 0, 90))
        draw.text((text_x, text_y), label, fill=(255, 255, 255, 180), font=font)

        # Merge overlay and return an RGB uint8 image, consistent with the save pipeline.
        out = Image.alpha_composite(base, overlay).convert("RGB")
        return np.array(out, dtype=np.uint8)


#####################################################################################


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
        

    ### HELPER FUNCTIONS ###

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