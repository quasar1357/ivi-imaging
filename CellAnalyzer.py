from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from cellpose import core, denoise, io, utils
from skimage import morphology
from skimage.filters import threshold_otsu
import pandas as pd
import seaborn as sns
from pathlib import Path
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
        self.signals_dicts = {}
        self.signals_lists = {}
        self.signals_masks = {}
        self.cells_df = None
        self.signal_mode = "mean"  # Default mode for signal calculation
        self.bin_masks = {}
        self.pop_imgs = []

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
            self.cells_df.to_csv(output_path / "cells_metadata.csv", index=False)

        # Save the object
        data_to_save = {
            'path': self.path,
            'df': self.samples_df,
            'projections': self.projections,
            'projections_types': self.projections_types,
            'seg_channels': self.seg_channels,
            'seg_diameter': self.seg_diameter,
            'masks': self.masks,
            'flows': self.flows,
            'styles': self.styles,
            'imgs_dn': self.imgs_dn,
            'outlines': self.outlines,
            'signal_means_dicts': self.signals_dicts,
            'signals_lists': self.signals_lists,
            'signals_masks': self.signals_masks,
            'cells_df': self.cells_df,
            'signal_mode': self.signal_mode,
            'bin_masks': self.bin_masks,
            'pop_imgs': self.pop_imgs
        }

        with open(output_path / "CellAnalyzer.pkl", "wb") as f:
            pickle.dump(data_to_save, f)

    @staticmethod
    def load(pkl_name=None, use_GPU=True):
        """
        Loads the object from a pickle file.
        
        Parameters:
            name : str
                The name of the file to load from.

        Returns:
            CellAnalyzer instance
                The loaded CellAnalyzer instance.
        """
        # Load the object
        with open(pkl_name, "rb") as f:
            data = pickle.load(f)
        # Create a new instance of the class
        loaded_instance = CellAnalyzer(data['path'])
        # Update the instance with the loaded data
        loaded_instance.__dict__.update(data)

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
        
    def read_data(self):
        """
        Parses structured microscopy .dv filenames from a given folder and returns a DataFrame
        with extracted metadata.
        
        Parameters:
            input_folder (str or Path): Path to the folder containing .dv files
        
        Returns:
            pd.DataFrame:
                DataFrame containing extracted metadata from filenames
        """
        input_path = self.path
        filenames = list(input_path.glob("*.dv"))

        # Regex pattern to extract components
        pattern = re.compile(
            r'(?P<condition>[a-zA-Z0-9]+)_'
            r'(?P<donor>BEC\d+)_'
            r'(?P<time>\d+h)_'
            r'(?P<date>\d{2}\.\d{2}\.\d{2})'
            r'(?:\.(?P<sample>\d+))?_'
            r'(?P<mode1>[A-Z0-9]+)_'
            r'(?P<mode2>[A-Z0-9]+)\.dv$'
        )

        records = []
        for file in filenames:
            match = pattern.match(file.name)
            if match:
                data = match.groupdict()
                data["filename"] = file.name
                data["filepath"] = str(file.resolve()) # Full path for loading
                records.append(data)

        # Create DataFrame
        df = pd.DataFrame(records)

        # Replace None as sample with 00
        df['sample'] = df['sample'].fillna('00')

        # Sort the DataFrame by condition, donor, time, date, and sample
        df.sort_values(by=['condition', 'donor', 'time', 'date', 'sample'], inplace=True)

        # Create a new column for "replicate", which is u unique number within each condition-donor group
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

        if not df.empty:
            df['date'] = pd.to_datetime(df['date'], format="%y.%m.%d")

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
                The type of projection to create for each channel. Options are "max", "min", "mean", "median", "sum".

        Returns:
            projections : np.array or list of np.arrays
                The projections of the images.
        """
        # Test number channels
        num_channels = self.img_arrays[0].shape[0]
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
                img_channel = img[i]
                # If the z_axis was behind the c_axis, it was moved one forward
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
        self.samples_df["projection_types"] = [types for _ in range(len(self.samples_df))]
        
        return projections
        
    def segment_cells(self, diameter=100, channels=[0,0], log=False):
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

        img_list = self.projections
        diam_list = [diameter]*len(img_list)

        masks, flows, styles, imgs_dn = self.cellpose_model.eval(img_list, diameter=diam_list, channels=channels)
        outlines = [utils.masks_to_outlines(m) for m in masks]

        # Make cell IDs unique (continuing from previous image)
        prev_max = 0
        new_masks = []
        for i, mask in enumerate(masks):
            new_mask = mask.copy().astype("int16")
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
        self.seg_channels = channels
        self.seg_diameter = diameter
        self.masks = new_masks
        self.flows = flows
        self.styles = styles
        self.imgs_dn = imgs_dn
        self.outlines = outlines

        # Start a new df with a row per cell
        self.create_cells_df()

        return new_masks, flows, styles, imgs_dn, outlines
    
    def create_cells_df(self):
        """
        Creates a DataFrame with a row for each cell in the images.
        The DataFrame contains all columns of the images df, plus specifications for each cell.
        """
        # Create a new DataFrame with a row for each cell
        cells_data = []
        for i, row in self.samples_df.iterrows():
            # Get the cell ID range for this image
            cell_id_min = row["cell_id_min"]
            cell_id_max = row["cell_id_max"]
            # Create a new row for each cell
            for cell_id in range(cell_id_min, cell_id_max + 1):
                new_row = row.copy()
                new_row["cell_id"] = cell_id
                cells_data.append(new_row)

        # Save the DataFrame in the object
        self.cells_df = pd.DataFrame(cells_data)
        # Drop the columns that are not needed on cell level
        self.cells_df.drop(columns=["cell_id_min", "cell_id_max", "num_cells"], inplace=True)
        # Reset the index and set it to the cell_id
        self.cells_df.reset_index(drop=True, inplace=True)
        self.cells_df.set_index("cell_id", inplace=True)

    def save_segmentation_imgs(self, folder_name="segmentations", background_channels=None, overwrite=False):
        """
        Saves the segmentation results to a file.
        """
        # Save the masks, flows, styles and denoised images
        out_folder = self.path / folder_name
        # Check if the folder already exists
        # if out_folder.exists():
        #     raise ValueError(f"Folder {folder_name} already exists. Please choose a different name.")
        # Create the folder if it doesn't exist
        out_folder.mkdir(parents=True, exist_ok=True)

        # OUTLINES WITH CHOSEN BACKGOUND CHANNELS
        if background_channels is None:
            background_channels = self.seg_channels
        else:
            background_channels = [n-1 for n in background_channels] # Decrease by 1 to make it 0-indexed
        # Take the denoised images and add channels such that it's a RGB image
        for i, img in enumerate(self.projections):
            outline = self.outlines[i]
            # Create a new image with 3 channels, to overlay the outlines
            _, h, w = img.shape
            img_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for c in range(2):
                channel = img[c, :, :]
                np.moveaxis(channel, 0, -1)  # Move the channel axis to the last dimension
                # Normalize the channel to 0-255
                channel = (channel - channel.min()) / (channel.max() - channel.min()) * 255
                channel = np.clip(channel, 0, 255)
                channel = channel.astype(np.uint8)
                img_rgb[:, :, c] = channel  # Copy existing channels
            # Add the outlines as the blue channel
            img_rgb[outline > 0, 2] = 255

            # Save the image
            img_rgb = Image.fromarray(img_rgb)
            img_dir = out_folder / f"{self.samples_df['filename'][i]}_outlines.png"
            # Check if the file already exists
            if img_dir.exists() and not overwrite:
                print(f"File {img_dir} already exists. Saving this file was skipped.")
            else:
                img_rgb.save(img_dir)
        print("Outlines saved.")
        
        # MASKS
        for i, mask in enumerate(self.masks):
            new_mask = self.masks[i].copy()
            # Subtract the minimum value, but only where it is not 0
            min_val = mask[mask>0].min()
            new_mask[mask > 0] -= (min_val -1)
            # Normalize to 0-1
            new_mask = (new_mask - new_mask.min()) / (new_mask.max() - new_mask.min())
            # Map to cmap
            mapped = plt.cm.viridis(new_mask)
            mapped = (mapped[:, :, :3] * 255).astype(np.uint8)
            
            # Save the image
            mapped = Image.fromarray(mapped)
            img_dir = out_folder / f"{self.samples_df['filename'][i]}_masks.png"
            # Check if the file already exists
            if img_dir.exists() and not overwrite:
                print(f"File {img_dir} already exists. Saving this file was skipped.")
            else:
                mapped.save(out_folder / f"{self.samples_df['filename'][i]}_masks.png")
        print("Masks saved.")

    def calculate_cell_signals(self, channels, dilate=None, mode="mean"):
        """
        Extracts the mean signal of each cell in the input image(s) based on the masks.
        Will populate the signal_means_dicts, signal_means_lists and signal_means_masks attributes.

        Parameters:
            channels : dict {str: int}
                Name and position of the channels to use for the mean signal calculation.
            dilate : list of int
                The amount of dilation to apply to the masks before calculating the mean signal.
                One value per each channel.
                If negative, erosion is applied instead of dilation.

        Returns:
            cells_df : pd.DataFrame
                The cells DataFrame with the mean signal of each cell in the image(s).
            signals_masks : dict {str: list of np.array}
                The masks of the signals for each channel, with the same shape as the input images.
                Keys are the channel names, values are lists of masks for each image.
        """
        # Register the signal mode
        self.signal_mode = mode

        if dilate is None:
            dilate = {name: 0 for name in channels.keys()}
        elif isinstance(dilate, int):
            dilate = {name: dilate for name in channels.keys()}
        elif not all([k in channels.keys() for k in dilate.keys()]):
            raise ValueError('dilate must be a list of the same length as channels, or a single int for all channels.')

        for name, num in channels.items():
            # Reduce the channel number by one (0-indexed)
            num -= 1
            # Prepare the empty containers
            signals_dicts_out = []
            signals_lists_out = []
            signals_masks_out = []
            for img, mask in zip(self.projections, self.masks):
                # Extract the channel from the image
                img = img[num]
                # Prepare the empty containers
                img_signals_dict = {} #{cell_id: np.mean(img[cell_mask_for_mean]) for cell_id in range(1, mask.max()+1)}
                img_signals_list = [] #[val for k, val in img_signal_means_dict.items()]
                img_signals_mask = np.zeros_like(img, dtype=np.float32)
                # add the signal to the dict and mask
                lowest_non_zero = mask[mask != 0].min()
                for cell_id in range(lowest_non_zero, mask.max()+1):
                    cell_mask = mask == cell_id
                    cell_mask_for_signal = cell_mask.copy()
                    if dilate[name] > 0:
                        cell_mask_for_signal = morphology.binary_dilation(cell_mask_for_signal, morphology.disk(dilate[name]))
                    elif dilate[name] < 0:
                        cell_mask_for_signal = morphology.binary_erosion(cell_mask_for_signal, morphology.disk(-dilate[name]))
                    if mode == "mean":
                        cell_signal = np.mean(img[cell_mask_for_signal])
                    elif mode == "median":
                        cell_signal = np.median(img[cell_mask_for_signal])
                    else:
                        raise ValueError(f"Mode '{mode}' not recognized. Check docstring for options.")
                    img_signals_dict[cell_id] = cell_signal
                    img_signals_list.append(cell_signal)
                    img_signals_mask += cell_signal * cell_mask

                    # Add the signal to the cell_df
                    self.cells_df.loc[self.cells_df["cell_id"] == cell_id, name+"_"+mode] = cell_signal
                    # Also add the log10 of the signal
                    self.cells_df.loc[self.cells_df["cell_id"] == cell_id, name+"_"+mode+"_log10"] = np.log10(cell_signal) if cell_signal > 0 else 0

                signals_dicts_out.append(img_signals_dict)
                signals_lists_out.append(img_signals_list)
                signals_masks_out.append(img_signals_mask)
        
            self.signals_dicts[name] = signals_dicts_out
            self.signals_lists[name] = signals_lists_out
            self.signals_masks[name] = signals_masks_out

        return self.cells_df, self.signals_masks
        
    def bin_cell_signal(self, signal, use_log=True, thresh=None, col_name=None):
        """
        Bins the signal of each cell in the cell_df dataframe based on one or multiple thresholds.
        The bins will be called "negative" and "positive" if only one threshold is given,
        "negative", "medium" and "positive" if three thresholds are given, and will be numbered otherwise.
        Also creates masks with the binning for each cell in the cells_df DataFrame, with the value being the bin number (0="negative" etc.)

        Parameters:
            signal: str
                The name of the signal to bin. Must be same as used for calculate_cell_signals().
            use_log: bool
                Whether to use the log10 of the signal for binning.
            thresh: float, list of floats or None
                The threshold(s) to use for binning the signal.
                If None, Otsu's method is used on the mean signal of the cells.
                If a list, three or more bins are created.
            col_name: str, optional
                The name of the column to create in the cells_df DataFrame.
                If None, the column name will be the signal name with "_bin" appended.

        Returns:
            None
        """
        column = f'{signal}_{self.signal_mode}{"_log10" if use_log else ""}'

        if column not in self.cells_df.columns:
            raise ValueError(f"Column '{column}' not found in cells_df. Please run calculate_cell_signals() first.")

        if thresh is None:
            signals = np.array(self.cells_df[column].dropna())
            # Use Otsu's method to find the threshold
            thresh = threshold_otsu(signals)
            print(f"Using Otsu's method to find the threshold for {column}: {thresh}")

        # Use thresholds if given
        if isinstance(thresh, float):
            thresh = [thresh]
            bins = ["negative", "positive"]
            bin_nums = {"negative": 1, "positive": 2}
        else:
            thresh.sort()
            bins = ["negative", "medium", "positive"] if len(thresh) == 2 else [i+1 for i in range(len(thresh)+2)]
            bin_nums = {bin_name: i+1 for i, bin_name in enumerate(bins)}
        col_name = col_name if col_name is not None else signal
        self.cells_df[col_name] = bins[0]  # Initialize the column with the first bin
        for thresh, bin_name in zip(thresh, bins[1:]):
            # Set the bin for the cells that are above the threshold
            self.cells_df.loc[self.cells_df[column] > thresh, col_name] = bin_name

        # Create masks for the bins
        bin_masks_out = []
        for mask in self.masks:
            bins_mask = np.zeros_like(mask, dtype=np.float32)
            lowest_non_zero = mask[mask != 0].min()
            for cell_id in range(lowest_non_zero, mask.max()+1):
                cell_mask = mask == cell_id
                cell_bin = self.cells_df.loc[self.cells_df["cell_id"] == cell_id, col_name].values[0]
                bin_num = bin_nums[cell_bin]
                bins_mask += bin_num * cell_mask
            bin_masks_out.append(bins_mask)

        self.bin_masks[signal] = bin_masks_out

    def create_populations(self, signal1, signal2):
        """
        Analyzes the bins in cells_df, creates a column with the combination of the two signals (= populations).
        The populations are named according to the first three letters of the signal names and the first three letters of the bin names.
        Also creates RGB images for the populations in the cells_df, with the red channels for signal1 and green for signal2.
        Further creates a DataFrame with the counts of the populations as a matrix between the two signals.

        Parameters:
            signal1 : str
                The name of the first signal to use for the population analysis. Must be according to the bins.
            signal2 : str
                The name of the second signal to use for the population analysis. Must be according to the bins.

        Returns:
            cells_df : pd.DataFrame
                The cells DataFrame with the bins and populations as columns.
            pop_imgs : list of np.array
                The RGB images of the populations, with red channel for signal1, green for signal2 and blue = outlines.
        """
        # Check if the signals are in the cells_df
        if signal1 not in self.cells_df.columns or signal2 not in self.cells_df.columns:
            raise ValueError(f"Signals '{signal1}' and/or '{signal2}' not found in cells_df. Please run calculate_cell_signals() and bin_cell_signal() first.")

        # Create a new column for the population, and temp columns
        pop_col_name = "population"
        self.cells_df["temp_signal1"] = signal1[:3]  # First three letters of signal1
        self.cells_df["temp_signal2"] = signal2[:3]  # First three letters of signal2
        # Create the population column by combining the two signals
        self.cells_df[pop_col_name] = self.cells_df["temp_signal1"] + "_" + self.cells_df[signal1].str.slice(0,3) + "_" + self.cells_df["temp_signal2"] + "_" + self.cells_df[signal2].str.slice(0,3)
        # Drop the temporary columns
        self.cells_df.drop(columns=["temp_signal1", "temp_signal2"], inplace=True)
        # Make sure the column is a string
        self.cells_df[pop_col_name] = self.cells_df[pop_col_name].astype(str)

        # Create RGB images for the populations
        signal1_bin_nums = {bin_name: i for i, bin_name in enumerate(self.cells_df[signal1].unique())}
        signal2_bin_nums = {bin_name: i for i, bin_name in enumerate(self.cells_df[signal2].unique())}
        pop_imgs_out = []
        for mask1, mask2, outline in zip(self.bin_masks[signal1], self.bin_masks[signal2], self.outlines):
            img_rgb = np.zeros((*mask1.shape, 3), dtype=np.uint8)
            img_rgb[:, :, 0] = mask1 * 255 // (len(signal1_bin_nums))  # Scale to 0-255
            img_rgb[:, :, 1] = mask2 * 255 // (len(signal2_bin_nums))  # Scale to 0-255
            # Add the outlines as the blue channel
            img_rgb[:, :, 2] = outline * 255  # Set blue channel to 255 where the cell outlines are
            pop_imgs_out.append(img_rgb)
        self.pop_imgs = pop_imgs_out

        return self.cells_df, pop_imgs_out

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