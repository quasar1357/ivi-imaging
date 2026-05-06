from PIL import Image, ImageDraw, ImageFont
import numpy as np
from matplotlib import pyplot as plt
from cellpose import core, denoise, io, utils
from skimage import morphology
from skimage.measure import regionprops, regionprops_table
from skimage.filters import threshold_otsu
import pandas as pd
import seaborn as sns
from pathlib import Path
import itertools
import re
from aicsimageio import AICSImage
import pickle
from tifffile.tifffile import imwrite

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
        self.cells_df = None
        self.signals = {}
        # self.signal_lists = {}
        self.signal_masks = {}
        self.signal_mode = {}
        self.bins = {}
        self.bin_masks = {}
        self.channels_df = None

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
        if self.channels_df is not None:
            self.channels_df.to_csv(output_path / "metadata_cfg.csv", index=False)

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
            'cells_df': self.cells_df,
            'signals': self.signals,
            # 'signal_lists': self.signal_lists,
            'signal_masks': self.signal_masks,
            'signal_mode': self.signal_mode,
            'bins': self.bins,
            'bin_masks': self.bin_masks,
            'channels_df': self.channels_df
        }

        with open(output_path / "CellAnalyzer.pkl", "wb") as f:
            pickle.dump(data_to_save, f)

    @staticmethod
    def load(pkl_name=None, load_images=False):
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
            # Mark all samples as loaded when images are loaded from pickle
            try:
                loaded_instance.samples_df["is_loaded"] = True
            except Exception:
                loaded_instance.samples_df["is_loaded"] = [True] * len(loaded_instance.samples_df)

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

    def read_data(self, parsing_settings="ALI",
                  regex_pattern=None, date_format=None, file_extension=None,
                  reset=False):
        """
        Parses structured microscopy .dv filenames from a given folder and returns a DataFrame
        with extracted metadata.
        Takes the path from the class initialization. Saves the DataFrame in the object.

        Parameters:
            parsing_settings : str, optional
                The parsing settings to use. Options are "ALI" (default), "jinglecells", or "custom".
                "ALI" expects filenames in the format:
                    <prefix>_<condition>_<temp>_<host>_<donor>_<mag>_<time>_<date>_<sample>.nd2
                "jinglecells" expects filenames in the format:
                    <condition>_<donor>_<time>_<date>.<sample>_<mode1>_<mode2>.dv
                If "custom" is chosen, the regex pattern, date format and file extension must be given as arguments as well.
            regex_pattern : str, optional
                The regex pattern to use for parsing the filenames. Only needed if parsing_settings is "custom".
            date_format : str, optional
                The date format to use for parsing the date in the filenames. Only needed if parsing_settings is "custom".
                Examples: "%y.%m.%d" for "23.05.01", "%Y%m%d" for "20230501", "%d-%m-%Y" for "01-05-2023".
            file_extension : str, optional
                The file extension to look for in the filenames. Only needed if parsing_settings is "custom".
                Examples: ".dv", ".nd2", ".tif".
            reset : bool, optional
                If True, resets all existing data in the object (images, projections, segmentations, cells_df) and starts fresh with only the new samples_df. Default is False.

        Returns:
            pd.DataFrame, np.array:
                DataFrame containing extracted metadata from filenames
                and a list of loaded image arrays.
        """
        input_path = self.path

        # Do resets if requested (and check consistency with existing data)
        if reset:
            self.img_arrays = None
            self.projections = None
            self.projections_types = None
            self.masks = None
            self.flows = None
            self.styles = None
            self.imgs_dn = None
            self.outlines = None
            self.cells_df = None
        elif self.samples_df is not None:
            raise ValueError("samples_df already exists. Use reset=True to clear existing data and start fresh with the new samples_df. This will also reset all downstream data (images, projections, segmentations, cells_df).")

        # Checks
        checks = [regex_pattern is not None, date_format is not None, file_extension is not None]
        if any(checks) and parsing_settings != "custom":
            raise ValueError("regex_pattern, date_format and file_extension should only be provided if parsing_settings is 'custom'.")
        if parsing_settings == "custom" and not all(checks):
            raise ValueError("For 'custom' parsing, all of regex_pattern, date_format, and file_extension must be provided.")
        
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
        elif parsing_settings=="ALI":
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
        elif parsing_settings=="custom":
            if not all([regex_pattern, date_format, file_extension]):
                raise ValueError("For 'custom' parsing, all of regex_pattern, date_format, and file_extension must be provided.")
            pattern = re.compile(regex_pattern)
        
        self.file_extension = file_extension
        self.date_format = date_format
        self.regex_pattern = regex_pattern

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

        # Save in the object
        self.samples_df = df

        # Track whether images have been loaded into memory
        # This is used for sequential processing of image loading/projection/segmentation
        self.samples_df["is_loaded"] = False

        return df

    def load_images(self, num_imgs=None, overwrite=False):
        """
        Loads the images from the file paths in samples_df and saves them in the object.
        Skips images that have already been loaded unless overwrite=True.
        Tracks progress in samples_df["is_loaded"].

        Parameters:
            num_imgs : int
                The maximum number of images to load. Default is None (all remaining images).
            overwrite : bool
                If True, re-project all images, discarding existing projections. Required when changing types. Default is False.
        """
        # Basic checks
        if self.samples_df is None:
            raise ValueError("samples_df is empty. Please run read_data() first.")

        n_total = len(self.samples_df)

        # Initialize img_arrays tracking if needed
        if self.img_arrays is None:
            self.img_arrays = [None] * n_total
        elif len(self.img_arrays) != n_total and not overwrite:
            raise ValueError(
                "Length mismatch between img_arrays and samples_df. "
                "Use overwrite=True in load_images() to start a clean image-loading round."
            )

        # Handle overwrite: reset loaded images
        if overwrite:
            prev_loaded = int(self.samples_df["is_loaded"].sum()) if "is_loaded" in self.samples_df.columns else 0
            if prev_loaded > 0:
                print(f"Using overwrite=True: resetting loaded images ({prev_loaded}/{n_total} images were loaded). Starting a new load round.")
            self.img_arrays = [None] * n_total
            self.samples_df["is_loaded"] = False

        # Keep tracking aligned with current img_arrays
        self.samples_df["is_loaded"] = [ia is not None for ia in self.img_arrays]

        # Determine which images still need to be loaded
        pending = self.samples_df.index[~self.samples_df["is_loaded"]].tolist()
        n_loaded_before = int(self.samples_df["is_loaded"].sum())

        if not pending:
            if overwrite:
                # If overwrite was requested but nothing pending now, be explicit
                print(f"Using overwrite=True: no images to load after reset. Status: {n_loaded_before}/{n_total} loaded.")
            else:
                print(f"Image loading status: {n_loaded_before}/{n_total} images already loaded. Nothing to do. Use overwrite=True to reload images.")
            return self.img_arrays

        if num_imgs is not None:
            pending = pending[:num_imgs]

        print(f"Loading {len(pending)} image(s). Status before this run: {n_loaded_before}/{n_total} loaded.")

        # Load only pending images
        for idx in pending:
            img = AICSImage(self.samples_df.at[idx, "filepath"])
            arr = img.get_image_data("CZYX", T=0)
            self.img_arrays[idx] = arr
            self.samples_df.at[idx, "is_loaded"] = True

        n_loaded_after = int(self.samples_df["is_loaded"].sum())
        print(f"{len(pending)} image(s) loaded. Status after this run: {n_loaded_after}/{n_total}.")

        return self.img_arrays

    def create_projections(self, types=["max","max","max","max"], c_axis=0, z_axis=1, num_imgs=None, overwrite=False):
        """
        Creates projections of all channels of the images in the image list.
        Skips images that already have a projection unless overwrite=True.
        Tracks progress in samples_df["has_projection"].
        
        Parameters:
            types : list of str
                The type of projection to create for each channel. Options are "max", "min", "mean", "median", "sum", "perc_X".
                "perc_X" means percentile, which picks the value at the X-th percentile; e.g. using 99 is similar to max but less sensitive to outliers.
            c_axis : int
                The axis of the channels in the image arrays. Default is 0 (CZYX).
            z_axis : int
                The axis of the z-dimension in the image arrays. Default is 1 (CZYX).
            num_imgs : int
                The maximum number of new projections to create. Default is None (all remaining images).
            overwrite : bool
                If True, re-project all images, discarding existing projections. Required when changing types. Default is False.
                If False, changing types is not allowed.

        Returns:
            projections : list of np.arrays
                The projections of the images (may contain None for images not yet projected).
        """
        # Basic checks
        if self.samples_df is None:
            raise ValueError("samples_df is empty. Please run read_data() first.")
        if self.img_arrays is None:
            raise ValueError("img_arrays is empty. Please run load_images() first.")

        n_imgs_total = len(self.samples_df)

        # Derive `is_loaded` from `img_arrays` (single source of truth)
        if len(self.img_arrays) != n_imgs_total:
            raise ValueError(
                "Length mismatch between img_arrays and samples_df. "
                "Use overwrite=True in load_images() to start a clean image-loading round."
            )
        self.samples_df["is_loaded"] = [ia is not None for ia in self.img_arrays]
        if self.projections is None:
            self.samples_df["has_projection"] = False # Set to False for all rows if projections not initialized yet
        elif len(self.projections) != n_imgs_total:
            raise ValueError(
                "Length mismatch between projections and samples_df. "
                "Use overwrite=True in create_projections() to start a clean projection round."
            )
        else:
            self.samples_df["has_projection"] = [p is not None for p in self.projections]

        # Check if the axis indices are valid
        if c_axis < 0 or c_axis > 3 or z_axis < 0 or z_axis > 3 or c_axis == z_axis:
            raise ValueError("Axis indices must be between 0 and 3 and different from each other.")
        # Test number channels
        num_channels = self.img_arrays[0].shape[c_axis]
        if len(types) != num_channels:
            raise ValueError(f"Number of types ({len(types)}) does not match number of channels ({num_channels}).")

        # Handle parameter consistency or reset projections (if overwrite=True)
        if overwrite:
            prev_proj = int(self.samples_df["has_projection"].sum()) if "has_projection" in self.samples_df.columns else 0
            if prev_proj > 0:
                print(f"Using overwrite=True: resetting projections ({prev_proj}/{n_imgs_total} images had projections). Starting a new projection round.")
            if self.projections_types is not None and self.projections_types != types:
                print(f"You changed projection types from {self.projections_types} to {types}.")
                print("\nWarning: if you have already performed downstream analysis, these might now be inconsistent with the new projections. Consider re-running those steps as well.")
            self.projections = [None] * n_imgs_total
            self.projections_types = None
            self.samples_df["has_projection"] = False
        else:
            if self.projections_types is not None and self.projections_types != types:
                raise ValueError(
                    f"Projection types {types} differ from existing {self.projections_types}. "
                    "Use overwrite=True to replace all projections with the new types."
                )

        # Initialize projection container if needed
        if self.projections is None:
            self.projections = [None] * n_imgs_total

        # Keep tracking aligned with current projections
        self.samples_df["has_projection"] = [p is not None for p in self.projections]

        # Project only images that are loaded and not projected yet
        pending = self.samples_df.index[
            (self.samples_df["is_loaded"]) & (~self.samples_df["has_projection"])
        ].tolist()
        pending = [idx for idx in pending if self.img_arrays[idx] is not None]
        n_projected_before = int(self.samples_df["has_projection"].sum())
        n_loaded = int(self.samples_df["is_loaded"].sum())
        n_not_loaded = n_imgs_total - n_loaded

        if not pending:
            if overwrite:
                # If overwrite was requested but nothing pending now, be explicit
                print(f"Using overwrite=True: no projections to create after reset. Status: {n_projected_before}/{n_imgs_total} projected.")
            else:
                if n_loaded > 0 and n_projected_before == n_loaded and n_loaded < n_imgs_total:
                    print(
                        f"All loaded images are already projected ({n_projected_before}/{n_imgs_total}), but {n_not_loaded} image(s) are not loaded yet. "
                        "Nothing was changed.\nComplete loading before continuing, as downstream analysis requires projections for all images."
                    )
                else:
                    print(f"Projection status: {n_projected_before}/{n_imgs_total} images already have projections. Nothing was changed. Use overwrite=True to start a new projection round.")
            return self.projections

        if num_imgs is not None:
            pending = pending[:num_imgs]

        print(
            f"Creating projections for {len(pending)} image(s). "
            f"Status before this run: {n_projected_before}/{n_imgs_total}."
        )

        #z_axis shifts by -1 after channel extraction if it came after c_axis
        _z_axis = z_axis - 1 if z_axis > c_axis else z_axis

        # Create projections for pending images
        for idx in pending:
            img = self.img_arrays[idx]
            img_projections = []
            for i in range(img.shape[c_axis]):
                # Get the projection type for the current channel
                proj_type = types[i]
                img_channel = np.take(img, indices=i, axis=c_axis)
                if proj_type == "max":
                    proj = np.max(img_channel, axis=_z_axis)
                elif proj_type == "min":
                    proj = np.min(img_channel, axis=_z_axis)
                elif proj_type == "mean":
                    proj = np.mean(img_channel, axis=_z_axis)
                elif proj_type == "median":
                    proj = np.median(img_channel, axis=_z_axis)
                elif proj_type == "sum":
                    proj = np.sum(img_channel, axis=_z_axis)
                elif "perc_" in proj_type:
                    perc = int(proj_type.split("_")[-1])
                    proj = np.percentile(img_channel, perc, axis=_z_axis)
                else:
                    raise ValueError(f"Projection type '{proj_type}' not recognized. Use 'sum', 'max', 'min', or 'mean'.")
                img_projections.append(proj)

            self.projections[idx] = np.stack(img_projections, axis=c_axis)
            self.samples_df.at[idx, "has_projection"] = True

        n_done = int(self.samples_df["has_projection"].sum())
        print(f"{len(pending)} projection(s) created. Status after this run: {n_done}/{n_imgs_total}.")

        # Save the projection types
        self.projections_types = types

        # Save the projection types in channels_df
        if self.channels_df is None:
            self.channels_df = pd.DataFrame({"projection": types})
        else:
            self.channels_df["projection"] = types

        # Inform user if some images are not loaded into memory yet
        if n_loaded < n_imgs_total:
            print()
            print(
                f"Important: Not all images are currently loaded ({n_loaded}/{n_imgs_total}). "
                "Please note that projections can only be created for loaded images. Run load_images() to load the remaining images."
            )

        return self.projections
    
    def directly_load_and_project(self, types=["max", "max", "max", "max"], c_axis=0, z_axis=1,
                                  num_imgs=None, overwrite=False):
        """
        Loads the images from the file paths in samples_df, directly creates projections and only saves those in the object.
        Skips images that already have a projection unless overwrite=True.
        Tracks progress in samples_df["has_projection"].

        Parameters:
            types : list of str
                The projection types for each channel.
            c_axis : int
                The axis of the channels in the image arrays. Default is 0 (CZYX).
            z_axis : int
                The axis of the z-dimension in the image arrays. Default is 1 (CZYX).
            num_imgs : int
                The maximum number of images to load. Default is None (all remaining images).
            overwrite : bool
                If True, re-project all images, discarding existing projections. Required when changing types. Default is False.
        """
        # Basic checks
        if self.samples_df is None:
            raise ValueError("samples_df is empty. Please run read_data() first.")

        n_total = len(self.samples_df)

        # Check if the axis indices are valid
        if c_axis < 0 or c_axis > 3 or z_axis < 0 or z_axis > 3 or c_axis == z_axis:
            raise ValueError("Axis indices must be between 0 and 3 and different from each other.")
        # Test number channels - only if at least one image is already loaded, otherwise we cannot know the number of channels yet
        if self.img_arrays is not None and len(self.img_arrays) > 0 and self.img_arrays[0] is not None:
            num_channels = self.img_arrays[0].shape[c_axis]
            if len(types) != num_channels:
                raise ValueError(f"Number of types ({len(types)}) does not match number of channels ({num_channels}).")

        # Handle parameter consistency or reset projections (if overwrite=True)
        if overwrite:
            prev_proj = int(self.samples_df["has_projection"].sum()) if "has_projection" in self.samples_df.columns else 0
            if prev_proj > 0:
                print(f"Using overwrite=True: resetting projections ({prev_proj}/{n_total} images had projections). Starting a new projection round.")
            if self.projections_types is not None and self.projections_types != types:
                print(f"You changed projection types from {self.projections_types} to {types}.")
                print("\nWarning: if you have already performed downstream analysis, these might now be inconsistent with the new projections. Consider re-running those steps as well.")
            self.projections = [None] * n_total
            self.projections_types = None
            self.samples_df["has_projection"] = False
        else:
            if self.projections_types is not None and self.projections_types != types:
                raise ValueError(
                    f"Projection types {types} differ from existing {self.projections_types}. "
                    "Use overwrite=True to replace all projections with the new types."
                )

        # Initialize projection container if needed
        if self.projections is None:
            self.projections = [None] * n_total

        # Keep tracking aligned with current projections
        self.samples_df["has_projection"] = [p is not None for p in self.projections]

        # Determine which images still need to be projected (only those that are not projected yet, regardless of whether they are loaded or not)
        pending = self.samples_df.index[~self.samples_df["has_projection"]].tolist()
        n_projected_before = int(self.samples_df["has_projection"].sum())

        if not pending:
            if overwrite:
                # If overwrite was requested but nothing pending now, be explicit
                print(f"Using overwrite=True: no images to load after reset. Status: {n_projected_before}/{n_total} projected.")
            else:
                print(f"Image projection status: {n_projected_before}/{n_total} images already projected. Nothing to do. Use overwrite=True to re-project images.")
            return self.projections

        if num_imgs is not None:
            pending = pending[:num_imgs]

        print(
            f"Loading and creating projections for {len(pending)} image(s). "
            f"Status before this run: {n_projected_before}/{n_total}."
        )

        # Projection preparations: z_axis shifts by -1 after channel extraction if it came after c_axis
        _z_axis = z_axis - 1 if z_axis > c_axis else z_axis
        
        # Load only pending images, project directly and save projections (without saving the full images in img_arrays to save memory)
        for idx in pending:
            img_in = AICSImage(self.samples_df.at[idx, "filepath"])
            img = img_in.get_image_data("CZYX", T=0)
            img_projections = []
            for i in range(img.shape[c_axis]):
                # Get the projection type for the current channel
                proj_type = types[i]
                img_channel = np.take(img, indices=i, axis=c_axis)
                if proj_type == "max":
                    proj = np.max(img_channel, axis=_z_axis)
                elif proj_type == "min":
                    proj = np.min(img_channel, axis=_z_axis)
                elif proj_type == "mean":
                    proj = np.mean(img_channel, axis=_z_axis)
                elif proj_type == "median":
                    proj = np.median(img_channel, axis=_z_axis)
                elif proj_type == "sum":
                    proj = np.sum(img_channel, axis=_z_axis)
                elif "perc_" in proj_type:
                    perc = int(proj_type.split("_")[-1])
                    proj = np.percentile(img_channel, perc, axis=_z_axis)
                else:
                    raise ValueError(f"Projection type '{proj_type}' not recognized. Use 'sum', 'max', 'min', or 'mean'.")
                img_projections.append(proj)

            self.projections[idx] = np.stack(img_projections, axis=c_axis)
            self.samples_df.at[idx, "has_projection"] = True

            # Clean up to avoid keeping large arrays in memory
            try:
                del img
                del img_in
            except Exception:
                pass

        n_projected_after = int(self.samples_df["has_projection"].sum())
        print(f"{len(pending)} image(s) projected. Status after this run: {n_projected_after}/{n_total}.")

        # Save projection types
        self.projections_types = types

        # Save the projection types in channels_df
        if self.channels_df is None:
            self.channels_df = pd.DataFrame({"projection": types})
        else:
            self.channels_df["projection"] = types

        return self.projections

    def segment_cells(self, diameter=100, channels=[0,0], log=False, calculate_neighbours=True, num_imgs=None, overwrite=False):
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
            num_imgs : int
                The maximum number of images to segment in this run. Default is None (all eligible images).
            overwrite : bool
                If True, clears existing segmentation and re-runs from scratch with the current diameter/channels.
                If False, changing diameter/channels is not allowed.

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

        # Basic checks
        if self.samples_df is None:
            raise ValueError("samples_df is empty. Please run read_data() first.")
        if self.projections is None:
            raise ValueError("projections are empty. Please run create_projections() first.")

        n_imgs_total = len(self.samples_df)

        # Derive tracking from current data structures (single source of truth)
        if len(self.projections) != n_imgs_total:
            raise ValueError(
                "Length mismatch between projections and samples_df. "
                "Please align data or re-run create_projections()."
            )
        self.samples_df["has_projection"] = [p is not None for p in self.projections] # Refresh projection tracking in case it got out of sync
        if self.masks is None:
            self.samples_df["has_segmentation"] = False # Set to False for all rows if masks not initialized yet
        elif len(self.masks) != n_imgs_total:
            raise ValueError(
                "Length mismatch between masks and samples_df. "
                "Use overwrite=True in segment_cells() to start a clean segmentation round."
            )
        else:
            self.samples_df["has_segmentation"] = [m is not None for m in self.masks]

        # Handle parameter consistency or reset segmentation (if overwrite=True)
        if overwrite:
            prev_seg = int(self.samples_df["has_segmentation"].sum()) if "has_segmentation" in self.samples_df.columns else 0
            if prev_seg > 0:
                print(f"Using overwrite=True: resetting segmentation ({prev_seg}/{n_imgs_total} images had segmentation). Starting a new segmentation round.")
            if (self.seg_diameter is not None and self.seg_diameter != diameter) or (self.seg_channels is not None and self.seg_channels != channels):
                print(f"You changed segmentation parameters (diameter and/or channels).")
                print("\nWarning: if you have already performed downstream analysis, these might now be inconsistent with the new segmentations. Consider re-running those steps as well.")
            self.masks = [None] * n_imgs_total
            self.flows = [None] * n_imgs_total
            self.styles = [None] * n_imgs_total
            self.imgs_dn = [None] * n_imgs_total
            self.outlines = [None] * n_imgs_total
            self.cells_df = None
            self.samples_df["has_segmentation"] = False
            for col in ["cell_id_min", "cell_id_max", "num_cells"]:
                self.samples_df[col] = pd.NA
        else:
            if self.seg_channels is not None and self.seg_channels != channels:
                raise ValueError(
                    f"Segmentation channels {channels} differ from existing {self.seg_channels}. "
                    "Use overwrite=True to replace all segmentation results with the new channels."
                )
            if self.seg_diameter is not None and self.seg_diameter != diameter:
                raise ValueError(
                    f"Segmentation diameter {diameter} differs from existing {self.seg_diameter}. "
                    "Use overwrite=True to replace all segmentation results with the new diameter."
                )

        # Initialize segmentation containers if needed
        if self.masks is None:
            self.masks = [None] * n_imgs_total
        if self.flows is None:
            self.flows = [None] * n_imgs_total
        if self.styles is None:
            self.styles = [None] * n_imgs_total
        if self.imgs_dn is None:
            self.imgs_dn = [None] * n_imgs_total
        if self.outlines is None:
            self.outlines = [None] * n_imgs_total

        # Keep tracking aligned with current masks
        self.samples_df["has_segmentation"] = [m is not None for m in self.masks]

        # Segment only images that have projections and are not segmented yet
        pending = self.samples_df.index[
            (self.samples_df["has_projection"]) & (~self.samples_df["has_segmentation"])
        ].tolist()
        pending = [idx for idx in pending if self.projections[idx] is not None]
        n_seg_done_before = int(self.samples_df["has_segmentation"].sum())
        n_projected = int(self.samples_df["has_projection"].sum())
        n_unprojected = n_imgs_total - n_projected

        if not pending:
            if overwrite:
                print("No images pending for segmentation (no projected images available to segment in this run). Segmentation state was reset because overwrite=True.")
            else:
                if n_projected > 0 and n_seg_done_before == n_projected and n_projected < n_imgs_total:
                    print(
                        f"All projected images are already segmented ({n_projected}/{n_imgs_total}), but {n_unprojected} image(s) still have no projections. "
                        "Nothing was changed." \
                        "\nThis is a partial state: complete projections before continuing, as downstream analysis (signals/binning/populations) "
                        "should only be run after *all* images are projected and segmented."
                    )
                else:
                    print(
                        f"Status: {n_projected}/{n_imgs_total} projected, {n_seg_done_before}/{n_imgs_total} segmented."
                        "\nNothing was changed. Use overwrite=True to start a new segmentation round. Or run create_cells_df() if there was an error during creation of cells_df."
                    )
            return self.masks, self.flows, self.styles, self.imgs_dn, self.outlines

        if num_imgs is not None:
            pending = pending[:num_imgs]

        print(
            f"Starting segmentation run for {len(pending)} image(s). "
            f"Status before this run: {n_projected}/{n_imgs_total} projected, {n_seg_done_before}/{n_imgs_total} segmented."
        )

        img_list = [self.projections[idx] for idx in pending]
        diam_list = [diameter] * len(img_list)

        if log:
            io.logger_setup()
            print(f"Step 1: Running Cellpose (diameter={diameter}, channels={channels})...")

        masks, flows, styles, imgs_dn = self.cellpose_model.eval(img_list, diameter=diam_list, channels=channels)
        outlines = [utils.masks_to_outlines(m) for m in masks]

        if log:
            num_masks = len(masks)
            print(f"Step 2: Post-processing results... Number of masks: {num_masks}")

        # Validate existing cells_df against samples_df before appending new cells
        if self.cells_df is not None and not self.cells_df.empty:
            existing_cell_ids = set(self.cells_df.index)
            segmented_rows = self.samples_df[self.samples_df["has_segmentation"]]
            for _, row in segmented_rows.iterrows():
                if pd.isna(row["num_cells"]) or int(row["num_cells"]) == 0:
                    continue
                if pd.isna(row["cell_id_min"]) or pd.isna(row["cell_id_max"]):
                    raise ValueError(
                        "samples_df is inconsistent: segmented row has NA in cell_id_min/cell_id_max. "
                        "Restart segmentation with overwrite=True to start a new segmentation round."
                    )
                cell_id_min = int(row["cell_id_min"])
                cell_id_max = int(row["cell_id_max"])
                if cell_id_min not in existing_cell_ids or cell_id_max not in existing_cell_ids:
                    raise ValueError(
                        "cells_df is inconsistent with samples_df: at least one stored cell_id_min/cell_id_max "
                        "is missing in cells_df. Rebuild cells_df before continuing segmentation."
                    )

        # Make cell IDs unique (continuing from current cells_df if available)
        prev_max = int(self.cells_df.index.max()) if self.cells_df is not None and not self.cells_df.empty else 0
        max_val = prev_max + sum([m.max() for m in masks]) # Maximum index is previous max + sum of new ids > 0
        int_type = "int16" if max_val < 32767 else "int32"
        for i, (idx, mask) in enumerate(zip(pending, masks)):
            if log:
                print(f"     Processing mask {i+1}/{num_masks} (int_type={int_type})...")
            new_mask = mask.copy().astype(int_type)
            # Add the number of cells to the DataFrame (as int)
            num_cells = int(new_mask.max())
            self.samples_df.at[idx, "num_cells"] = num_cells
            # Make the cell IDs unique
            new_mask += prev_max
            new_mask[new_mask == prev_max] = 0

            # Save the cell IDs in the DataFrame
            if num_cells > 0:
                self.samples_df.at[idx, "cell_id_min"] = prev_max + 1
                self.samples_df.at[idx, "cell_id_max"] = int(new_mask.max())
            else:
                self.samples_df.at[idx, "cell_id_min"] = pd.NA
                self.samples_df.at[idx, "cell_id_max"] = pd.NA

            # Set the previous max to the current max
            prev_max = new_mask.max()

            # Save segmentation outputs in-place
            self.masks[idx] = new_mask
            self.flows[idx] = flows[i]
            self.styles[idx] = styles[i]
            self.imgs_dn[idx] = imgs_dn[i]
            self.outlines[idx] = outlines[i]
            self.samples_df.at[idx, "has_segmentation"] = True

        # Keep nullable integer dtype to allow non-segmented rows
        self.samples_df["cell_id_min"] = self.samples_df["cell_id_min"].astype("Int64")
        self.samples_df["cell_id_max"] = self.samples_df["cell_id_max"].astype("Int64")
        self.samples_df["num_cells"] = self.samples_df["num_cells"].astype("Int64")

        # Save segmentation settings
        self.seg_channels = channels # NOTE: These are 1-indexed
        self.seg_diameter = diameter

        n_done = int(self.samples_df["has_segmentation"].sum())

        if log:
            print("Step 3: Creating/updating cells DataFrame...")

        # Append only the new cells to cells_df
        self.create_cells_df(log=log, calculate_neighbours=calculate_neighbours, sample_indices=pending, append=not overwrite)

        print(f"Segmentation run complete. Status after this run: {n_projected}/{n_imgs_total} projected, {n_done}/{n_imgs_total} segmented.")

        if n_projected < n_imgs_total:
            print()
            print(
                f"Important: Not all images currently have projections ({n_projected}/{n_imgs_total}), and segmentation can only process projected images.\n"
                f"However, downstream analysis is intended to be run on all images together. "
                "Run downstream analysis only after *all* images are projected and segmented."
            )

        return self.masks, self.flows, self.styles, self.imgs_dn, self.outlines
    
    def create_cells_df(self, log=False, calculate_neighbours=True, sample_indices=None, append=False):
        """
        Creates a DataFrame with a row for each cell in the images.
        The DataFrame contains all columns of the images df, plus specifications for each cell.
        """
        # Create a new DataFrame with a row for each cell
        cells_data = []
        if sample_indices is None:
            sample_iter = self.samples_df.iterrows()
        else:
            sample_iter = self.samples_df.loc[sample_indices].iterrows()

        n = 0
        tot = len(sample_indices) if sample_indices is not None else len(self.samples_df)
        for i, row in sample_iter:
            n += 1
            if log:
                print(f"     Processing image {n}/{tot} for cells DataFrame...")
            if self.masks is None or i >= len(self.masks) or self.masks[i] is None:
                continue
            # Get the cell ID range for this image
            cell_id_min = row["cell_id_min"]
            cell_id_max = row["cell_id_max"]
            if pd.isna(cell_id_min) or pd.isna(cell_id_max):
                continue
            cell_id_min = int(cell_id_min)
            cell_id_max = int(cell_id_max)
            if cell_id_max < cell_id_min:
                continue
            # Create a new row for each cell
            mask = self.masks[i]
            for cell_id in range(cell_id_min, cell_id_max + 1):
                new_row = row.copy()
                new_row["cell_id"] = cell_id
                # Calculate the area of the cell
                # cell_mask = mask == cell_id
                # cell_area = np.sum(cell_mask)
                # new_row["cell_area_px"] = cell_area
                # Calculate the neighbours of the cell
                if calculate_neighbours:
                    num_neighbours = self.count_surrounding_cells(mask, cell_id) #, expected_diameter=self.seg_diameter)
                    new_row["num_neighbours"] = int(num_neighbours)
                # Append the new row to the list
                cells_data.append(new_row)

        # Save the DataFrame in the object
        if len(cells_data) == 0:
            if not append or self.cells_df is None:
                self.cells_df = pd.DataFrame()
            return

        new_cells_df = pd.DataFrame(cells_data)
        # Drop the columns that are not needed on cell level
        new_cells_df.drop(columns=["cell_id_min", "cell_id_max", "num_cells", "has_projection", "has_segmentation"], inplace=True, errors="ignore")
        # Reset the index and set it to the cell_id
        new_cells_df.reset_index(drop=True, inplace=True)
        new_cells_df.set_index("cell_id", inplace=True)

        if append and self.cells_df is not None and not self.cells_df.empty:
            self.cells_df = pd.concat([self.cells_df, new_cells_df], axis=0)
        else:
            self.cells_df = new_cells_df

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
        if self.projections is None or self.projections[0] is None:
            raise ValueError("Projections are empty. Please run create_projections() first.")
        if self.outlines is None or self.outlines[0] is None:
            raise ValueError("Outlines are empty. Please run segment_cells() first.")
        if self.masks is None or self.masks[0] is None:
            raise ValueError("Masks are empty. Please run segment_cells() first.")
        overall_mins, overall_maxs = {}, {}
        for bg_channel in bg_channels:
            if bg_channel < 0 or bg_channel >= len(self.projections[0]):
                raise ValueError(f"Channel {bg_channel+1} is out of bounds for the projections. Available channels: {len(self.projections[0])}.")
            # overall_mins[bg_channel] = min([img[bg_channel].min() for img in self.projections])
            # overall_maxs[bg_channel] = max([img[bg_channel].max() for img in self.projections])
            all_values = np.concatenate([img[bg_channel].ravel() for img in self.projections if img is not None])
            overall_mins[bg_channel] = np.percentile(all_values, norm_perc)
            overall_maxs[bg_channel] = np.percentile(all_values, 100-norm_perc)

        # Take empty images and add channels such that it's an RGB image
        n = 0
        tot = 0
        for img_num, outline in enumerate(self.outlines):
            tot += 1
            img = self.projections[img_num]
            if img is None or outline is None:
                continue
            n += 1
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
            img_dir = out_folder / f"{Path(self.samples_df['filename'][img_num]).stem}_outlines.png"
            # Check if the file already exists
            if img_dir.exists() and not overwrite:
                print(f"File {img_dir} already exists. Saving this file was skipped.")
            else:
                img_rgb.save(img_dir)

        print(f"{n}/{tot} outlines saved.")

        # MASKS
        n = 0
        tot = 0
        for img_num, mask in enumerate(self.masks):
            tot += 1 # count all masks
            if mask is None:
                continue
            n += 1 # count non-empty masks
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
            img_dir = out_folder / f"{Path(self.samples_df['filename'][img_num]).stem}_masks.png"
            # Check if the file already exists
            if img_dir.exists() and not overwrite:
                print(f"File {img_dir} already exists. Saving this file was skipped.")
            else:
                mapped.save(img_dir)

        print(f"{n}/{tot} masks saved.")

    def calculate_single_cell_signal(self, channel_name, channel_num, dilate=None, mode="mean", subtract_min=False):
        """
        Extracts the mean signal of each cell in the input image(s) based on the masks.
        Will populate the signals and signal_masks attributes.

        Parameters:
            channel_name : str
                Name of the channel to use for the signal calculation.
            channel_num : int
                Position of the channel to use for the signal calculation.
            dilate : int
                The amount of dilation to apply to the masks before calculating the mean signal.
                If negative, erosion is applied instead of dilation.
                Note: dilation/erosion here means that the cell boundaries are expanded or contracted by approximately this many pixels.
            mode: str
                The mode used to calculate the representative signal for each cell
                Default = "mean"; "perc_X" means X-th percentile
            subtract_min: bool
                Whether to subtract the minimum value of the signal in the image from the cell signals. Default is False.

        Returns:
            cells_df : pd.DataFrame
                The cells DataFrame with the calculated representative signal of each cell in the image(s).
            signal_masks : list of np.array
                The masks of the signals in each image, with the same shape as the input images.
        """
        # Perform checks
        if self.samples_df is None:
            raise ValueError("samples_df is empty. Please run read_data() first.")

        n_samples = len(self.samples_df)
        if self.projections is None or len(self.projections) != n_samples:
            raise ValueError("Projections are missing or incomplete. Please run create_projections() for all images first.")
        if self.masks is None or len(self.masks) != n_samples:
            raise ValueError("Segmentations are missing or incomplete. Please run segment_cells() for all images first.")

        has_projection = [p is not None for p in self.projections]
        has_segmentation = [m is not None for m in self.masks]
        self.samples_df["has_projection"] = has_projection
        self.samples_df["has_segmentation"] = has_segmentation
        n_proj = int(sum(has_projection))
        n_seg = int(sum(has_segmentation))
        if n_proj < n_samples or n_seg < n_samples:
            raise ValueError(
                f"calculate_single_cell_signal() requires full preprocessing on all images first. "
                f"Current status: projected {n_proj}/{n_samples}, segmented {n_seg}/{n_samples}."
            )

        if self.cells_df is None:
            raise ValueError("cells_df is empty. Please run previous methods in the pipeline first.")
        
        if channel_num < 1 or channel_num >= len(self.projections[0])+1:
            raise ValueError(f"Channel number {channel_num} for channel {channel_name} is out of bounds for the projections. " +
                             f"Available channels: {len(self.projections[0])}.")
        if channel_num == 0:
            raise ValueError(f"You chose 0 as a channel for {channel_name}. This must be an accident. " +
                             "Note that the input channels are 1-indexed, and with the input 0, you would be using -1 as index.")
        channel_num -= 1 # Decrease by 1 to make it 0-indexed

        # Register the signal mode
        self.signal_mode[channel_name] = mode

        # Make sure dilate is a valid input
        if dilate is None:
            dilate = 0

        # Perform the calculation
        cells_df = self.cells_df.copy()
        all_signals_out = []
        # signal_lists_out = []
        signal_masks_out = []
        for img, mask in zip(self.projections, self.masks):

            # Extract the channel from the image
            img = img[channel_num]
            # Prepare the empty containers
            img_cell_signals_dict = {} #{cell_id: np.mean(img[cell_mask_for_mean]) for cell_id in range(1, mask.max()+1)}
            # img_signal_list = [] #[val for k, val in img_signal_means_dict.items()]
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

                # Calculate the mean, median or percentile signal for the cell
                if mode == "mean":
                    cell_signal = np.mean(img[cell_mask_for_signal])
                elif mode == "median":
                    cell_signal = np.median(img[cell_mask_for_signal])
                elif "perc_" in mode:
                    perc = int(mode.split("_")[-1])
                    cell_signal = np.percentile(img[cell_mask_for_signal], perc)
                else:
                    raise ValueError(f"Mode '{mode}' not recognized for channel {channel_name}. Check docstring for options.")
                # Optionally subtract the minimum value of the signal in the image (e.g. to correct for background)
                if subtract_min:
                    cell_signal -= img[cell_mask_for_signal].min()

                # Assign the signal to the cell ID in the dict and mask
                if np.isnan(cell_signal):
                    cell_signal = 0
                img_cell_signals_dict[cell_id] = cell_signal
                # img_signal_list.append(cell_signal)
                img_signal_mask += cell_signal * cell_mask # NOTE: use un-altered mask here to have no overlaps between cells, even though for the calculation of the signal, the dilated/eroded mask was used

                # Add the signal to the cells_df
                # cells_df.loc[cell_id, channel_name+"_"+mode] = cell_signal
                cells_df.loc[cell_id, channel_name+"_signal"] = cell_signal

                # Also add the log10 of the signal
                cells_df.loc[cell_id, channel_name+"_signal_log10"] = np.log10(cell_signal) if cell_signal > 0 else 0

            all_signals_out.append(img_cell_signals_dict)
            # signal_lists_out.append(img_signal_list)
            signal_masks_out.append(img_signal_mask)

        self.signals[channel_name] = all_signals_out
        # self.signal_lists[channel_name] = signal_lists_out
        self.signal_masks[channel_name] = signal_masks_out

        # Save the cells_df in the object
        self.cells_df = cells_df
        # Add the signal dilation and mode to the channels_df
        if self.channels_df is not None:
            self.channels_df.loc[channel_num, "channel_name"] = channel_name
            self.channels_df.loc[channel_num, "signal_dilate"] = dilate
            self.channels_df.loc[channel_num, "signal_mode"] = mode
            # Move name to first column
            cols = self.channels_df.columns.tolist()
            cols = ["channel_name"] + [col for col in cols if col != "channel_name"]
            self.channels_df = self.channels_df[cols]
        else:
            self.channels_df = pd.DataFrame({channel_num: {"channel_name": channel_name, "signal_dilate": dilate, "signal_mode": mode}}).T
        # Ensure int, since with NaN values, it becomes float
        self.channels_df["signal_dilate"] = self.channels_df["signal_dilate"].astype("Int64")

        return cells_df, signal_masks_out

    def calculate_cell_signals(self, channels, dilate=None, mode="mean", subtract_min=False):
        """Extracts the mean signal of each cell in the input image(s) for multiple channels based on the masks.
        Will populate the signals and signal_masks attributes.

        Parameters:
            channels : dict
                A dictionary with channel names as keys and channel numbers as values, indicating which channels to use for the signal calculation.
            dilate : int or dict
                The amount of dilation to apply to the masks before calculating the mean signal for each channel.
                If negative, erosion is applied instead of dilation.
                If a single int is given, it is applied to all channels.
                If a dict is given, it should have the same keys as channels, with the corresponding dilation values.
                Note: dilation/erosion here means that the cell boundaries are expanded or contracted by approximately this many pixels.
            mode: str or dict
                The mode used to calculate the representative signal for each cell. If a single str is given, it is applied to all channels. If a dict is given, it should have the same keys as channels, with the corresponding mode values.
                Default = "mean"; "perc_X" means X-th percentile
            subtract_min: bool
                Whether to subtract the minimum value of the signal in the image from the cell signals. Default is False.

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
            self.calculate_single_cell_signal(channel_name=name, channel_num=channels[name], dilate=dilate[name], mode=mode[name], subtract_min=subtract_min)

        return self.cells_df, self.signal_masks
    
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
                img_dir = out_folder / f"{Path(self.samples_df['filename'][img_num]).stem}_{signal_name}_mask.png"
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

        # Quick check: binning requires the signal step to be complete for all images
        if signal not in self.signal_masks:
            raise ValueError(
                f"Signal '{signal}' is missing in signal_masks. "
                "Please run calculate_cell_signals() first."
            )
        if len(self.signal_masks[signal]) != len(self.samples_df) or any([m is None for m in self.signal_masks[signal]]):
            raise ValueError(
                f"Signal '{signal}' is incomplete across images. "
                "Please complete calculate_cell_signals() for all images before binning."
            )
        if self.cells_df[column].isna().any():
            raise ValueError(
                f"Signal column '{column}' contains NA values. "
                "Please complete calculate_cell_signals() before binning."
            )

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
            print(f"Thresholding '{bin_name}' at {t}")
            self.cells_df.loc[self.cells_df[column] > t, col_name] = bin_name

        # Add a column with the thresholds and parameters used to the channels_df
        if self.channels_df is not None:
            if "channel_name" in self.channels_df.columns and signal in self.channels_df["channel_name"].values:
                channel_num = self.channels_df[self.channels_df["channel_name"] == signal].index[0]
                thresh_type = f"{'otsu' if use_otsu else 'manual'}"
                # self.channels_df.loc[channel_num, f'threshold_{thresh_type}'] = str(thresh)
                self.channels_df.loc[channel_num, 'bin_use_log'] = str(use_log)
                self.channels_df.loc[channel_num, 'bin_threshold_type'] = thresh_type
                self.channels_df.loc[channel_num, f'bin_threshold(s)'] = str(thresh)
            else:
                print(f"Warning: signal '{signal}' not found in channels_df. Thresholds not saved in channels_df.")
        else:
            print("Warning: channels_df is None. Thresholds not saved in channels_df.")

        # Create masks for the bins
        # print(f"Creating bin masks for signal '{signal}'...")
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
            # print(f"Created bin mask for image {len(bin_masks_out)}")

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
        print(f"Starting binning of signals: {signals}")
        for signal in signals:
            self.bin_single_cell_signal(signal=signal, use_log=use_log[signal], thresh=thresh[signal])
        print(f"Finished binning of signals: {signals}")

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
                img_dir = out_folder / f"{Path(self.samples_df['filename'][img_num]).stem}_{signal_name}_bin_mask.png"
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
            source_signal = signal[:-4] if signal[-4:] == "_bin" else signal
            if signal[-4:] != "_bin":
                signal = signal+"_bin"
                signals[i] = signal
            if signal not in self.cells_df.columns:
                raise ValueError(f"Bin column for signal '{signal}' not found in cells_df. Please run calculate_cell_signals() and bin_cell_signal() first.")
            # Quick check: populations require binning to be complete for all images
            if source_signal not in self.bin_masks or source_signal not in self.bins:
                raise ValueError(
                    f"Binning results for signal '{source_signal}' not found. "
                    "Please run bin_cell_signals() first."
                )
            if len(self.bin_masks[source_signal]) != len(self.samples_df) or any([m is None for m in self.bin_masks[source_signal]]):
                raise ValueError(
                    f"Binning for signal '{source_signal}' is incomplete across images. "
                    "Please complete bin_cell_signals() before creating populations."
                )
            if self.cells_df[signal].isna().any():
                raise ValueError(
                    f"Bin column '{signal}' contains NA values. "
                    "Please complete bin_cell_signals() before creating populations."
                )

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
            img_dir = out_folder / f"{Path(self.samples_df['filename'][i]).stem}_{pop_name}.png"
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
    def count_surrounding_cells(mask, cell_id): #, expected_diameter):
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
        # props = regionprops((mask == cell_id).astype(np.uint8))
        # if not props:
        #     raise ValueError(f"Cell ID {cell_id} not found in mask.")
        # region = props[0]

        # # Cell centroid (y, x)
        # cy, cx = region.centroid
        # # Equivalent circular radius
        # area = np.sum(mask == cell_id)
        # radius = np.sqrt(area / np.pi)
        # extended_radius = radius + expected_diameter/2
        # # print("Cell ID:", cell_id, "Centroid:", (cy, cx), "Radius:", radius, "Extended radius:", extended_radius)

        # y_indices, x_indices = np.indices(mask.shape)
        # dist = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)
        # circle_mask = dist <= extended_radius

        # # Fraction of circle inside the image (edge correction)
        # # Theoretical total circle area:
        # circle_area = np.pi * extended_radius**2
        # # Pixels actually inside image:
        # inside_area = np.sum(circle_mask)
        # inside_fraction = inside_area / circle_area

        # # Get IDs within the circle
        # surrounding_ids = np.unique(mask[circle_mask])
        # surrounding_ids = surrounding_ids[(surrounding_ids != 0) & (surrounding_ids != cell_id)]

        cell = mask == cell_id
        # dilation = expected_diameter // 2
        dilated = morphology.dilation(cell, morphology.disk(1))

        neighbors = np.unique(mask[dilated])
        neighbors = neighbors[(neighbors != 0) & (neighbors != cell_id)]
        return len(neighbors)

        # Edge-corrected estimate
        # inside_fraction = np.count_nonzero(cell) / np.count_nonzero(dilated) if np.count_nonzero(dilated) > 0 else 0
        # corrected_count = len(neighbors) / inside_fraction if inside_fraction > 0 else np.nan
        # return corrected_count

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
    
    def export_data_as_tiff_stacks(self, stack_dims=["projections", "seg_masks", "outlines", "signals", "bins"], folder_name="tiffs", overwrite=False):
        """
        Saves the images, masks, signal masks and bin masks as TIFF stacks.

        Parameters:
            stack_dims : tuple of str
                The dimensions to include in the TIFF stacks. Possible values are "projections", "seg_masks", "outlines", "signals" and "bins". Default is all.
            folder_name : str
                The name of the (sub-)folder to save the TIFF stacks to.
            overwrite : bool
                Whether to overwrite existing files. Default is False.
        """
        stack_keys = {"projections": self.projections,
                      "seg_masks": self.masks,
                      "outlines": self.outlines,
                      "signals": self.signal_masks,
                      "bins": self.bin_masks}
        for dim in stack_dims:
            if dim not in stack_keys:
                raise ValueError(f"Invalid dimension '{dim}' in stack_dims. Valid options are: {list(stack_keys.keys())}.")
            if stack_keys[dim] is None:
                print(f"Warning: {dim} is selected for TIFF stacks but is None. This dimension will be skipped.")
                stack_dims = [d for d in stack_dims if d != dim]

        suffix = "_".join(d[:3] for d in stack_dims)
        
        # Create the folder if it doesn't exist
        out_folder = self.path / folder_name
        out_folder.mkdir(parents=True, exist_ok=True)

        for i in range(len(self.samples_df)):
            if any([stack_keys[dim][i] is None for dim in ("projections", "seg_masks", "outlines")]):
                break  # Stop when projection/segmentation has not been performed for this image, to avoid saving incomplete stacks
            stacks = []
            for dim in stack_dims:                    
                if dim == "projections":
                    stacks.append(self.projections[i])
                    # print(f"Projection shape for image {i}: {self.projections[i].shape}")
                if dim == "seg_masks":
                    mask = self.masks[i].reshape(1, *self.masks[i].shape)  # add an extra axis to make it 3D (1, H, W) for stacking
                    stacks.append(mask.astype(np.uint16))
                    # print(f"Segmentation mask shape for image {i}: {mask.shape}")
                if dim == "outlines":
                    outline = self.outlines[i].reshape(1, *self.outlines[i].shape)  # add an extra axis to make it 3D (1, H, W) for stacking
                    stacks.append(outline.astype(np.uint16))
                    # print(f"Outline shape for image {i}: {outline.shape}")
                if dim == "signals" and self.signal_masks:
                    for signal_name in self.signal_masks.keys():
                        signal = self.signal_masks[signal_name][i].reshape(1, *self.signal_masks[signal_name][i].shape)  # add an extra axis to make it 3D (1, H, W) for stacking
                        stacks.append(signal.astype(np.uint16))
                    # print(f"Signal mask shapes for image {i}: {[s.shape for s in stacks[-len(self.signal_masks):]]}")
                if dim == "bins" and self.bin_masks:
                    for signal_name in self.bin_masks.keys():
                        bin_mask = self.bin_masks[signal_name][i].reshape(1, *self.bin_masks[signal_name][i].shape)  # add an extra axis to make it 3D (1, H, W) for stacking
                        stacks.append(bin_mask.astype(np.uint16))
                    # print(f"Bin mask shapes for image {i}: {[s.shape for s in stacks[-len(self.bin_masks):]]}")

            if not stacks:
                print(f"Found no valid data for the chosen input. Please make sure stack_dims is a choice of the following:")
                print(f"{list(stack_keys.keys())}")
                return

            # Stack along a new axis (the first axis)
            tiff_stack = np.concatenate(stacks, axis=0)
            # print(f"TIFF stack shape for image {i}: {tiff_stack.shape}")

            # Save the TIFF stack
            img_dir = out_folder / f"{Path(self.samples_df['filename'][i]).stem}_{suffix}_stack.tiff"
            if img_dir.exists() and not overwrite:
                print(f"File {img_dir} already exists. Saving this file was skipped.")
            else:
                imwrite(img_dir, tiff_stack)
        
        print(i+1, "TIFF stacks saved.")

    def add_region_props(self, properties=["area", "perimeter", "eccentricity", "orientation", "centroid"]):
        """
        Adds region properties to the cells_df DataFrame based on the segmentation masks.

        Parameters:
            properties : tuple of str
                The region properties to calculate. Default is ("area", "perimeter", "eccentricity", "orientation", "centroid", "bbox").
                For more information on available properties, see skimage.measure.regionprops_table.
        """
        # Checks for the properties argument (needs to be a list)
        if not isinstance(properties, (list, tuple)):
            raise ValueError("properties must be a list or tuple of strings corresponding to skimage.measure.regionprops_table properties.")
        elif isinstance(properties, tuple):
            properties = list(properties)

        temp_df = pd.DataFrame()  # Temporary DataFrame to hold properties for all images
        for i, mask in enumerate(self.masks):
            if mask is None:
                break
            props = regionprops_table(
                mask,
                intensity_image=np.moveaxis(self.projections[i],0,-1),
                properties=["label"] + properties
            )
            props_df = pd.DataFrame(props)
            # 
            temp_df = pd.concat([temp_df, props_df], ignore_index=True, axis=0)

        # Join properties by cell ID while preserving the existing cells_df index.
        # Using merge(..., right_on="label") resets the index and can shift row alignment.
        props_by_cell_id = temp_df.set_index("label")
        self.cells_df.drop(columns=props_by_cell_id.columns, inplace=True, errors='ignore')
        self.cells_df = self.cells_df.join(props_by_cell_id, how="left")
