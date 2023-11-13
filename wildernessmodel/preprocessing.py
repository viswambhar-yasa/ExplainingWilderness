import os
import torch
import rasterio
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SentinelDataset(Dataset):
    def __init__(self, csv_filepath, root_dir, input_image="rgb", output_channels=2,filter_label=None, datasettype="train", device="cpu", transform=None, min_percentile=1, max_percentile=99):
        """
        Initializes the SentinelDataset class object with the provided parameters.

        Args:
            csv_filepath (string): The file path of the CSV file containing the dataset information.
            root_dir (string): The root directory of the dataset.
            input_image (string, optional): The type of input image, either "rgb" or "other". Defaults to "rgb".
            output_channels (int, optional): The number of output channels. Defaults to 1.
            batch_size (int, optional): The batch size for the DataLoader. Defaults to 16.
            datasettype (string, optional): The type of dataset, either "train" or "test". Defaults to "train".
            device (string, optional): The device to be used for computation, either "cpu" or "gpu". Defaults to "cpu".
            transform (callable, optional): Optional data transformation to be applied to the input image. Defaults to None.
            min_percentile (int, optional): The minimum percentile for image normalization. Defaults to 1.
            max_percentile (int, optional): The maximum percentile for image normalization. Defaults to 99.

        Returns:
            None
        """
        self.datasettype = datasettype
        self.device = device
        self.input_imagetype = input_image
        self.image_index = [3, 2, 1] if self.input_imagetype == "rgb" else [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.output_channels = output_channels
        self.filter_label=filter_label
        self.data_df, self.categories, self.labels = self.load_csv(csv_filepath, self.datasettype,self.filter_label)
        self.root_dir = root_dir
        self.transform = transform
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.mask_categories = { # we need to use the CORINE masks, which has nearly 44 classes, it has masks such as wetlands and bare lands which we need to interpret the results
            0: "NO_DATA",
            1: "SATURATED_OR_DEFECTIVE",
            2: "CAST_SHADOWS",
            3: "CLOUD_SHADOWS",
            4: "VEGETATION",
            5: "NOT_VEGETATED",
            6: "WATER",
            7: "UNCLASSIFIED",
            8: "CLOUD_MEDIUM_PROBABILITY",
            9: "CLOUD_HIGH_PROBABILITY",
            10: "THIN_CIRRUS",
            11: "SNOW or ICE"
        }
        
    
    
    def load_csv(self,csvfilepath, datasettype, label_filter=None):
        """

        Args:
            csvfilepath (str): _description_
            datasettype (str): _description_
            label_filter (list, optional): _description_. Defaults to None.

        Returns:
            filtered_df: dataframe
            categories :dataframe
            label:list
        """
        df = pd.read_csv(csvfilepath)
        filtered_df = df[df['datasplit'] == datasettype].copy()  # Use .copy() to create a copy of the DataFrame

        # Extract unique labels and categories before modifying the DataFrame
        label = filtered_df["label"].unique()
        categories = filtered_df["category"].unique()

        # Add category labels column using .loc[]
        filtered_df.loc[:, 'category_labels'] = pd.factorize(filtered_df['category'])[0]

        # Apply label filter if provided
        if label_filter is not None:
            filtered_df = filtered_df[filtered_df['label'].isin(label_filter)].copy()  # Make a copy after the filter
        return filtered_df, categories, label
    

    def load_Sentinel(self,filepath,channel_index):
        """

        Returns:
            np.array: _description_
        """  
        with rasterio.open(filepath) as src:
            rgb_image = src.read(channel_index)
        return np.array(rgb_image)
    

    def load_Sentinel_L2A(self,maskfilepath):
        """

        Args:
            maskfilepath (_type_): _description_

        Returns:
            mask: np.array
            mask_fraction_percentage: float
        """
        L2A_mask=np.array(Image.open(maskfilepath))
        total_pixels=L2A_mask.size
        L2A_mask[(L2A_mask==4)|(L2A_mask==5)|(L2A_mask==6)|(L2A_mask==7)]=12
        L2A_mask[L2A_mask<11]=0
        L2A_mask[L2A_mask==12]=1
        mask_fraction_percentage=(np.sum(L2A_mask==0)/total_pixels)*100
        return L2A_mask,mask_fraction_percentage

    def normalize_image(self,rgbimage,mask=None,min_percentile=1,max_percentile=99):
        """
        Args:
            rgbimage (np.array): rgb image from the sentinel image
            mask (np.array, optional): mask . Defaults to None.
            min_percentile (int, optional): . Defaults to 1.
            max_percentile (int, optional): . Defaults to 99.

        Returns:
            normalized_img: np.array
        """
        if mask is not None:
            valid_pixels = rgbimage[:,mask != 0]
        else:
            valid_pixels = rgbimage
        min_val = np.percentile(valid_pixels, min_percentile)
        max_val = np.percentile(valid_pixels, max_percentile)
        normalized_img = (rgbimage - min_val) / (max_val - min_val)
        normalized_img = np.clip(normalized_img, 0, 1)
        return normalized_img
    
    def channel_normalize_image(self,rgbimage, mask=None, min_percentile=1, max_percentile=99):
        """

        Args:
            rgbimage (np.array): rgb image from the sentinel image
            mask (np.array, optional): . Defaults to None.
            min_percentile (int, optional): _description_. Defaults to 1.
            max_percentile (int, optional): _description_. Defaults to 99.

        Returns:
            _type_: _description_
        """
        normalized_img = np.zeros_like(rgbimage, dtype=np.float32)  # Initialize an array to store normalized image
        for i in range(rgbimage.shape[2]):  # Loop through each channel
            channel_data = rgbimage[:,:,i]  # Extract the current channel data
            if mask is not None:
                valid_pixels = channel_data[mask != 0]
            else:
                valid_pixels = channel_data
            min_val = np.percentile(valid_pixels, min_percentile)
            max_val = np.percentile(valid_pixels, max_percentile)
            normalized_channel = (channel_data - min_val) / (max_val - min_val)
            normalized_channel = np.clip(normalized_channel, 0, 1)
            normalized_img[:,:,i] = normalized_channel  # Assign the normalized channel to the corresponding position

        return normalized_img
    
    def __getitem__(self, idx):
        random_idx = np.random.randint(len(self.data_df))
        row = self.data_df.iloc[random_idx]
        filename = row['file']
        img_filepath = os.path.join(self.root_dir, "tiles/s2", filename)
        mask_filepath = os.path.join(self.root_dir, "tiles/s2_scl", filename)
        rgbimage=self.load_Sentinel(img_filepath,self.image_index)
        mask,maskfraction=self.load_Sentinel_L2A(mask_filepath)
        if maskfraction <= 5.0:
            mask=None
            sentinel_img=self.normalize_image(rgbimage,mask,self.min_percentile,self.max_percentile)
            if self.transform is not None:
                sentinel_img=self.transform(sentinel_img)
            # Return the image and label
            sentinel_img = torch.Tensor(sentinel_img).to(self.device)  # Send image to GPU
            if self.output_channels==2:
                label = torch.tensor(row['label']).to(self.device)  # Send label to GPU
            else:
                label = torch.tensor(row['category_labels']).to(self.device) 
            return sentinel_img, label
        else:
            return self.__getitem__(idx)
    
    def __len__(self):
        return len(self.data_df)

    
     

if __name__=="__main__":
    # Initialize the custom dataset
    csv_filepath = 'D:/master-thesis/Dataset/anthroprotect/infos.csv'
    root_dir = 'D:/master-thesis/Dataset/anthroprotect'  # Update to your root directory
    batch_size = 8  # Choose an appropriate batch size
    custom_dataset = SentinelDataset(csv_filepath, root_dir)
    # Create a DataLoader for batching
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
    # Example usage within a DataLoader loop
    for batch in dataloader:
        images, labels = batch
        print(images.shape,labels)
        break
        
