# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: Data preprocessing classes which extract raw data generate RGB image and perform necessary operation to prepare the data for training.
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: The function are built based on the general training loop structure found in pytorch tutorials.


import os
import torch
import rasterio
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight # to calculate the weight to balance the dataset


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
        self.datasettype = datasettype # setting the type of dataset ["train","val","test"]
        self.device = device
        self.input_imagetype = input_image # ["rgb"]
        self.image_index = [3, 2, 1] if self.input_imagetype == "rgb" else [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.output_channels = output_channels
        self.filter_label=filter_label
        self.data_df, self.categories, self.labels,self.class_weight= self.load_csv(csv_filepath, self.datasettype,self.filter_label)
        self.root_dir = root_dir
        self.transform = transform
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        
        self.mask_categories = {
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
            class_weight=1
        else:
            class_weight = compute_class_weight(class_weight='balanced',
                                    classes=label,
                                    y=filtered_df['label'])
        return filtered_df, categories, label,class_weight
    

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
        """
        Retrieve a random image and its corresponding class from the dataset.
        Args:
            idx (int): The index of the image to retrieve.
        Returns:
            torch.Tensor: The image tensor.
        """

        #random_idx = np.random.randint(len(self.data_df))
        row = self.data_df.iloc[idx]
        filename = row['file']
        img_filepath = os.path.join(self.root_dir, "tiles/s2", filename)
        mask_filepath = os.path.join(self.root_dir, "tiles/s2_scl", filename)
        rgbimage=self.load_Sentinel(img_filepath,self.image_index)
        mask,maskfraction=self.load_Sentinel_L2A(mask_filepath)
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
        
    
    def __len__(self):
        return len(self.data_df)



class LCSDataset(Dataset):
    """
    A dataset class for retrieving items from the LCS dataset.
    Args:
        csv_filepath (str): The file path of the CSV file containing the dataset.
        root_dir (str): The root directory of the dataset.
        datasettype (str, optional): The type of the dataset. Defaults to "train".
        input_image (str, optional): The type of input image. Defaults to "rgb".
        modeltype (str, optional): The type of the model. Defaults to "multilabel".
        channel_name (str, optional): The name of the channel. Defaults to "corine2".
        device (str, optional): The device to use (GPU or CPU). Defaults to "cpu".
        filter_label (str, optional): The label to filter the dataset. Defaults to None.
        transform (callable, optional): A function/transform to apply to the dataset. Defaults to None.
        min_percentile (int, optional): The minimum percentile for image normalization. Defaults to 1.
        max_percentile (int, optional): The maximum percentile for image normalization. Defaults to 99.
    """
    def __init__(self,csv_filepath, root_dir,datasettype="train",input_image="rgb",modeltype="multilabel",channel_name="corine2",device="cpu",filter_label=None,transform=None, min_percentile=1, max_percentile=99) -> None:
        
        self.datasettype = datasettype
        self.device = device
        self.input_imagetype = input_image
        self.filter_label=filter_label
        self.modeltype=modeltype
        self.image_index = [3, 2, 1] if self.input_imagetype == "rgb" else [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.data_df, self.categories, self.labels,self.class_weight= self.load_csv(csv_filepath, self.datasettype,self.filter_label)
        self.root_dir = root_dir
        self.channels_namelist=['CORINE', 'MODIS_1', 'CGLS', 'GlobCover']
        self.channel_name=channel_name
        self.channel_index = next((i for i, x in enumerate(self.channels_namelist) if x.lower() == channel_name.lower()), -1)+1
        if self.channel_name=="corine":
            # corine dataset has 44 classes
            self.channellabels = [111, 112, 121, 122, 123, 124, 131, 132, 133, 141, 142, 211, 212, 213, 221, 222, 223, 231, 241, 242, 243, 244, 
                                  311, 312, 313, 321, 322, 323, 324, 331, 332, 333, 334, 335, 411, 412, 421, 422, 423,511, 512, 521, 522, 523]
            self.output_channels = len(self.channellabels)
            self.transform = transform
        elif self.channel_name=='corine2':
            # we resampled them into 10 classes based on the avaliabilty of the data
            self.corine_to_category = {
           
            111: "Artificial Surfaces", 112: "Artificial Surfaces", 121: "Artificial Surfaces",
            122: "Artificial Surfaces", 123: "Artificial Surfaces", 124: "Artificial Surfaces",
            131: "Artificial Surfaces", 132: "Artificial Surfaces", 133: "Artificial Surfaces",
            141: "Artificial Surfaces", 142: "Artificial Surfaces",
             # Agricultural areas
            
            211:"Arable land",212:"Arable land",213:"Arable land",
            221:"Permanent crops",222:"Permanent crops",223:"Permanent crops",
            231: "Pastures", 241: "heterogeneous agricultural areas", 242: "heterogeneous agricultural areas",
            # Forest
            311: "Forest", 312: "Forest", 313: "Forest",
            321: "scrub ", 322: "scrub ", 323: "scrub ",324:"scrub ",
            331:"open spaces",332:"open spaces",333:"open spaces",
            # Wetlands
            411: "Wetlands", 412: "Wetlands", 421: "Wetlands", 422: "Wetlands", 423: "Wetlands",
            # Water bodies
            511: "Waterbodies", 512: "Waterbodies", 521: "Waterbodies", 522: "Waterbodies", 523: "Waterbodies"
            # Add other mappings as needed
            }
            self.output_channels = 10
            self.transform = transform
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile 
        pass


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
            class_weight=1
        else:
            class_weight = compute_class_weight(class_weight='balanced',
                                    classes=label,
                                    y=filtered_df['label'])
        return filtered_df, categories, label,class_weight
    

    def load_Sentinel(self,filepath,channel_index):
        """

        Returns:
            np.array: _description_
        """  
        with rasterio.open(filepath) as src:
            rgb_image = src.read(channel_index)
        return np.array(rgb_image)
    
    def get_labels(self, filepath):
        with rasterio.open(filepath) as masktiff:
            maskimage = np.array(masktiff.read(self.image_index))
            unique_labels = np.int16(np.unique(maskimage))
        if self.channel_name=="corine2":
            # Map the unique labels to the 4 broader categories
            categories_present = set()
            for label in unique_labels:
                category = self.corine_to_category.get(label)
                if category:
                    categories_present.add(category)

            # Generate a binary list for the 4 categories
            broad_categories =  ["Artificial Surfaces","Arable land","Permanent crops","Pastures","heterogeneous agricultural areas", "Forest","scrub ","open spaces", "Wetlands", "Waterbodies"]
            
            multilabels = [1 if category in categories_present else 0 for category in broad_categories]

            return multilabels
        else:
            multilabels = [1 if label in unique_labels else 0 for label in self.channellabels]
            return multilabels

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

    def __getitem__(self, idx):
        """
        Retrieve a random image and its corresponding class and label or label from the dataset.
        Args:
            idx (int): The index of the image to retrieve.
        Returns:
            torch.Tensor: The image tensor.
        """

        #random_idx = np.random.randint(len(self.data_df))
        row = self.data_df.iloc[idx]
        filename = row['file']
        img_filepath = os.path.join(self.root_dir, "tiles/s2", filename)
        mask_filepath = os.path.join(self.root_dir, "tiles/s2_scl", filename)
        lcs_filepath= os.path.join(self.root_dir, "tiles/lcs", filename)
        rgbimage=self.load_Sentinel(img_filepath,self.image_index)
        mask,maskfraction=self.load_Sentinel_L2A(mask_filepath)
        labels=self.get_labels(lcs_filepath) # labels are calculated
        mask=None
        sentinel_img=self.normalize_image(rgbimage,mask,self.min_percentile,self.max_percentile)
        if self.transform is not None:
            sentinel_img=self.transform(sentinel_img)
        # Return the image and label
        sentinel_img = torch.Tensor(sentinel_img).to(self.device)  # Send image to GPU
        if self.modeltype=="multiclassnlabel":
            return sentinel_img,torch.tensor(labels,dtype=float).to(self.device), torch.tensor(row['label']).to(self.device)
        else:
            return sentinel_img,torch.tensor(labels,dtype=float).to(self.device)
       
    
    def __len__(self):
        return len(self.data_df)



if __name__=="__main__":
    # Initialize the custom dataset
    csv_filepath = 'D:/master-thesis/Dataset/anthroprotect/infos.csv'
    root_dir = 'D:/master-thesis/Dataset/anthroprotect'  # Update to your root directory
    batch_size = 8  # Choose an appropriate batch size
    custom_dataset = SentinelDataset(csv_filepath, root_dir)
    print(custom_dataset.class_weight)
    subsetsize=0.25
    subsample_size = int(subsetsize * len(custom_dataset))
    indices = np.random.choice(len(custom_dataset), subsample_size, replace=False)
    subsample_dataset = torch.utils.data.Subset(custom_dataset, indices)
    loader = torch.utils.data.DataLoader(subsample_dataset, batch_size=8,shuffle=True)
    print(len(loader))
    