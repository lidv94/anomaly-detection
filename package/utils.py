import time 
from datetime import datetime
import argparse
import pandas as pd
import numpy as np
import yaml
import pytz
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
<<<<<<< HEAD
# import boto3
=======
import boto3
>>>>>>> adf012660f1047d629039d8e55346312dbb0d79e


class DotDict:
    """a utility class helps converting python dictionary in to class with dot attribute
    
    Args:
        input_dict (dict) : the input dictionary
        
    Examples:
        >>> dotdict = DotDict(input_dict={'a', 1, 'b', 2})
        >>> dotdict.a
        1
    """
    def __init__(self, input_dict:dict):
        self.__dict__.update(input_dict)
        
def timer(func):
    """a wrapper function for timeing the execution time. It can be used as python decorator
    
    Examples:
        >>> timer_fn = timer(lambda x: x+1)
        >>> timer_fn(1)
        function: <lambda> is starting...
        function: <lambda> successfully executed 0.002s
        2
    """
    def wrap(*args, **kwargs):
        start = time.time()
        print(f"function: {func.__name__} is starting...", ) 
        result = func(*args, **kwargs)
        end = time.time()
        print(f"function: {func.__name__} successfully executed at {end-start}s", ) 
        return result 
    return wrap

@timer
def get_config(config_path:str) -> DotDict:
    """convert a ymal file into DoTDict, a dot notation
    Args:
        config_path (str) : a path of configuration file
    Returns:
        DotDict : a dot notation
    """
    with open(config_path, 'r') as f:
        conf = DotDict(input_dict=yaml.safe_load(f))
    return conf

@timer
def get_parameters(batch_size:int=1000, 
                   start:int=0, 
                   end:int=2000,
                   n_jobs:int=1
                  ):
    """this function will be used in final script for getting new input arguments
    
    Args:
        batch_size (int) : the size of data that will be used in each batch
        start (int) : the start index using in df.iloc[start_index:end_index,:] for filtering purpose
        end (int) : the end index using in df.iloc[start_index:end_index,:] for filtering purpose
        n_jobs (int) : the number of cpu cores will be used in this program
    Returns:
        argparse.Namespace : an object containing all input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--start', type=int, default=start)
    parser.add_argument('--end', type=int, default=end)
    parser.add_argument('--n_jobs', type=int, default=n_jobs)
    args, _ = parser.parse_known_args()
    return args

@timer
def save_file(df:pd.core.frame.DataFrame, path:str, method:str='csv')->None:
    """a function for saving pd.core.frame.DataFrame
    
    Args:
        df (pd.core.frame.DataFrame) : input dataframe
        path (str) : the save location
        method (str) : method of saving. now it has only method='csv'
    """
    if method=='csv':
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Method {method} is not implemented. Choose csv instead.")
    print(f"saved successfully at {path}")

@timer
def profile_data(data:pd.core.frame.DataFrame, 
                 # path=None, 
                 sample=5) -> None:
    """profile data
    Args:
        data () : a dataset
        path (str) : a path for saving file
        sample (int) : a sample of each feature
    """
    df = pd.DataFrame(data.isna().sum(), columns=['missing'])
    df['missing%'] = data.isna().sum()/data.shape[0]
    df['nunique'] = data.nunique()
    df['sample'] = [data[col].unique()[:sample] for col in data.columns]
    df = df.reset_index(names='features')
    df.insert(1, 'dtype', data.dtypes.values)
    # df.insert(1, 'description', '')
    # df.insert(1, 'feature_group', '')
    # df.insert(1, 'type', '')
    # save_file(df=df, path=path, method='csv')
    return df
    
@timer
def load_data(path:str) -> pd.core.frame.DataFrame:
    """load a csv dataset
    Args:
        path (str) : a path for csv file
    Returns:
        pd.core.frame.DataFrame : a dataset
    """
    print(f"loading data from: {path}")
    data = pd.read_csv(path,)
    data.columns = data.columns.str.lower()
    print(f"Succesfully loaded data from: {path}")
    return data

@timer
def merge_data(left:pd.core.frame.DataFrame, right:pd.core.frame.DataFrame, how:str, left_on:list, right_on:list)->pd.core.frame.DataFrame:
    """merge between two datasets and print out data shape before & after merging
    Args:
        left (pd.core.frame.DataFrame) : a base dataset
        right (pd.core.frame.DataFrame) : a dataset to join
        how (str) : joining methods including left, right, inner, outer
        left_on (list) : a list of reference keys from left dataset
        right_on (list) : a list of reference keys from right dataset
    Returns:
        pd.core.frame.DataFrame : a joined dataset
    """
    print(f"left shape: {left.shape}")
    print(f"right shape: {right.shape}")
    merged = left.merge(right=right, how='left', left_on=left_on, right_on=right_on)
    print(f"merged dataset shape: {merged.shape}")
    return merged

@timer
def convert_datetime(data:pd.core.frame.DataFrame, columns:list, format:str)->pd.core.frame.DataFrame:
    """convert datatype from string into datetime
    Args:
        data (pd.core.frame.DataFrame) : a dataset
        columns (list) : target columns that will be converted
        format (str) : a string format of the target columns
    Returns:
        pd.core.frame.DataFrame : a dataset with datetime columns
    """
    proxy = data.copy()
    for col in columns:
        proxy[col] = pd.to_datetime(proxy[col], format=format)
    return proxy

def get_datetime():
    return datetime.now()

# alice's added
def get_format_time(timezone="Asia/Bangkok"):
    unix_time = time.time()
    utc_time = datetime.utcfromtimestamp(unix_time)
    target_timezone = pytz.timezone(timezone)
    local_time = utc_time.replace(tzinfo=pytz.utc).astimezone(target_timezone)
    formatted_time = local_time.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time

# alice's added
def fig_to_array(fig):
    """
    Convert a Matplotlib figure to a NumPy array.

    Args:
        fig (matplotlib.figure.Figure): A Matplotlib figure to convert.

    Returns:
        np.ndarray: An image array.
    """
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    im = Image.open(buf)
    image_array = np.array(im)
    print("Image shape:", image_array.shape)
    # print("First 3x3 pixels and RGB values:")
    # print(image_array[:3, :3, :])
    return image_array

# alice's added
def load_image(path):
    """Load an image file from a local path and display it.
    Args:
        path (str): Path to the image file.
    Returns:
        np.ndarray: Image array.
    """
    im = Image.open(path)
    image_array = np.array(im)
    plt.imshow(image_array)
    plt.axis('off')  # Hide axes
    plt.show()
    return image_array

# alice's added
def save_image(img_array, path):
    """Write an image array to a local path.
    Args:
        img_array (np.ndarray): Image array.
        path (str): Path to save the image file.
    Returns:
        None
    """
    im = Image.fromarray(img_array)
    if im.mode == 'RGBA':
        im = im.convert('RGB')  # Convert to RGB mode
    im.save(path, format='jpeg')


# alice's added
def load_image_from_s3(bucket, key, region_name='ap-southeast-1'):
    """Load an image file from an S3 bucket and display it.

    Args:
        bucket (str): The name of the S3 bucket.
        key (str): The path to the image file in the S3 bucket.
        region_name (str, optional): The AWS region name. Defaults to 'ap-southeast-1'.

    Returns:
        np.ndarray: Image array.
    """
    s3 = boto3.resource('s3', region_name=region_name)
    bucket = s3.Bucket(bucket)
    object = bucket.Object(key)
    response = object.get()
    file_stream = response['Body']
    im = Image.open(file_stream)
    image_array = np.array(im)
    plt.imshow(image_array)
    plt.axis('off')  # Hide axes
    plt.show()
    return image_array

# alice's added
def save_image_to_s3(img_array, bucket, key, region_name='ap-southeast-1'):
    """Write an image array to an S3 bucket.

    Args:
        img_array (np.ndarray): Image array.
        bucket (str): The name of the S3 bucket.
        key (str): The path to save the image file in the S3 bucket.
        region_name (str, optional): The AWS region name. Defaults to 'ap-southeast-1'.

    Returns:
        None
    """
    s3 = boto3.resource('s3', region_name=region_name)
    bucket = s3.Bucket(bucket)
    object = bucket.Object(key)
    file_stream = BytesIO()
    im = Image.fromarray(img_array)
    im.save(file_stream, format='jpeg')
    object.put(Body=file_stream.getvalue())
    
@timer
def drop_null(before_df:pd.core.frame.DataFrame,
              key_notna:str):
    print('shape before drop null :', before_df.shape)
    after_df = before_df[before_df[key_notna].notna()]
    print('shape after drop null ::',after_df.shape)
    print('#missing transaction',before_df.shape[0]-after_df.shape[0],'/','%missing transaction',round(((before_df.shape[0]-after_df.shape[0])/before_df.shape[0])*100,2),'%')
    return after_df
