# assignment/data/sampling.py

import dask.dataframe as dd
import logging

logger = logging.getLogger(__name__)

def uniform_subsample(ddf: dd.DataFrame, size: int, random_seed: int = 42) -> dd.DataFrame:
    """
    Perform uniform subsampling on a Dask DataFrame.

    Args:
        ddf (dd.DataFrame): Input Dask DataFrame to subsample.
        size (int): Size of the subsample.
        random_seed (int): Seed for random number generator for reproducibility.

    Returns:
        dd.DataFrame: Subsampled Dask DataFrame.
    """

    # With dask you have to compute the current state of the dataframe to retrieve values
    data_len = ddf.shape[0].compute()
    max_size = min(size, data_len)

    # Dask requires that the fractional value is used
    frac = max_size / data_len
    
    try:
        subsampled_ddf = ddf.sample(frac=frac, random_state=random_seed)
        new_len = subsampled_ddf.shape[0].compute()
        logger.info(f"Subsampling completed. New shape: {new_len}")
        return subsampled_ddf
    except Exception as e:
        logger.error(f"Error during subsampling: {str(e)}", exc_info=1)
        return None
