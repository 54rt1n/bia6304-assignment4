# assignment/data/io.py

import dask.dataframe as dd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def load_wikipedia_dataset(file_path: str, delimiter: str = "\t", npartitions: int = None) -> Optional[dd.DataFrame]:
    """
    Load the Wikipedia Summary Dataset using Dask.

    Args:
        file_path (str): Path to the dataset file (parquet or csv/tsv format expected).
        delimiter (str, optional): Delimiter used in the dataset file. Defaults to tsv "\t".
        npartitions (int, optional): Number of partitions for Dask DataFrame. Defaults to None.

    Returns:
        dd.DataFrame: Dask DataFrame containing the Wikipedia summaries.
    """
    try:

        if file_path.endswith(".parquet"):
            ddf = dd.read_parquet(file_path)
        else:
            ddf = dd.read_csv(file_path, delimiter=delimiter, engine='python',
                              header=None, skipinitialspace=True, quoting=3)
        
            ddf.columns = ['topic', 'summary']

        if npartitions:
            ddf = ddf.repartition(npartitions=npartitions)
        
        logger.info(f"Dataset loaded successfully. {ddf.shape[0].compute()} entries.")
        return ddf
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}", exc_info=1)
        return None

def save_processed_data(ddf: dd.DataFrame, output_path: str) -> None:
    """
    Save the processed Dask DataFrame to disk.

    Args:
        ddf (dd.DataFrame): Processed Dask DataFrame to save.
        output_path (str): Path to save the processed data.

    Returns:
        None
    """
    try:
        if output_path.endswith(".parquet"):
            ddf.to_parquet(output_path, engine='pyarrow')
        elif output_path.endswith(".csv"):
            ddf.to_csv(output_path, index=False, single_file=True)
        elif output_path.endswith(".tsv"):
            ddf.to_csv(output_path, sep='\t', index=False, single_file=True)
        else:
            raise ValueError("Unsupported format. Use 'csv', `tsv`, or 'parquet'.")
        
        logger.info(f"Processed data saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Error saving processed data: {str(e)}", exc_info=1)

def load_processed_data(file_path: str) -> Optional[dd.DataFrame]:
    """
    Load the processed data from disk.

    Args:
        file_path (str): Path to the processed data file.

    Returns:
        Optional[dd.DataFrame]: Loaded data as a Dask DataFrame. None, if no file was found,
            or unsupported format.
    """
    try:
        if file_path.endswith(".parquet"):
            data = dd.read_parquet(file_path)
        elif file_path.endswith(".csv"):
            data = dd.read_csv(file_path, quoting=3)
        elif file_path.endswith(".tsv"):
            data = dd.read_csv(file_path, delimiter='\t')
        else:
            raise ValueError("Unsupported format. Use 'csv', `tsv`, or 'parquet'.")
        
        logger.info(f"Processed data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading processed data: {str(e)}", exc_info=1)
        return None
