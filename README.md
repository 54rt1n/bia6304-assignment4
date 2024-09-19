# Assignment 4: Topic Modeling of Wikipedia Summary Dataset

## Overview

This project implements a topic modeling system using Latent Dirichlet Allocation (LDA) on a subset of the Wikipedia summary dataset. It's designed to demonstrate proficiency in natural language processing, topic modeling, and data processing at scale. The system allows users to subsample a large dataset, perform topic modeling, and visualize the results.

## Key Features

- Uniform subsampling of large datasets using Dask
- Topic modeling using Latent Dirichlet Allocation (LDA)
- Efficient data processing with Dask DataFrames
- Visualization of topic similarities using NetworkX
- Jupyter notebook for interactive analysis
- Comprehensive configuration management

## Technology Stack

- Python 3.9+
- Poetry for dependency management
- Dask for large-scale data processing
- Gensim for topic modeling
- NetworkX for graph creation and analysis
- Matplotlib for visualization
- Jupyter for interactive development

## Project Structure

```
assignment4/
├── assignment/
│   ├── __init__.py
│   ├── assignment4.py
│   ├── config.py
│   └── data/
│       ├── __init__.py
│       ├── io.py
│       ├── lda.py
│       └── sampling.py
├── main.ipynb
├── pyproject.toml
└── README.md
```

## Setup

1. Ensure you have Python 3.9 or higher installed.
2. Install Poetry if you haven't already:
   ```
   pip install poetry
   ```
3. Clone the repository:
   ```
   git clone https://github.com/54rt1n/bia6304-assignment4.git
   cd bia6304-assignment4
   ```
4. Install dependencies:
   ```
   poetry install
   ```
5. Download the Wikipedia summary dataset:
   - Reduced dataset: [wikipedia-summary-128k.tsv](https://huggingface.co/datasets/mbukowski/wikipedia-summary-dataset-128k/resolve/main/wikipedia-summary-128k.tsv)
   - Full dataset (1.78GB): [wikipedia-summary.parquet](https://huggingface.co/datasets/mbukowski/wikipedia-summary-dataset/resolve/main/wikipedia-summary.parquet)

## Running the Analysis

The main analysis is performed in the Jupyter notebook `main.ipynb`. To run it:

1. Start Jupyter Lab or Jupyter Notebook:
   If you don't already have Jupyter installed, you can install it using conda, pip, or your preferred method.

2. Open `main.ipynb` and run the cells sequentially.

## Assignment Components

### Subsampler Pipeline

The `subsampler_pipeline` function in `assignment4.py` loads the Wikipedia dataset, performs uniform subsampling, and saves the processed data. You should use the io and sampling modules from the assignment4/assignment/data directory for this task.

### LDA Topic Model

The `lda_topic_model` function in `assignment4.py` preprocesses the text data, creates a gensim dictionary and corpus, and trains an LDA model. You will implement the LDA model training using the gensim library.

## Core Components

### Config

The `Config` class in `config.py` manages configuration settings for the project, including file paths, sample size, and LDA parameters.

### Data Processing

- `io.py`: Functions for loading and saving data using Dask DataFrames.
- `sampling.py`: Implementation of uniform subsampling on Dask DataFrames.
- `lda.py`: Utility functions for working with LDA models.

## Visualization

The notebook includes visualizations of the topic similarity graph using different NetworkX layouts:
- Fruchterman-Reingold Layout
- Spectral Layout
- Kamada-Kawai Layout
- Shell Layout

## Configuration

When you instantiate a `Config` instance, you can customize the behavior of the system, including:
- Changing input and output file paths
- Target sample size
- Random seed
- Number of topics for LDA
- Number of LDA passes

## Troubleshooting

If you encounter any issues:

1. Ensure all dependencies are correctly installed (`poetry install`).
2. Check that you're using a compatible Python version (3.9+).
3. Verify that the input data is in the correct format.
4. If you're having memory issues, try reducing the `target_sample_size` in the configuration.

## License

MIT

## Acknowledgments

- Dask for enabling large-scale data processing
- Gensim for providing the LDA implementation
- NetworkX and Matplotlib for graph visualization
- Claude Sonnet 3.5 and GPT-4-o1-preview for code assistance and review
