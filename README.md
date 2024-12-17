# preference_location
This repository provides the parameter generation and instances used in the study titled:  
**TITULO**  

The full text of the study is available on Arxiv:  
[Link to Arxiv](URL_GOES_HERE)  

## Data Files  
The `BlobData` folder contains the dataset of centers, radii, and norms for the values of `n` used in the study (10, 20, 50, 100, 500).  
The `BlobCenters` folder includes the reference centers used for the distance-based preference function.  

## Python Scripts  
Two Python scripts are provided:  

1. **`github_gen_parameters_library.py`**  
   This script contains the functions to generate parameters for the preference functions. Due to the randomness inherent in the parameter generation process, the script ensures reproducibility by generating the same parameters each time it is executed, given the same inputs (e.g., functions, sizes, seeds, etc.).  

2. **`github_example_generation.py`**  
   This script demonstrates how to use the parameter generation library to produce instances for a given combination of data and preference function configurations.  

## Contact  
For more information, please contact:  
**rgazquez[at]ugr.es**  

