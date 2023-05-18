# grit-lab-public-soil-moisture-retrieval-repo
## Public Release v1.0.1


Welcome to our repository. Here, you will find the Python scripts and the lab and UAS dataset that were used in the research article "Comparison of Soil Moisture Content Retrieval Models Utilizing Hyperspectral Goniometer Data and Hyperspectral Imagery from an Unmanned Aerial System" by Nayma Binte Nur and Charles M. Bachmann.

For the most recent updates, kindly visit our GitHub repository at [grit-lab-public-soil-moisture-retrieval-repo](https://github.com/grit-lab/grit-lab-public-soil-moisture-retrieval-repo).

## Repository Structure

The repository comprises Python implementations of three soil moisture content retrieval models:

1. MARMIT
2. Original (SWAP)-Hapke
3. Modified (SWAP)-Hapke

These models are implemented for two types of datasets: ``Lab data`` and ``UAS data``. Each model directory contains its API documentation in the ``docs`` folder to help you understand the Python modules better. The ``input`` folder holds the datasets used in this study. Please note that in our Python scripts, K, psi, and alpha correspond to A, B, and ψ of the article respectively.
This code has been developed with Python version 3.8. For a list of required packages, please refer to `requirements.txt`.

## Usage Instructions

### Working with Lab Data

To use the Lab data, execute the following program:

```
python3 run_project.py
```

On launching the program, you will be prompted with the following message:

```
Choose the lab dataset to process (alg/nev/hogp/hogb):
```

Here, you should select the lab dataset to be processed (e.g., 'alg') and hit Enter. This will kick-start the program to retrieve SMC using the corresponding model for the selected dataset. All outputs from the program will be stored in the ``outputs`` folder.

After processing all four lab datasets, you can execute the following program to plot the retrieved SMC vs. estimated SMC for all four lab datasets:

```
python3 plot_smc_mes_est.py
```

### Working with UAS Data

To use the UAS data, execute the following program:

```
python3 run_project.py
```

This will start the program to retrieve SMC using the corresponding model for the UAS dataset.

Please note that the retrieval of SMC from the UAS dataset is performed through a bootstrapping process. The data is randomly split 80/20 into training and testing datasets. This random splitting is repeated 1000 times, so due to the random nature of assigning the testing & training data, the final output may slightly vary each time you run the program. All outputs from the program will be stored in the``outputs`` folder.

