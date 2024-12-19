# Introduction
This repository contains the source code for our paper "Spotting Deep Neural Network Vulnerabilities in Mobile Traffic Forecasting with an Explainable AI Lens" published on Infocom 2022. 
You can cite us as: 
> Moghadas Gholian, S., Fiandrino, C., Collet, A., Attanasio, G., Fiore, M., & Widmer, J. (2023). Spotting Deep Neural Network Vulnerabilities in Mobile Traffic Forecasting with an Explainable AI Lens. In IEEE International Conference on Computer Communications.
## Structure of the repo
This repository contains several folders, each with a specific purpose. The following is a brief overview of what you can expect to find in each folder:\
**Results** folder includes the figures in the paper.\
**Scripts** folder includes the python files and notebooks that we used to generate the results.\
**Trained_models** folder includes the DNN models in `.h5` format that we have trained with the provided script.
## Dependencies 
- Python 3.8
- [Anaconda](https://www.anaconda.com/products/distribution). 
- jupyterlab or any IDE that supports `.ipynb` files.

## Cloning the environment
The required libararies and their compatable versions can be extracted from xai.yml file or you can directly create a conda environment which will download and install all the required packages by running `conda env create -f xai.yml` in the command line.

# Datasets
We use two datasets:\
**Milan dataset** and **EU Metropolitan Area (EUMA) dataset:**
*Unfortunately we cannot make EUMA dataset public, nevertheless, the scripts work for both cities.* 
- Each dataset provides the temporally aggregated internet activity of each cell withing it's region.
-  Milan dataset is structured as 100x100 grid area where different CDR data is captured from various base stations in the region and distributed using Voronoi-tessellation technique among all the cells.
- We only extract internet activity from these cells which is proxy of load used in each cell. The load in each cell is captured during 1 Nov 2013- 1 Jan 2014 further this load is temporally aggregated every 10 minutes. More information regarding this dataset is available [here](https://doi.org/10.1038/sdata.2015.55). 
- The EUMA dataset is structured as 48x120 grid area and contains more recent data captured in 2019 for 15 days. The load is direclty measured and is temporally aggregated every minute in each cell.
- Milan dataset is publicly available and can be accessed and downloaded from [here](https://doi.org/10.7910/DVN/EGZHFV).
-  After downloading the [dataset](https://doi.org/10.7910/DVN/EGZHFV), use the script `extract_bs.py` , to extract the internet activity from the rest of data and seperate them for each cell. 

# Methodology 
* From each region, we select a 21x21 grid in a way that the distributions of the load at each 5x5 region in that 21x21 area are similar.
* The 21x21 grid makes a total of 441 cells and in each cell we construct a 5x5 grid centered at that cell. Then, we use two state-of-the-art ML architectures for the training:
	* [Capacity forecasting model](https://gitlab.networks.imdea.org/serly_moghadas/2023_infocom_moghadas_deexp/-/tree/main/Trained_models/capacity_forecasting): Aims at forecasting the future traffic of the center cell with the goal of allocating sufficient resources for the operator to jointly minimize overprovisioning and penalty for non-served demands (Service Level Agreement (SLA) violations) 
	* [Traffic forecasting model](https://gitlab.networks.imdea.org/serly_moghadas/2023_infocom_moghadas_deexp/-/tree/main/Trained_models/mae): Aims at forecasting the future traffic of the center cell of the 5x5 grid with the goal of minimizing mean absolute error. 

# Results 

| Content in paper | Folder in Repo. | Description |
| --- | --- | --- |
| Figure 3 | [Results/fig3](https://gitlab.networks.imdea.org/serly_moghadas/2023_infocom_moghadas_deexp/-/tree/main/Results/fig3) | `Relevance scores from the analysis of the Milan dataset with the capacity forecasting predictor` |
| Figure 4 | [Results/fig4](https://gitlab.networks.imdea.org/serly_moghadas/2023_infocom_moghadas_deexp/-/tree/main/Results/fig4) | `Relevance scores from the analysis of the EUMA dataset with the capacity forecasting predictor for a grid with high capacity` |
| Figure 5 | [Results/fig5](https://gitlab.networks.imdea.org/serly_moghadas/2023_infocom_moghadas_deexp/-/tree/main/Results/fig5) | `Relevance scores from the analysis of the EUMA dataset with the traffic forecasting predictor for a grid with low traffic volumes` |
| Figure 6 | [Results/fig6](https://gitlab.networks.imdea.org/serly_moghadas/2023_infocom_moghadas_deexp/-/tree/main/Results/fig6) | `Grounding the relevance scores with traffic volumes` |
| Figure 7 | [Results/fig7](https://gitlab.networks.imdea.org/serly_moghadas/2023_infocom_moghadas_deexp/-/tree/main/Results/fig7) |  `Example of damage to the capacity predictor` |
| Figure 8 | [Results/fig8](https://gitlab.networks.imdea.org/serly_moghadas/2023_infocom_moghadas_deexp/-/tree/main/Results/fig8) | `Capacity forecasting analysis for the Milan dataset. DEEXPH/L denote respectively the perturbation attack upon having identified with DEEXP the most relevant and the least relevant cell to perturb.` |
| Figure 9 | [Results/fig9](https://gitlab.networks.imdea.org/serly_moghadas/2023_infocom_moghadas_deexp/-/tree/main/Results/fig9) | `Capacity forecasting analysis for the EUMA dataset. DEEXPH/L denote respectively the perturbation attack upon having identified with DEEXP the most relevant and the least relevant cell to perturb.` |
| Figure 10 | [Results/fig10](https://gitlab.networks.imdea.org/serly_moghadas/2023_infocom_moghadas_deexp/-/tree/main/Results/fig10) |  `Traffic forecasting analysis for the Milan dataset. We express MAE as the percentage of error increase with respect to the no attack case.` |
| Figure 11 | [Results/fig11](https://gitlab.networks.imdea.org/serly_moghadas/2023_infocom_moghadas_deexp/-/tree/main/Results/fig11) | `Traffic forecasting analysis for the EUMA dataset. MAE is expressed as the percentage of error increase with respect to the no attack case.` |
