# Directory structure and running order

## Preparing the data
* Download the [dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV). You can find 62 files in 	`.txt` format. After downloading, put them all in a folder.
* Copy the script `exract_bs.py` in the same folder of the dataset.
* Create a folder named `PerBS`. Note that you will have to change the paths in the script accordingly.This is the folder where the generated files from the script are going to be copied.
* Run the script `extract_bs.py`. Running this script, will generate 10000 .csv files where each file contains the data at each cell for all the services.

There are 3 Notebooks files in the main directory. 

# Train.ipynb

* This notebook, trains models and saves the models and the test data in the `Trained_models` folder.
* The parameters of training can be adjusted in the `Parameters` cell.
* There are two different loss functions to train the models with, namely `Capacity forecasting` and `Traffic Forecasting`.
* Other Parametrs can also be adjusted, i.e. `nr` the grid size of nr*nr centered at `cell` and also `lookback` and `alpha`.

# LRP.ipynb

* This notebook runs the relevance mapping propagation algorithm on the trained models by the test data.
* You can select the center cell id at `cell` and this constructs a 21x21 region centered around `cell`. 
* For each `cell` in the 21x21 region, the algorithm constructs a 5x5 area centered at each cell.
* Further the algorithm constructs subfolders by getting the models and data of the cells in the 5*5 grid and runs the LRP algorithm.
* The LRP algorithm, returns a final image containing the LRP values which has 3 layers, each for 1 lookup. We do a smoothing average on these 3 lookups and end up with an image of 1 layer.
* Further, we find the highest and lowest value in the image at each time instance and find the equivalent cell id of those cells.
* The cell ids are stored in `cell_ids` folder. Inside this folder, there are cell_ids of most and least relevant ids for every 5x5 region in the 21x21 region and the center of the 5x5 region is the id in the numpy file.

# Attack.ipynb

* We use FGSM attack as the baseline attack for perturbing all the cells in a 5x5 grid.
* Inorder to validate our explainability tool, we sum the amount of perturbation to each of those 25 cells done by FGSM attack at each time instance, and we inject this traffic load only to the most or least relevant cells already acquired by `LRP.ipynb` at each time instance.
* The 5x5 region to perform the attack can be selected by giving the cell id in `cell`. The damage (Capacity/Traffic forecasting) can also be selected from the `Parameters` cell in the notebook.
* We call these attacks DEEXP_H/L and we evaluate the damages caused by these attacks on the center cell.
* Overall, we will have 4 different attacks namely: FGSM, BIM, DEEXP_H, DEEXP_L.

