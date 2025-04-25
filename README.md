# ircu-los-occupancy-simulation
Python script and data for simulating IRCU occupancy based on LOS analysis for Figure 5 of IRCU Patient Dynamics and Outcomes in the COVID-19 Pandemic.

# IRCU Length-of-Stay Analysis and Occupancy Simulation

This repository contains the Python script (`LOS_Model.py`) and input data (`ircu_data.csv`) used for the Length-of-Stay (LOS) analysis and subsequent Intensive Respiratory Care Unit (IRCU) occupancy simulations presented in Figure 5 of the manuscript:

IRCU Patient Dynamics and Outcomes in the COVID-19 Pandemic.

## Description

The script performs the following main tasks:

1.  **Loads and processes IRCU Length-of-Stay (LOS) data** from the `ircu_data.csv` file.
2.  Calculates the empirical Probability Mass Function (PMF) and Survival Function (S(d)) from the LOS data.
3.  **Fits smoothing models** (LOWESS and Gaussian Process) to the observed LOS frequency/survival data for characterization.
4.  **Simulates IRCU occupancy** over time using a convolution of hypothetical admission scenarios and the empirical LOS survival function S(d).
5.  **Generates plots** corresponding to Figure 5A-F of the associated manuscript, visualizing the LOS characterization and occupancy simulation results.
6.  **Exports the data** used for generating the plots into CSV files.

## Dependencies

The script requires the following Python libraries:

*   numpy
*   pandas
*   matplotlib
*   seaborn
*   scikit-learn (specifically `StandardScaler`, `GaussianProcessRegressor`, `kernels`)
*   statsmodels (specifically `sm.nonparametric.lowess`)

You can typically install these using pip:
`pip install numpy pandas matplotlib seaborn scikit-learn statsmodels`

## Input Data

The script requires the input file `ircu_data.csv` to be present in the same directory. This file should contain the following columns:

*   `Dias en IRCU`: Integer representing the length of stay in days.
*   `Numero de Pacientes`: Integer representing the number of patients with that specific LOS.

The script expects the CSV file to use a comma (`,`) as the separator.

## How to Run

1.  Ensure all dependencies listed above are installed in your Python environment.
2.  Place the `ircu_data.csv` file in the same directory as the `LOS_Model.py` script.
3.  Run the script from your terminal: `python LOS_Model.py`


## Outputs

Running the script will:

1.  **Print status messages** to the console indicating progress and calculated values (e.g., mean LOS).
2.  **Display the generated plots** (Figures 5A-C and 5D-F) on screen.
3.  **Save the generated plots** as JPG files inside a subdirectory named `./simulation_results_LOS/plots/`:
    *   `fig5_abc_los_characterization.jpg` (Corresponds to Fig 5A, B, C)
    *   `fig5_def_occupancy_sims_revised.jpg` (Corresponds to Fig 5D, E, F)
4.  **Create a subdirectory** named `./simulation_results_LOS/plot_data_csv/`.
5.  **Save the data underlying each plot panel** as separate CSV files within the `./simulation_results_LOS/plot_data_csv/` directory (e.g., `fig5a_observed_los_counts.csv`, `fig5d_surge_occupancy.csv`, etc.).
