# DR_BMI Project
<a href="https://zenodo.org/doi/10.5281/zenodo.11430076"><img src="https://zenodo.org/badge/568540530.svg" alt="DOI"></a>
## Overview

This repository contains the code used for the paper "Dopamine D1 receptor activation in the striatum is sufficient to drive reinforcement of anteceding cortical patterns". The project is organized into several folders, each serving a specific purpose in the workflow. Below is an explanation of each folder and the files within them.

## Folder Structure

### analysis

This folder contains scripts for various types of analyses:

- **dynamics_analysis.py**: Calculates dynamics for one session.
- **dynamics_mat.py**: Performs mathematical calculations related to dynamics.
- **dynamics_population.py**: Aggregates dynamics calculations across all sessions.
- **learning_analysis.py**: Calculates learning metrics for one session.
- **learning_population.py**: Aggregates learning metrics across all sessions.
- **occupancy_analysis.py**: Analyzes occupancy data for one session.
- **occupancy_population.py**: Aggregates occupancy data across all sessions.

### motion

This folder includes scripts related to motion analysis:

- **motion_analysis.py**: Analyzes motion data for one session.
- **motion_population.py**: Aggregates motion data across all sessions.

### plots

This folder contains scripts for generating plots:

- **dynamics_plot.py**: Generates plots for dynamics data.
- **figures.py**: Creates various figures that didn't really fit anywhere.
- **learning_plot.py**: Generates plots for learning data.
- **motion_plot.py**: Generates plots for motion data.
- **occupancy_plot.py**: Generates plots for occupancy data.

### preprocess

This folder includes scripts for data preparation:

- **prepare_data.py**: Prepares the raw data for analysis.
- **process_data.py**: Processes the prepared data.
- **sessions.py**: Contains a dictionary with the sessions included for the analysis for each experiment type.
- **sessions_all.py**: Contains a dictionary with all the sessions.

  
### utils

This folder contains utility scripts:

- **analysis_command.py**: Contains a class with variables to run the analysis.
- **analysis_constants.py**: Contains a class with constants to run the analysis.
- **util_plots.py**: Utility functions for plotting.
- **utils_analysis.py**: General utility functions for analysis.

## Usage

1. **Preprocess the Data**: Use the scripts in the `preprocess` folder to prepare and process your data.
2. **Run Analyses**: Use the scripts in the `analysis` and `motion` folders to perform various analyses on individual sessions or aggregate data.
3. **Generate Plots**: Use the scripts in the `plots` folder to generate plots.

