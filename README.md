# DGSR_Induction

## System Requirements
- CUDA 12.1
- Python 3.8 or higher

## Installation Instructions

1. First, ensure you have CUDA 12.1 installed on your system. You can download it from the [NVIDIA website](https://developer.nvidia.com/cuda-12-1-0-download-archive).

2. Create and activate a new Python virtual environment (recommended):
```bash
# Using venv
python -m venv dgsr_env
# On Windows
.\dgsr_env\Scripts\activate
# On Linux/Mac
source dgsr_env/bin/activate
```

3. Install the required packages using pip:
```bash
pip install -r requirements.txt
```

If you encounter any issues with the installation:
- Make sure your CUDA version matches (12.1)
- For CPU-only installation, modify requirements.txt to use CPU versions of PyTorch and DGL
- If you have compatibility issues, try installing PyTorch first, then DGL, and then the remaining requirements

## Usage

After obtaining the cleaned data for the initial graph construction, you need to run new_data.py to generate the graph and subgraphs needed for the 
initial training. 

Run the file in the terminal using the following syntax:

```bash
python new_data.py --data "data"
```

instead of "data" write the name of the csv file.

Next, run generate_neg.py to speed up the test. You have to change the data name in the code before running it.

### **Initial Model training**
After creating the graph and subgraphs, run main_train_test.py to train the DGSR model. The model should be run the same way as new_data.py. Make sure to 
set the parameter model_record to true to save the model state dictionary.

### **Offline Inference**
Using new_data.py, construct a new graph using the new csv file that contains the old users and the new users.

Then run DGSR_inference.py after adding the path to the saved model. The file should also be run using the new csv file. 

### **Partial Retraining**
Construct a new graph with new and old users using new_data.py.

run partial_retrain.py to use the saved model to partially retrain for the new users. 

## Acknowledgement
The implementation is based on https://github.com/CRIPAC-DIG/DGSR 