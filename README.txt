The following requirements must be installed:
DGL 2.0.0
PyTorch 2.2.2
Cuda 12.1


After obtaining the cleaned data for the initial graph construction, you need to run new_data.py to generate the graph and subgraphs needed for the 
initial training. 

Run the file in the terminal using the following syntax:

python \new_data.py --data "data"

instead of "data" write the name of the csv file.

Next, run generate_neg.py to speed up the test. You have to change the data name in the code before running it.

**Initial Model training**
After creating the graph and subgraphs, run main_train_test.py to train the DGSR model. The model should be run the same way as new_data.py. Make sure to 
set the parameter model_record to true to save the model state dictionary.

**Offline Inference**
Using new_data.py, construct a new graph using the new csv file that contains the old users and the new users.

Then run DGSR_inference.py after adding the path to the saved model. The file should also be run using the new csv file. 

**Partial Retraining**
Construct a new graph with new and old users using new_data.py.

run partial_retrain.py to use the saved model to partially retrain for the new users. 

