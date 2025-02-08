# Learning with Noisy Graph Labels

## Approach:

1. We first make use of custom GINConv model to generate Graph Embeddings for the training and testing graphs.

2. Then we initialize a classifier model.

3. Both, the classifier model and GINConv model parameters are learnt simultaneously during the training process.

5. The *load_and_test()* function loads the appropriate classifier model for predicting the true labels on the test set.

## Steps to run the code:

1. Run the below command to install the python packages and dependencies.
```shell
pip install -r requirements.txt
```

2. Run the *main.py* file using suitable command line arguments.

3. The *--batch_size* command line argument decides the batch size to be moved to the computing device. It has a default value of 128 and can be adjusted according to the back end machine.

4. The *--test_path* command line argument should contain the path to the compressed json file of the test graphs.
```shell
python main.py --test_path /path/to/test.json.gz
```

5. Optionally, the *--train_path* command line argument should contain the path to the compressed json file of the train graphs, if the user wishes to train on new graphs.
```shell
python main.py --test_path /path/to/test.json.gz --train_path /path/to/train.json.gz
```

## Output interpretation:

1. When *main.py* is run only using *test_path* argument, the predictions will be saved in the *submission* folder with the appropriate dataset name (one among A, B, C and D).

2. When *main.py* is run using both *train_path* and *test_path*,
    - The training logs will be saved in E.log file of the *logs* folder.
    - The model checkpoints will be saved in the subfolder E of the *checkpoints* folder.
    - The predictions will be saved in the *testset_E.csv* of the *submission* folder.