# Proejct
For the detial of the project please see [FYP_report.pdf](https://github.com/SuperElephant/Bisimulation_fyp_2019/blob/master/FYP_report/FYP_report.pdf)

# User Manuel

## Installation
To run these tools, the depend packages need to be installed.
```
$ pip install -r requirements.txt
```
Notice that, it is recommend to installed in virtual environment.

## Usage

These tools is developed on ```Python 2.7```.

For quick test, please run the command directly without any parameters.
```
$ python ml_algorithm/ml_algorithm.py
$ python standard_bisim/test_cases_generator.py
```

To run the wide and deep experiments:
```
$ python experiments.py 
```

Follows are specific direction:

```
$ python ml_algorithm/ml_algorithm.py -h

usage: ml_algorithm.py [-h] [-e EPOCH] [-l LEARNING_RATE] [-b BATCH_SIZE]
                       [-p DATA_PATH] [-r TEST_TRAIN_RATE] 
                       [-c CONTINUE_TRAIN] [-n MODEL_NAME]

optional arguments:
  -h, --help            show this help message and exit
  -e EPOCH, --epoch EPOCH
                        Number of training epochs
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        Initial learning rate
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Number of data for one batch
  -p DATA_PATH, --data_path DATA_PATH
                        Path to input data
  -r TEST_TRAIN_RATE, --test_train_rate TEST_TRAIN_RATE
                        The rate of test cases and train cases
  -c CONTINUE_TRAIN, --continue CONTINUE_TRAIN
                        Continue last training
  -n MODEL_NAME, --model_name MODEL_NAME
                        The name of the model
```

```
$ python standard_bisim/test_cases_generator.py -h

usage: test_cases_generator.py [-h] [-t {random,all}] [-n NUMBER]
                               [-f FILE_NAME] [-v NODE_NUMBER]
                               [-e EDGE_TYPE_NUMBER] [-r P_RATE]
                               [-p PROBABILITY]

optional arguments:
  -h, --help            show this help message and exit
  -t {random,all}, --type {random,all}
                        Type of data set
  -n NUMBER, --number NUMBER
                        The length of data set
  -f FILE_NAME, --file_name FILE_NAME
                        Name of the output file
  -v NODE_NUMBER, --node_number NODE_NUMBER
                        Number of the nodes of the graph in the data set
  -e EDGE_TYPE_NUMBER, --edge_type-number EDGE_TYPE_NUMBER
                        The total types of the edge in the graphs
  -r P_RATE, --p_rate P_RATE
                        Rate of the positive cases over all cases
  -p PROBABILITY, --probability PROBABILITY
                        The density of the random generate graphs
```


To visulise the performance of machine learning please use ```TensorBoard```
```
$ tensorboard --logdir <path-to-summary-folder> --host localhost
```

*Note: the experiments data in the report can be downloaded [HERE](https://drive.google.com/file/d/1xuVKh_E8Z671xtPgPN6QsE7gROKCKNY2/view?usp=sharing)*
