# BowelSoundsDetection
Machine Learning LSTM based NN with MFCC features for Bowel Sounds Detection (Audio Classification) 
# Setup

Create environment with required libraries using conda:

```
$ conda env create -f environment.yml
```


To a selected model train model follow the argparse arguments given in bowel.models.controller.
To see them from comman line type:
```
python3 -m bowel.models.controller -h
```

Running the model looks like e.g.:
```
python3 -m bwel.model.controller 
--mode
train_test
--data_dir
./data/processed
--division_file
./data/processed/files.csv
--trans_config
mfcc_transformation.yaml
--train_config
mfcc_train.yaml
--save_path
./models/federated
--log
--wandb_log_name
"train_test Bidirectional LSTM(256), LSTM(256), Convolution(64, kernel=15); #mfcc=40_2,5kHzHamming"
```
