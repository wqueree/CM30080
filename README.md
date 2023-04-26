# CM30080

## Environment

Packages included in `requirements.txt`. Create a virtual environment and install:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Task 1

#### Prediction and Evaluation

1. Change into the directory `task1`:

```bash
cd ./task1
```

2. Run `task1.py`:

```bash
python ./task1.py
```

This will output predicted angles, mean squared error, accuracy, and elapsed time to the console.

### Task 2

#### Prediction

1. Change into the directory `task2`:

```bash
cd ./task2
```

2. Run `task2.py`:

```bash
python ./task2.py --test_directory_path ./test/images --train_directory_path ./train/png --sampling_levels 3 4
```

This will write prediction annotations and images to `./predict/annotations` and `./predict/images` respectively based on the templates in `./train/png` and test images in `./test/images` for sampling levels 3 and 4. You may wish to run sampling prediction in parallel in two separate shells for speed.

#### Evaluation

1. Change into the directory `task2`:

```bash
cd ./task2
```

2. Run `evaluate.py`:

```
python ./evaluate.py --pred_path_root ./predict/annotations/3 --gt_path_root ./test/annotations
````

This will output evaluation scores for the predictions in `./predict/annotations/3` against the ground truth values in `./test/annotations`.

### Task 3

#### Prediction and Evaluation

1. Change into the directory `task3`:

```bash
cd ./task3
```

2. Run `task3.py`:

```
python ./task3.py
````

This will output the TP, FP, FN and recall for each image. You can change the output type with the boolean SHOW and EVAL variables at the beginning of `task3.py`.
