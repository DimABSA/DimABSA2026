# Evaluation

## Format

Before using the evaluation script, make sure that both the predictions and the gold data meet the required format.

For Subtask 1:  
- Gold data should contain the columns: [ID (str), Aspect_VA (list)]  
- Prediction data should contain the columns: [ID (str), Aspect_VA (list)]  

For Subtask 2:  
- Gold data should contain the columns: [ID (str), Quadruplet (list)]  
- Prediction data should contain the columns: [ID (str), Triplet (list)]  

For Subtask 3:  
- Gold data should contain the columns: [ID (str), Quadruplet (list)]  
- Prediction data should contain the columns: [ID (str), Quadruplet (list)]  

Note:
1. Columns irrelevant to evaluation will be ignored.
2. Missing values will raise an error.  

---

## Usage

The evaluation script provides three main functions:  
- `read_jsonl_file`: read JSONL files  
- `evaluate_predictions`: scoring function for Subtasks 2 and 3  
- `evaluate_predictions_task1`: scoring function for Subtask 1  

Dependencies can be installed with:  
```bash
pip install -r requirements.txt
```

Example command for evaluating predictions:  
```bash
python metrics_subtask_1_2_3.py -t 2 -p eng_restaurant_subtask2_output.jsonl -g eng_restaurant_subtask2_gold.jsonl
```

Three variables need to be specified in order to switch between different subtasks:  
- `-t` or `--task`: the evaluation task, with three options [1, 2, 3]  
- `-g` or `--gold_data_path`: the path to the gold file  
- `-p` or `--pred_data_path`: the path to the prediction file  
