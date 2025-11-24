Starter Kit (Subtask 2 & 3)

Train and run the Dimensional Aspect-Based Sentiment Analysis (DimABSA) model for multilingual datasets.  
Supports Task 2 (Triplet Extraction) and Task 3 (Quadruplet Extraction).

#---- Folder Structure ----#
./data/              # Training and inference data (.jsonl)
./model/             # Saved model checkpoints
./log/               # Training logs
./tasks/subtask_2/   # Task 2 output files
./tasks/subtask_3/   # Task 3 output files


#---- Note ----#
Before running:
“Place all dataset files in ./data/ and ensure required dependencies (transformers, torch, etc.) are installed.”

After running:
“Predictions will be saved automatically to ./tasks/subtask_2/ and ./tasks/subtask_3/ depending on the task.”


#----Key Arguments----#
--task <int>
Task type: 2 or 3
2 → outputs triplets to ./tasks/subtask_2/
3 → outputs triplets and quadruplets to both ./tasks/subtask_2/ and ./tasks/subtask_3/

--domain <str>
Dataset domain (res | lap | hot | fin)

--language <str>
Dataset language (eng | zho)

--train_data <str>
Training data filename under ./data/

--infer_data <str>
Inference (test) data filename under ./data/

--bert_model_type <str>
Pretrained BERT model name or local path
Example: bert-base-multilingual-uncased

--mode <str>
Operation mode:
train → trains model and performs inference
inference → loads trained model and performs prediction only

--epoch_num <int>
Number of training epochs (default: 3)

--batch_size <int>
Training batch size (default: 4)

--learning_rate <float>
Learning rate for non-BERT parameters (default: 1e-3)

--tuning_bert_rate <float>
Learning rate for BERT fine-tuning (default: 1e-5)

--inference_beta <float>
Confidence threshold for prediction filtering (default: 0.9)

--gpu <bool>
Enable CUDA (default: True)

--reload <bool>
Resume training from checkpoint (default: False)


#---- Task 2 – Triplet Extraction ----#
{"ID": "res_dev_1", "Triplet": [
  {"Aspect": "food", "Opinion": "great", "VA": "7.8#7.2"}
]}

#---- Task 3 – Quadruplet Extraction ----#
{"ID": "res_dev_1", "Quadruplet": [
  {"Aspect": "food", "Category": "FOOD#QUALITY", "Opinion": "great", "VA": "7.8#7.2"}
]}


#---- Model Training Example----#
python run_task2&3_trainer_multilingual.py \
  --task 3 \
  --domain res \
  --language eng \
  --train_data eng_restaurant_train_alltasks.jsonl \
  --infer_data eng_restaurant_dev_task2.jsonl \
  --bert_model_type bert-base-multilingual-uncased \
  --mode train

#---- Model Inference Example----#
python run_task2&3_trainer_multilingual.py \
  --task 3 \
  --domain res \
  --language eng \
  --train_data eng_restaurant_train_alltasks.jsonl \
  --infer_data eng_restaurant_dev_task2.jsonl \
  --bert_model_type bert-base-multilingual-uncased \
  --mode inference