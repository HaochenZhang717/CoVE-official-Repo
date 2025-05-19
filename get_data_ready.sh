
# process data for CoVE
python process_data_tianyi.py --root "dataset/amazon/raw/beauty"
python process_data_tianyi.py --root "dataset/amazon/raw/sports"
python process_data_tianyi.py --root "dataset/amazon/raw/toys"

# process data for ablation experiment, where text information is removed
python process_data_ablation.py --root "dataset/amazon/raw/beauty"
python process_data_ablation.py --root "dataset/amazon/raw/sports"
python process_data_ablation.py --root "dataset/amazon/raw/toys"


# apart from getting dataset ready, we also need to make some change to tokenizers
cd tokenizers
python tokenizer_utils.py --model_id "Llama-3.2-3B" --data_name "beauty"
python tokenizer_utils.py --model_id "Llama-3.2-3B" --data_name "sports"
python tokenizer_utils.py --model_id "Llama-3.2-3B" --data_name "toys"


