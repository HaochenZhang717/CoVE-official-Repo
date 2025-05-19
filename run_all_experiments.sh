#python process_data_tianyi.py --root dataset/amazon/raw/sports
#python process_data_tianyi.py --root dataset/amazon/raw/toys


# run training for CoVE
# compress rate 2
bash finetune_sports_outdoors.sh 2 1e-4 14 8 2 4
bash finetune_beauty.sh 2 1e-4 14 8 2 4
bash finetune_toys_games.sh 2 1e-4 14 8 2 4
# compress rate 4
bash finetune_sports_outdoors.sh 2 1e-4 14 8 4 4
bash finetune_beauty.sh 2 1e-4 14 8 4 4
bash finetune_toys_games.sh 2 1e-4 14 8 4 4
# compress rate 8
bash finetune_sports_outdoors.sh 2 1e-4 14 8 8 4
bash finetune_beauty.sh 2 1e-4 14 8 8 4
bash finetune_toys_games.sh 2 1e-4 14 8 8 4
# compress rate 16
bash finetune_sports_outdoors.sh 2 1e-4 14 8 16 4
bash finetune_beauty.sh 2 1e-4 14 8 16 4
bash finetune_toys_games.sh 2 1e-4 14 8 16 4
# compress rate 32
bash finetune_sports_outdoors.sh 2 1e-4 14 8 32 4
bash finetune_beauty.sh 2 1e-4 14 8 32 4
bash finetune_toys_games.sh 2 1e-4 14 8 32 4

# run training for CoVE tuning weights only but not tune embedding layer
# compress rate 2
bash finetune_sports_outdoors_lora_only.sh 2 1e-4 14 8 2 4
bash finetune_beauty_lora_only.sh 2 1e-4 14 8 2 4
bash finetune_toys_games_lora_only.sh 2 1e-4 14 8 2 4
# compress rate 4
bash finetune_sports_outdoors_lora_only.sh 2 1e-4 14 8 4 4
bash finetune_beauty_lora_only.sh 2 1e-4 14 8 4 4
bash finetune_toys_games_lora_only.sh 2 1e-4 14 8 4 4
# compress rate 8
bash finetune_sports_outdoors_lora_only.sh 2 1e-4 14 8 8 4
bash finetune_beauty_lora_only.sh 2 1e-4 14 8 8 4
bash finetune_toys_games_lora_only.sh 2 1e-4 14 8 8 4
# compress rate 16
bash finetune_sports_outdoors_lora_only.sh 2 1e-4 14 8 16 4
bash finetune_beauty_lora_only.sh 2 1e-4 14 8 16 4
bash finetune_toys_games_lora_only.sh 2 1e-4 14 8 16 4
# compress rate 32
bash finetune_sports_outdoors_lora_only.sh 2 1e-4 14 8 32 4
bash finetune_beauty_lora_only.sh 2 1e-4 14 8 32 4
bash finetune_toys_games_lora_only.sh 2 1e-4 14 8 32 4


# run training for CoVE without text information
# compress rate 2
bash finetune_sports_outdoors_no_text.sh 2 1e-4 14 8 2 4
bash finetune_beauty_no_text.sh 2 1e-4 14 8 2 4
bash finetune_toys_games_no_text.sh 2 1e-4 14 8 2 4
# compress rate 4
bash finetune_sports_outdoors_no_text.sh 2 1e-4 14 8 4 4
bash finetune_beauty_no_text.sh 2 1e-4 14 8 4 4
bash finetune_toys_games_no_text.sh 2 1e-4 14 8 4 4
# compress rate 8
bash finetune_sports_outdoors_no_text.sh 2 1e-4 14 8 8 4
bash finetune_beauty_no_text.sh 2 1e-4 14 8 8 4
bash finetune_toys_games_no_text.sh 2 1e-4 14 8 8 4
# compress rate 16
bash finetune_sports_outdoors_no_text.sh 2 1e-4 14 8 16 4
bash finetune_beauty_no_text.sh 2 1e-4 14 8 16 4
bash finetune_toys_games_no_text.sh 2 1e-4 14 8 16 4
# compress rate 32
bash finetune_sports_outdoors_no_text.sh 2 1e-4 14 8 32 4
bash finetune_beauty_no_text.sh 2 1e-4 14 8 32 4
bash finetune_toys_games_no_text.sh 2 1e-4 14 8 32 4


#run evaluation for different datasets
# below is an example
#bash evaluate-finetune.sh ./llama3.23B-lora-only-noqloraall-ours-toys-2-lora8-batch32-num_hashes4-lr1e-4/checkpoint-2428 8 llama3.23B-lora-only-noqloraall-ours-toys-2-lora8-batch32-num_hashes4-lr1e-4-epoch4 0 ./dataset/amazon/raw/toys ./tokenizers/Llama-3.2-3B/toys/tokenizer 11924 2
