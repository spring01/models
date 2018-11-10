# Export variables
PARAM_SET=big
DATA_DIR=./data
MODEL_DIR=./model_$PARAM_SET
VOCAB_FILE=$DATA_DIR/vocab.ende.32768

# Download training/evaluation datasets
python data_download.py --data_dir=$DATA_DIR

# Train the model for 10 epochs, and evaluate after every epoch.
python transformer_main.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
    --vocab_file=$VOCAB_FILE --param_set=$PARAM_SET \
    --bleu_source=test_data/newstest2014.en --bleu_ref=test_data/newstest2014.de

# Run during training in a separate process to get continuous updates,
# or after training is complete.
tensorboard --logdir=$MODEL_DIR

# Translate some text using the trained model
python translate.py --model_dir=$MODEL_DIR --vocab_file=$VOCAB_FILE \
    --param_set=$PARAM_SET --text="hello world"

# Compute model's BLEU score using the newstest2014 dataset.
python translate.py --model_dir=$MODEL_DIR --vocab_file=$VOCAB_FILE \
    --param_set=$PARAM_SET --file=test_data/newstest2014.en --file_out=translation.en
python compute_bleu.py --translation=translation.en --reference=test_data/newstest2014.de
