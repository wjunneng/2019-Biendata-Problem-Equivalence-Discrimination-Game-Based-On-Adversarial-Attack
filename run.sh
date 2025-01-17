CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model
export GLUE_DIR=$CURRENT_DIR/datasources
export OUTPUR_DIR=$CURRENT_DIR/demos/pytorch/outputs
TASK_NAME="diac"
python $CURRENT_DIR/demos/pytorch/run.py \
  --model_type=albert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train=False \
  --do_eval=False \
  --do_predict=True \
  --do_lower_case \
  --data_dir=$GLUE_DIR/${TASK_NAME}/ \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=8.0 \
  --logging_steps=14923 \
  --save_steps=14923 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir

# 每一个epoch保存一次
# 每一个epoch评估一次