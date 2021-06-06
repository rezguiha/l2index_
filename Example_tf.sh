source ~/anaconda3/etc/profile.d/conda.sh
conda activate hamdi

LR=1e-3
FASTTEXT_PATH=/home/mrim/rezguiha/work/repro_chap7_res/fastText/cc.en.300.bin
VOCAB_PATH=/home/mrim/rezguiha/work/repro_chap7_res/enwikIR_full_documents_indexed
REDUCTION_RATE=1e0
python3 Example_tf.py  -f ${FASTTEXT_PATH} -v ${VOCAB_PATH}  --lr $LR -r $REDUCTION_RATE > out_python_Example_tf 2>&1  &


mail -s "training_on_trec_collection" hamdi.rezgui1993@gmail.com <<< "finished"