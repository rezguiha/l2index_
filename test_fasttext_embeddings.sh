source activate python3.7


INDEXED_PATH=/home/mrim/rezguiha/work/repro_chap7_res/enwikIR_full_documents_indexed
TEST_DIR_PATH=/home/mrim/rezguiha/work/repro_chap7_res/test_evaluation_wikIR
FASTTEXT_PATH=/home/mrim/rezguiha/work/repro_chap7_res/fastText/cc.en.300.bin

python3 test_fasttext_embeddings.py -i ${INDEXED_PATH}  -f ${FASTTEXT_PATH}  > ${TEST_DIR_PATH}/stdout/out_fasttext_python.log 2> ${TEST_DIR_PATH}/stderr/err_fasttext_python.log &


mail -s "Testing" hamdi.rezgui1993@gmail.com <<< "finished"