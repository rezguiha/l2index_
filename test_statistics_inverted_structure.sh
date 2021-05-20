source activate python3.7


INDEXED_PATH=/home/mrim/rezguiha/work/repro_chap7_res/enwikIR_full_documents_indexed
TEST_DIR_PATH=/home/mrim/rezguiha/work/repro_chap7_res/test_evaluation_wikIR


python3 test_statistics_inverted_structure.py -i ${INDEXED_PATH}  -s ${TEST_DIR_PATH}  > ${TEST_DIR_PATH}/stdout/out_stat_python.log 2> ${TEST_DIR_PATH}/stderr/err_stat_python.log &


mail -s "Testing" hamdi.rezgui1993@gmail.com <<< "finished"