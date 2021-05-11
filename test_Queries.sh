source activate python3.7


TEST_DIR_PATH=/home/mrim/rezguiha/work/repro_chap7_res/test_queries_structure

python3 test_Queries.py -f ${TEST_DIR_PATH}  > ${TEST_DIR_PATH}/stdout/out_python.log 2> ${TEST_DIR_PATH}/stderr/err_python.log &


mail -s "Testing" hamdi.rezgui1993@gmail.com <<< "finished"