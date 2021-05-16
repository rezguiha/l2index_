source activate python3.7


TEST_DIR_PATH=/home/mrim/rezguiha/work/repro_chap7_res/test_baseline_generator

python3 test_baseline_generator.py -f ${TEST_DIR_PATH} -s ${TEST_DIR_PATH}/score_file -q ${TEST_DIR_PATH} > ${TEST_DIR_PATH}/stdout/out_python.log 2> ${TEST_DIR_PATH}/stderr/err_python.log &


mail -s "Testing" hamdi.rezgui1993@gmail.com <<< "finished"