# =============================================================================
# Created By  : Hamdi REZGUI
# Created Date: March 30 2021
# E-mail: hamdi.rezgui@grenoble-inp.org
# Description: Script to evaluate the baseline models without training on WikIR
# collection.The result will be in stdout directory if you have no error (check
# directory stderr before)
# =============================================================================
source activate python3.7

COLLPATH=/home/mrim/rezguiha/work/repro_chap7_res/wikIRS_78/
INDEXPATH=/home/mrim/rezguiha/work/repro_chap7_res/enwikIRS_full_documents_indexed

RESULT_PLOT_PATH=/home/mrim/rezguiha/work/repro_chap7_res/test_speed_baseline/

# COLLPATH=/home/mrim/rezguiha/work/repro_chap7_res/wikIR_78/
# INDEXPATH=/home/mrim/rezguiha/work/repro_chap7_res/enwikIR_full_documents_indexed

# RESULT_PLOT_PATH=/home/mrim/rezguiha/work/repro_chap7_res/test_speed_baseline/



# COLLPATH=/home/mrim/rezguiha/work/repro_chap7_res/mini_wikIR_78/
# INDEXPATH=/home/mrim/rezguiha/work/repro_chap7_res/mini_indexed_wikIR/indexed_collection

# RESULT_PLOT_PATH=/home/mrim/rezguiha/work/repro_chap7_res/test_evaluation_wikIR/


python3 test_baseline_speed.py -c ${COLLPATH} -i $INDEXPATH  > ${RESULT_PLOT_PATH}stdout/out_speed_wikIR.log 2> ${RESULT_PLOT_PATH}stderr/err_speed_wikIR.log &

mail -s "Test speed baseline" hamdi.rezgui1993@gmail.com <<< "finished"