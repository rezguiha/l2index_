# =============================================================================
# Created By  : Hamdi REZGUI
# Created Date: March 23 2021
# E-mail: hamdi.rezgui@grenoble-inp.org
# Description: Script to train baseline models on WikIR collection
# =============================================================================

source activate python3.7

NB_EPOC=100
L1=1e-5
DROPOUT=0.0
LR=1e-3
FASTTEXT_PATH=/home/mrim/rezguiha/work/repro_chap7_res/fastText/cc.en.300.bin

COLLPATH=/home/mrim/rezguiha/work/repro_chap7_res/wikIR_78/
INDEXPATH=/home/mrim/rezguiha/work/repro_chap7_res/enwikIR_indexed/indexed_collection


for MODEL in tf_idf DIR BM25 JM
do
    python training_wikir_collection.py -c ${COLLPATH} -i $INDEXPATH -p ${COLLPATH}plots -r ${COLLPATH}results -w ${COLLPATH}weights -e 100 -l ${L1} -n ${COLL}_${MODEL}_${L1}_${DROPOUT} --lr $LR -d ${DROPOUT}  --IR_model ${MODEL} > ${COLLPATH}stdout/${MODEL}_${L1}_${DROPOUT} 2> ${COLLPATH}stderr/${MODEL}_${L1}_${DROPOUT} &
done
mail -s "training_on_wikir_collection" hamdi.rezgui1993@gmail.com <<< "finished"