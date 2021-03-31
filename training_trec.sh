# =============================================================================
# Created By  : Hamdi REZGUI
# Created Date: March 21 2021
# E-mail: hamdi.rezgui@grenoble-inp.org
# Description: Script to train the classical IR models on the 3 TREC collections
# =============================================================================
source activate python3.7

NB_EPOC=100
L1_WEIGHT=1e-5
DROPOUT=0.0
FOLDS=5
LR=1e-3
FASTTEXT_PATH=/home/mrim/rezguiha/work/repro_chap7_res/fastText/cc.en.300.bin

for COLL in AP88-89 LA FT91-94
do
    COLLPATH=TREC/${COLL}/
    INDEXPATH=TREC/${COLL}/indexed_collection

    for MODEL in tf_idf DIR BM25 JM
    do
        python training_on_trec_collection.py -c ${COLLPATH} -i $INDEXPATH -p ${COLLPATH}plots -r ${COLLPATH}results -w ${COLLPATH}weights -e $NB_EPOC -l ${L1_WEIGHT} -n ${COLL}_${MODEL}_${L1_WEIGHT}_${DROPOUT} --lr $LR -d ${DROPOUT}  --IR_model ${MODEL} > ${COLLPATH}stdout/${MODEL}_${L1_WEIGHT}_${DROPOUT} 2> ${COLLPATH}stderr/${MODEL}_${L1_WEIGHT}_${DROPOUT} &
    done
done
mail -s "training_on_trec_collection" hamdi.rezgui1993@gmail.com <<< "finished"
