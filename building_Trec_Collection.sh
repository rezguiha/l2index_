# =============================================================================
# Created By  : Hamdi REZGUI
# Created Date: March 23 2021
# E-mail: hamdi.rezgui@grenoble-inp.org
# Description: Script to index TREC collections
# =============================================================================
source activate python3.7
LANGUAGE=en
FASTTEXT_PATH=/home/mrim/data/embeddings/cc.${LANGUAGE}.300.bin
BUILD_FOLDS=True

for COLL in AP88-89 LA FT91-94
do
    COLLPATH=TREC/${COLL}/
    INDEXPATH=TREC/${COLL}/

    python build_Trec_Collection.py -c $COLLPATH -i $INDEXPATH -f $FASTTEXT_PATH -b $BUILD_FOLDS > ${COLLPATH}stdout/build 2> ${COLLPATH}stderr/build &

done
mail -s "Building_Trec_Colleciton_FT91-94" hamdi.rezgui1993@gmail.com <<< "finished"