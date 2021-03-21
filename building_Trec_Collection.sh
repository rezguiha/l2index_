source activate python3.7
LANGUAGE=en
FASTTEXT_PATH=/home/mrim/data/embeddings/cc.${LANGUAGE}.300.bin


for COLL in AP88-89 LA FT91-94
do
    COLLPATH=TREC/${COLL}/
    INDEXPATH=TREC/${COLL}/indexed_collection

    python build_Trec_Collection.py -c $COLLPATH -i $INDEXPATH -f $FASTTEXT_PATH > ${COLLPATH}stdout/build 2> ${COLLPATH}stderr/build &

done
mail -s "Building_Trec_Colleciton" hamdi.rezgui1993@gmail.com <<< "finished"