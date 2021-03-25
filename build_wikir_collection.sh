source activate /home/mrim/rezguiha/anaconda3/envs/python3.7


COLL_PATH=/home/mrim/rezguiha/work/repro_chap7_res/wikIR_78
FASTTEXT_PATH=/home/mrim/rezguiha/work/repro_chap7_res/fastText/cc.en.300.bin
INDEX_PATH=/home/mrim/rezguiha/work/repro_chap7_res/enwikIR_indexed

python building_wikir_collection.py -c $COLL_PATH -i $INDEX_PATH -f $FASTTEXT_PATH > out_python_build_wikir.log 2>&1 &
mail -s "building_wikir_collection" hamdi.rezgui1993@gmail.com <<< "finished"