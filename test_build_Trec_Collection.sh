#!/bin/bash
# =============================================================================
# Created By  : Jean-Pierre Chevallet
# Created Date: May 3 2021
# Description: Test of the script build_Trec_Collection.py
# =============================================================================
LANGUAGE=en
FASTTEXT_PATH=/home/mrim/data/embeddings/cc.${LANGUAGE}.300.bin
BUILD_FOLDS=False
# Path of the original collection source
ORIGINAL_COLL_SRC="/home/mrim/data/collection/trec/TREC_VOL_5/LATIMES"
# Path to the original queries
ORIGINAL_TOPICS_SRC="/home/mrim/data/collection/trec/topics/topics.101-150.gz"
# Path to the original qrels
ORIGINAL_QRELS_SRC="/home/mrim/data/collection/trec/qrels/qrels.trec2.adhoc.gz"
# Path to the output result directory
EXPE_PATH="EXPE/TREC/LATIMES"

# Check if conda is available
if command -v conda
then
  echo "conda => OK"
else
  echo "#ERROR: conda not installed, please run 'install.sh'"
  exit 1
fi

# Check if in current env, python3.7 is active
if conda list python | grep 3.7 >/dev/null
then
  echo "python3.7 => OK"
else
  echo "#ERROR: python3.7 not found"
  echo "# wrong env ? select correct env using: conda activate <yourenv>"
  exit 1
fi

if [ -f "$FASTTEXT_PATH" ]
then
  echo "Fasttext => OK"
else
  echo "#ERROR: fasttest not found"
  echo "# Check path: $FASTTEXT_PATH"
  exit 1
fi

if [ -d "$ORIGINAL_COLL_SRC" ]
then
  echo "Original collection path => OK"
else
  echo "#ERROR: incorrect Original collection path"
  echo "# Check path: $ORIGINAL_COLL_SRC"
  exit 1
fi

# Create the experiment path
mkdir -p "$EXPE_PATH"

COLLPATH=$EXPE_PATH
INDEXPATH=$EXPE_PATH
LOGPATH=$EXPE_PATH/log
mkdir -p "$LOGPATH"

# Log files
DATE=`date +%y%m%d_%H%M%S`
LOGFILE=$LOGPATH/${DATE}.log
LOGERRORFILE=$LOGPATH/${DATE}_error.log

# Topic file
QUERIES="$EXPE_PATH/queries"

# Create topics
if [ -f "$QUERIES" ]
then
  echo "Query => exists"
else
  zcat "$ORIGINAL_TOPICS_SRC" > "$QUERIES"
fi

# Create qrels
QREL="$EXPE_PATH/qrels"
if [ -f "$QREL" ]
then
  echo "Qrel => exists"
else
  zcat "$ORIGINAL_QRELS_SRC" > "$QREL"
fi


#python build_Trec_Collection.py -c $COLLPATH -i $INDEXPATH -f $FASTTEXT_PATH -b $BUILD_FOLDS >$LOGFILE 2>$LOGERRORFILE

python build_Trec_Collection.py -o $ORIGINAL_COLL_SRC -c $COLLPATH -i $INDEXPATH -f $FASTTEXT_PATH -b $BUILD_FOLDS
