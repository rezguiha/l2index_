#!/bin/bash
# ATTENTION : script prévu pour bash

# Choix du répertoire poir le téléchargement de anaconda
TMPDIR="$HOME/tmp/anaconda"
# Choix du nom de l'environnement pour python
MYENV="env1"
# Nom du fichier anaconda à télécharger
ANACONDA="Anaconda3-2020.11-Linux-x86_64.sh"

echo "Install anaconda"
echo "Using $TMPDIR"

# Création de ce répertoire
mkdir -p "$TMPDIR"
cd "$TMPDIR"

# Télécharge une version de anaconda si elle n'existe pas déjà
if [ -f "$ANACONDA" ]
then
  echo "File exists, anaconda already installed : $ANACONDA"
else
  # Téléchargement
  wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh

  # Lance l'installation
  bash Anaconda3-2020.11-Linux-x86_64.sh
fi

# Active les changements dans .bashrc
source "$HOME/.bashrc"

# Crée un environnement spécifique pour python 3.7 s'il n'existe pas déjà
if conda list -n "$MYENV" > /dev/null
then
  echo "Conda env allready exists: $MYENV"
else
  echo "Create conda env '$MYENV' with python 3.7"
  conda create -n "$MYENV" python=3.7
fi

#
# To activate this environment, use
#
#     $ conda activate env1
#
# To deactivate an active environment, use
#
#     $ conda deactivate


echo "activate $MYENV"
conda activate "$MYENV"

# Installation de tensorflow s'il n'existe pas déjà
if conda list | grep tensorflow
then
  echo "Tensorflow already installed in env '$MYENV'"
else
  conda install -c conda-forge tensorflow=1.15
fi

# Installation de fastttext s'il n'existe pas déjà => NE FONCTIONNE PAS AVEC CONDA
#if conda list | grep fasttext
#then
#  echo "fasttext already installed in env '$MYENV'"
#else
#  conda install -c conda-forge fasttext
#fi

# Installe le reste avec PIP

pip install fasttext
pip install pytrec-eval
pip install panda
pip install pandas
pip install nltk

# Telecharge les stopwords
python -c "import nltk;nltk.download('stopwords')"

echo "END of script"
echo "please activate env: conda activate $MYENV"
