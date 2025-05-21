export PYTHONPATH=$PYTHONPATH:$(pwd)
jupyter notebook --ip='*' --NotebookApp.token='' --NotebookApp.password=''
