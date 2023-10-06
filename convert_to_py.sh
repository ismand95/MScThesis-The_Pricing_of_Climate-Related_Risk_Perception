#!/bin/bash

# run jupyter nbconvert on all iPython notebooks in folder
for file in *.ipynb; do
    jupyter nbconvert --to script $file
done
