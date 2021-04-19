#!/bin/bash

mkdir "$10";
mkdir "$11";

for i in {1..5};
do
    for f in "$1dir_$i/"*; 
    do 
        if [ "${f: -9}" == "*.*.png" ]
        then
            mv "$f" "$11";
        elif [ "${f: -4}" == ".png" ]
        then
            mv "$f" "$10";
        fi

    done
done

find $1 -maxdepth 1 -wholename "$1dir_*" -delete;
