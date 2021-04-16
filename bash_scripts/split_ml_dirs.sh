# source: https://askubuntu.com/a/584730

#!/bin/bash

mkdir "$1train";
mkdir "$1val";
mkdir "$1test";

mkdir "$1train/0";
mkdir "$1train/1";
mkdir "$1val/0";
mkdir "$1val/1";
mkdir "$1test/0";
mkdir "$1test/1";

filecount=$(ls "$10" | wc -l)

i=0; 
# split 60/20/20

for f in "$10/"*; 
do 
    if [ $i -le $((filecount/5*3)) ]
    then
        mv "$f" "$1train/0"; 
    elif [ $i -le $((filecount/5*4)) ]
    then
        mv "$f" "$1val/0";
    else
        mv "$f" "$1test/0";
    fi 
    
    let i++; 
done

i=0; 
for f in "$11/"*; 
do 
    if [ $i -le $((filecount/5*3)) ]
    then
        mv "$f" "$1train/1"; 
    elif [ $i -le $((filecount/5*4)) ]
    then
        mv "$f" "$1val/1";
    else
        mv "$f" "$1test/1";
    fi 
    
    let i++; 
done

rm -r "$10";
rm -r "$11";
