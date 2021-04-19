#!/bin/bash

i=0; 
total=$(ls "$1" | wc -l)
quarter=$((total/4))

for f in "$1"*; 
do 
    rm $f; 
    let i++;
    
    if [ $i -eq $quarter ]
    then
	   break
    fi 
done
