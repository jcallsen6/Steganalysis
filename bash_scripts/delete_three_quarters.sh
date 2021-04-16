#!/bin/bash

i=0; 
total=$(ls "$1" | wc -l)
half=$((total/4))

for f in "$1"*; 
do 
    rm $f; 
    let i++;
    
    if [ $i -eq $half ]
    then
	   break
    fi 
done
