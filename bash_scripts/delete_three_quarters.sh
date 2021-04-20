#!/bin/bash

i=0; 
total=$(ls "$1" | wc -l)
three_quarter=$((3*total/4))

for f in "$1"*; 
do 
    rm $f; 
    let i++;
    
    if [ $i -eq $three_quarter ]
    then
	   break
    fi 
done
