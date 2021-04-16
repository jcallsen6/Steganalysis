# source for looping through subdirectories: https://stackoverflow.com/questions/4515866/iterate-through-subdirectories-in-bash
#!/bin/bash

i=0; 
for file in "$1"*.jpg
do
    mogrify -define png:format=png24 -type TrueColor -format png -colorspace RGB $file 2>&1 | grep -q "warning"
    if [ $? -eq 0 ]
    then
        rm ${file%.*}.png
    fi
    rm $file
done
