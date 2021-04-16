# source: https://askubuntu.com/a/584730

#!/bin/bash

i=0; 
for f in "$1"*; 
do 
    d=$1dir_$(printf %d $((i/8000+1))); 
    mkdir -p $d; 
    mv "$f" $d; 
    let i++; 
done
