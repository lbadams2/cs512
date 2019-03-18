#!/bin/bash

tweet="$(jq '.[0][0]' toy.json | awk '{gsub("\"","");print}')"
tags=$(echo $(jq '.[0][1]' toy.json) | awk '{gsub(",|\\[|\\]|\"","");print}')
#echo $tweet
#echo $tags
tag_array=($tags)
word=''
for (( i=0; i<${#tweet}; i++ )); do
    current_char=${tweet:$i:1}
    current_tag=tag_array[$i]
    begin_entity=false

    if [[ $current_tag == B* ]]
    then
        begin_entity=true
    fi

    if [ ! $current_char =~ "\s+" ]; then
        word="$word$current_char"
    else
        "$word $current_tag" >> toy_train
        word=''
    fi
done