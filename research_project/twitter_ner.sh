#!/bin/bash

rm -f toy_train
#tweet="$(jq '.[3][0]' toy.json | awk '{gsub("\"","");print}')"
c=0
while [ 'null' != "$(jq ".[$c][0]" toy.json)" ]; do
    c=$c+1
    tweet=$(jq ".[$c][0]" toy.json)
    tweet="${tweet:1}"
    tweet=${tweet::${#tweet}-3}
    tweet="$tweet "
    #tweet=$(echo $tweet | sed -E 's/[[:punct:]]/ /g')
    echo $tweet
    tags=$(echo $(jq ".[$c][1]" toy.json) | awk '{gsub(",|\\[|\\]|\"","");print}')
    tag_array=($tags)
    word=''
    prev_tag=''
    last_char_space=''
    multi_word_entity=''

    for (( i=0; i<${#tweet}; i++ )); do
        current_char=${tweet:$i:1}
        if [[ $current_char == *['!'.,?:\;\"\\]* ]]; then # maybe add \# and \@
            continue
        fi
        current_tag=${tag_array[$i]}

        if [[ $current_tag = I* && $last_char_space = 'yes' ]]; then
            echo "Multi word entity $word $current_char"
            last_char_space=''
            multi_word_entity='yes'
        fi

        if [[ $current_char = *[!\ ]* ]]; then # if current char not space
            word="$word$current_char"
            last_char_space=''
        else
            if [[ $multi_word_entity = 'yes' ]]; then
                tag="I${prev_tag:1}"
                str="$word $tag"
                multi_word_entity=''
            else
                if [[ $prev_tag = 'O' ]]; then
                    str="$word $prev_tag"
                else
                    tag="B${prev_tag:1}"
                    str="$word $tag"
                fi            
            fi
            echo $str >> toy_train
            word=''
            last_char_space='yes'
        fi
        prev_tag=$current_tag
    done
    echo '' >> toy_train
done