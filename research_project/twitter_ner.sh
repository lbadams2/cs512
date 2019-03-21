#!/bin/bash

TRAIN_FILE='train.json'
PROCESSED_FILE='train_processed'
rm -f $PROCESSED_FILE
#tweet="$(jq '.[3][0]' toy.json | awk '{gsub("\"","");print}')"
json_count=0
while [ 'null' != "$(jq ".[$json_count][0]" $TRAIN_FILE)" ]; do    
    tweet="$(jq ".[$json_count][0]" $TRAIN_FILE)"
    tweet="${tweet:1}"
    tweet=${tweet::${#tweet}-3}
    tweet="$tweet "
    #tweet=$(echo $tweet | sed -E 's/[[:punct:]]/ /g')
    echo $tweet
    tags=$(echo $(jq ".[$json_count][1]" $TRAIN_FILE) | awk '{gsub(",|\\[|\\]|\"","");print}')
    json_count=$(($json_count+1))
    tag_array=($tags)
    word=''
    prev_tag=''
    last_char_space=''
    multi_word_entity=''
    char_count=0
    for (( i=0; i<${#tweet}; i++ )); do
        current_char=${tweet:$i:1}        
        if [[ $current_char == *['!'.“”,?:\[\]\(\)\;\"\\]* ]]; then # maybe add \# and \@
            #echo "skip $current_char previous tag is $prev_tag"
            if [[ $current_char != \\ ]]; then
                char_count=$(($char_count+1))
            fi
            continue
        fi
        current_tag=${tag_array[$char_count]}
        #echo "$current_char $char_count $current_tag"
        char_count=$(($char_count+1))

        if [[ ( $current_tag = I* || $current_tag = E* ) && $last_char_space = 'yes' ]]; then
            #echo "Multi word entity $word $current_char"
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
            if [[ $word = *[!\ ]* ]]; then
                echo $str >> $PROCESSED_FILE
            fi
            word=''
            last_char_space='yes'
        fi
        prev_tag=$current_tag
    done
    echo '' >> $PROCESSED_FILE
done