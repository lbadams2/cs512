import json
import nltk

#awk '$3 == "I" {print $1, $2, $3}' bio.txt
skip_chars = ['!', '.', ',', '?', ':', '[', ']', '(', ')', ';', '"', '\\', '“', '”', '‘', '’']
with open('/Users/liamadams/Documents/school/cs512/research_project/toy2.json', 'r') as json_file:
    with open('/Users/liamadams/Documents/school/cs512/research_project/elmo/bio.txt', 'w+') as out_file:
        data = json.load(json_file)        
        for node in data:
            char_count = 0
            last_char_space = False
            multi_word_entity = False
            word = ''
            pos = ''
            prev_tag = ''
            tweet = node[0]
            tags = node[1]
            for c in tweet:
                if c in skip_chars or (last_char_space and c.isspace()):
                    if c != '\\':
                        char_count += 1
                    continue
                current_tag = tags[char_count]
                char_count += 1

                if (current_tag.startswith('I') or current_tag.startswith('E')) and last_char_space:
                    last_char_space = False
                    multi_word_entity = True
                if not c.isspace():
                    word += c
                    last_char_space = False
                else:
                    if word:
                        pos = nltk.pos_tag(word)[0][1]
                    if multi_word_entity:
                        # for apostrohpes
                        if prev_tag[1:]:
                            tag = 'I' + prev_tag[1:]
                        else:
                            tag = 'I' + tag[1:]
                        line = word + ' ' + pos + ' ' + tag
                        multi_word_entity = False
                    else:
                        if prev_tag == 'O':
                            line = word + ' ' + pos + ' ' + prev_tag
                        else:
                            tag = 'B' + prev_tag[1:]
                            line = word + ' ' + pos + ' ' + tag
                    if not word.isspace() and word:
                        out_file.write(line + '\n')
                    word = ''
                    pos = ''
                    last_char_space = True
                prev_tag = current_tag
            out_file.write('\n')