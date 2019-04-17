
output_ent_tuples = {}
test_ent_tuples = {}

debug_output_tweet_dict = {}
debug_test_tweet_dict = {}

def convert_output_tag(tag):
    if tag == 'person':
        return 'PER'
    elif tag == 'geo-loc':
        return 'LOC'
    elif tag in ['band', 'company', 'sportsteam']:
        return 'ORG'
    else:
        return 'MISC'

def add_entity_ranges(entity_tags, tweet_num, isOutput):
    last_token_tag = False
    start_tag = 0
    end_tag = 1
    tag_type = ''
    tag_ranges = []
    for i, et in enumerate(entity_tags):
        tag = et[1]
        if tag == 'O' or tag.startswith('B'):
            if last_token_tag:
                tag_ranges.append((range(start_tag, end_tag), tag_type))
                last_token_tag = False
        if tag != 'O':
            tag_type = tag[2:]
            if isOutput:
                tag_type = convert_output_tag(tag_type)
            if last_token_tag:
                end_tag += 1
            else:
                start_tag = i
                end_tag = start_tag + 1
                last_token_tag = True
    if last_token_tag:
        tag_ranges.append((range(start_tag, end_tag), tag_type))
    if isOutput:
        #if len(tag_ranges) > 0:
        #    print('pred entity line', tweet_num)
        output_ent_tuples[tweet_num] = tag_ranges
    else:
        test_ent_tuples[tweet_num] = tag_ranges

def read_output():
    skip_chars = ['!', '.', ',', '?', ':', '[', ']', '(', ')', ';', '"', '\\', '“', '”', '‘', '’']
    with open('research_project/osu/output.txt', 'r') as output:
        line_num = 0
        for line in output:
            words = line.split()
            entity_tags = []
            for word in words:                
                tokens = word.split('/')
                if tokens[0] in skip_chars:
                    continue
                tag = tokens[-1]
                entity = ''.join(tokens[:-1])
                entity_tags.append((entity, tag))
            debug_output_tweet_dict[line_num] = entity_tags            
            add_entity_ranges(entity_tags, line_num, True)
            line_num += 1
        print('output samples', line_num)

def read_test():
    per_true_pos = 0
    per_num_predictions = 0
    per_true_total = 0
    
    loc_true_pos = 0
    loc_num_predictions = 0
    loc_true_total = 0

    org_true_pos = 0
    org_num_predictions = 0
    org_true_total = 0

    misc_true_pos = 0
    misc_num_predictions = 0
    misc_true_total = 0

    true_total = 0
    num_predictions = 0
    true_pos = 0
    with open('research_project/test_processed') as test_file:
        tweet_num = 0
        entity_tags = []
        for line in test_file:
            if line.isspace():
                debug_test_tweet_dict[tweet_num] = entity_tags
                add_entity_ranges(entity_tags, tweet_num, False)
                output_tag_ranges = output_ent_tuples[tweet_num]
                test_tag_ranges = test_ent_tuples[tweet_num]

                per_true_total += sum(tr[1] == 'PER' for tr in test_tag_ranges)
                loc_true_total += sum(tr[1] == 'LOC' for tr in test_tag_ranges)
                org_true_total += sum(tr[1] == 'ORG' for tr in test_tag_ranges)
                misc_true_total += sum(tr[1] == 'MISC' for tr in test_tag_ranges)

                per_num_predictions += sum(tr[1] == 'PER' for tr in output_tag_ranges)
                loc_num_predictions += sum(tr[1] == 'LOC' for tr in output_tag_ranges)
                org_num_predictions += sum(tr[1] == 'ORG' for tr in output_tag_ranges)
                misc_num_predictions += sum(tr[1] == 'MISC' for tr in output_tag_ranges)

                true_total += len(test_tag_ranges)
                num_predictions += len(output_tag_ranges)
                for pred in output_tag_ranges:
                    end_equal = sum(pred[1] == t[1] for t in test_tag_ranges)
                    if pred in test_tag_ranges:
                        true_pos += 1
                        if pred[1] == 'PER':
                            per_true_pos += 1
                        elif pred[1] == 'LOC':
                            loc_true_pos += 1
                        elif pred[1] == 'ORG':
                            org_true_pos += 1
                        else:
                            misc_true_pos += 1
                entity_tags = []
                tweet_num += 1
            else:
                tokens = line.split()
                entity_tags.append((tokens[0], tokens[2]))
        print('test samples', tweet_num)
    
    precision = true_pos / num_predictions
    recall = true_pos / true_total
    f1 = 2 * precision * recall / (precision + recall)
    print('precision: ', precision)
    print('recall: ', recall)
    print('f1: ', f1)

    precision = per_true_pos / per_num_predictions
    recall = per_true_pos / per_true_total
    f1 = 2 * precision * recall / (precision + recall)
    print('\nper precision: ', precision)
    print('per recall: ', recall)
    print('per f1: ', f1)

    precision = loc_true_pos / loc_num_predictions
    recall = loc_true_pos / loc_true_total
    f1 = 2 * precision * recall / (precision + recall)
    print('\nloc precision: ', precision)
    print('loc recall: ', recall)
    print('loc f1: ', f1)

    precision = org_true_pos / org_num_predictions
    recall = org_true_pos / org_true_total
    f1 = 2 * precision * recall / (precision + recall)
    print('\norg precision: ', precision)
    print('org recall: ', recall)
    print('org f1: ', f1)

    precision = misc_true_pos / misc_num_predictions
    recall = misc_true_pos / misc_true_total
    f1 = 2 * precision * recall / (precision + recall)
    print('\nmisc precision: ', precision)
    print('misc recall: ', recall)
    print('misc f1: ', f1)


def run():
    read_output()
    read_test()

if __name__ == '__main__':
    run()