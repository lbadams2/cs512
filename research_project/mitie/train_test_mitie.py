import sys, os
parent = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parent + '/../../mitielib')

from mitie import *


def read_file(isTest, filename):
    train_samples = []
    test_samples = []
    # examples/python/
    with open(filename, 'r') as train_file:
        sentence = []
        start_tag = 0
        end_tag = 1
        word_counter = 0
        last_token_tag = False
        tag_type = ''
        tag_ranges = []
        for line in train_file:
            #line.rstrip()
            if line.isspace():
                if last_token_tag:
                    tag_ranges.append((range(start_tag, end_tag), tag_type))
                    last_token_tag = False
                if not isTest:
                    sample = ner_training_instance(sentence)
                    for tag_tuple in tag_ranges:
                        try:
                            sample.add_entity(tag_tuple[0], tag_tuple[1])
                        except:
                            print('Error ' + tokens[0] + ' ' + tokens[1] + ' ' + tokens[2])
                            exit()
                    train_samples.append(sample)
                else:
                    test_samples.append((sentence, tag_ranges))
                sentence.clear()
                tag_ranges.clear()
                start_tag = 0
                end_tag = 1
                word_counter = 0
                last_token_tag = False
                continue
            tokens = line.split()
            tag = tokens[2]
            sentence.append(tokens[0])
            if tag == 'O' or tag.startswith('B'):
                if last_token_tag:
                    tag_ranges.append((range(start_tag, end_tag), tag_type))
                    last_token_tag = False
            if tag != 'O':
                if len(tag) < 5:
                    raise ValueError('Wrong tag ' + tokens[0] + ' ' + tokens[1] + ' ' + tokens[2])
                tag_type = tag[2:]
                if last_token_tag:
                    end_tag += 1
                else:
                    if tag.startswith('I'):
                        print('Wrong tag ' + tokens[0] + ' ' + tokens[1] + ' ' + tokens[2])
                    start_tag = word_counter
                    end_tag = start_tag + 1
                    last_token_tag = True
            word_counter += 1
    if isTest:
        return test_samples
    else:
        return train_samples

train_samples = read_file(False, 'train.txt')
trainer = ner_trainer("../../MITIE-models/english/total_word_feature_extractor.dat")
for sample in train_samples:
    trainer.add(sample)
trainer.num_threads = 4
ner = trainer.train()
ner.save_to_disk("twitter_ner_model.dat")


ner = named_entity_extractor('twitter_ner_model.dat')
test_samples = read_file(True, 'test.txt')
for ts in test_samples:
    sentence = ts[0]
    tag_ranges = ts[1]
    ground_truth_entities = []
    for tr in tag_ranges:
        tag_range = tr[0]
        entity = " ".join(sentence[i] for i in tag_range)
        ground_truth_entities.append((entity, tr[1]))
    
    prediction = ner.extract_entities(sentence)
    pred_entities = []
    for e in pred_entities:
        tag_range = e[0]
        tag = e[1]
        entity_text = " ".join(sentence[i] for i in tag_range)
        pred_entities.append((entity_text, tag))

