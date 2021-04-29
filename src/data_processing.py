import os
import json
import random

def make_list(path):
    text_units = []
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            unit = os.path.join(path, filename)
            with open(unit, 'r+') as f:
                text = f.read()
            text_units.append(text)
        else:
            continue
    return text_units

train_pos = make_list('./train/pos')
train_neg = make_list('./train/neg')

test_pos = make_list('./test/pos')
test_neg = make_list('./test/neg')

def make_data(positive, negative):
    labelled_data = []
    for unit in positive:
        mapping = {'content': unit,
                   'label': 1}
        labelled_data.append(mapping)
    for unit in negative:
        mapping = {'content': unit,
                   'label': 0}
        labelled_data.append(mapping)
    random.seed(42)
    random.shuffle(labelled_data)
    return labelled_data

train_map = make_data(train_pos, train_neg)
test_map = make_data(test_pos, test_neg)

# Saving the json files
with open('./train.json', 'w+') as f:
    json.dump(train_map, f)
    
with open('./test.json', 'w+') as f:
    json.dump(test_map, f)
