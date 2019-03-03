import xml.etree.ElementTree as ET

e = ET.parse('/Users/liamadams/Documents/school/AutoPhrase/models/NYT/segmentation.txt').getroot()
#e = ET.parse('test_seg.txt').getroot()
multi_set = set()
single_set = set()
for aphrase in e.findall('phrase'):
    phrase_text = aphrase.text
    if ' ' in phrase_text:
        multi_set.add(phrase_text)
    else:
        single_set.add(phrase_text)

print('Number of unique single word phrases ' + str(len(single_set)))
print('Number of unique multi word phrases ' + str(len(multi_set)))
total_count = len(single_set) + len(multi_set)
print('Number of total unique phrases ' + str(total_count))

# add root
# replace &, <<<, <<, !</, <.IO62-CNI, <.MIAPJ0000PUS.>, <a