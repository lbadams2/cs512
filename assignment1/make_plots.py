import matplotlib.pyplot as p

y1 = [102263, 36553, 27701, 22408, 15952, 8372]
y2 = [81575, 50982, 46781, 41168, 33316, 30178]
x1 = [0, .2, .4, .6, .8, 1]

p.plot(x1, y1, label='Yelp')
p.plot(x1, y2, label='NYT')
p.title('Highlight Threshold vs Unique Multi Word Phrase Count')
p.xlabel('Highlight Threshold')
p.ylabel('Unique Phrase Count')
p.legend(loc='lower left')
p.show()