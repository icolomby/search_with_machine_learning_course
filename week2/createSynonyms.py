import fasttext
model = fasttext.load_model('/workspace/datasets/fasttext/title_model.bin')
file = open('/workspace/datasets/fasttext/top_words.txt','r')
threshold = 0.75
for word in file.readlines():
  word = word.replace('\n', '')
  neighbors = [x[1] for x in list(filter(lambda x: x[0] >= threshold, model.get_nearest_neighbors(word)))]
  if len(neighbors) > 0:
    print(word + ',' + ','.join(neighbors))