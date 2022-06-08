import pandas as pd
pd.set_option('display.max_colwidth', -1)
import warnings
warnings.filterwarnings('ignore')
from detoxify import Detoxify


df1 = pd.read_csv("data/labeled_comments.csv", usecols=[0])

results = []
for i in df1['comments']:
  r = Detoxify('original').predict(i)
  results.append(r)


df2 = pd.DataFrame(results)


df2.to_csv('labels.csv', index=False)

print('Finished!')
  
