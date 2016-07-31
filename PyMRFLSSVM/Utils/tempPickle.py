import pickle
data = {
    'params': params,
    'instance': instance
}
with open('devInfCPPt0.pickle', 'wb') as f:
    pickle.dump(data, f)

import pickle
from main import Instance
from linEnvLearn import Params
with open('./tmpData/devInfCPPt0.pickle', 'rb') as f:
    data = pickle.load(f)
    params = data['params']
    instance = data['instance']