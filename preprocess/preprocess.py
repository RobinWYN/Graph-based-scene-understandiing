from interaction import *
from physicalProperty import *
from trafficRule import *
import time
import pickle
from tqdm import tqdm, trange

DATABASE = 'G:/datasets/argo1/train_data_1/train'
SAVEPATH = 'G:/datasets/argo1/process_data_pkl/'

def preProcess (raw_data):
    a = interaction(raw_data)
    b = physicalFeature(raw_data)
    c = trafficRule(raw_data)
    return torch.cat([a,b,c],dim=0)

if __name__ == "__main__":
    num = 0
    start = time.time()
    for i in trange(2059, desc='Test1'):
        # if i ==1:
        #     break
        processed_data = []
        sample = torch.load(DATABASE+str(i)+'.pkl')
        for j in range(100):
            data = sample[j]
            scene_data = preProcess(data)
            processed_data.append(scene_data)
            # print("feature vector: ",preProcess(data))

            # num = num + 1
            # if num == 1:
            #     break
        savepath = SAVEPATH+'train_process_'+str(i)+'.pkl'
        torch.save(processed_data, savepath)
    end = time.time()
    print("running time: ",str(end-start))
    # torch.save()
    # sample = torch.load(savepath)
    # output = sample[40]
    # print(output)
    # torch.load()


    # data = sample[40]
    # model_input = physicalFeature(data)
    # print("ego history: ",egoHistory(data))
    # print("feature vector: ",preProcess(data))
    # print("road conflict: ",roadRightConflict(data))
    # print(model_input)