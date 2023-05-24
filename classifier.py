import pandas as pd
import numpy as np
import os, shutil

class Preprocess():
    def split(self, label_path="train.csv", dir="train/", to_dir="valid/", train_file="train.csv", valid_file="valid.csv", col_list=["", ""]):
        label = pd.read_csv(label_path)
        filename_valid = []
        category_valid = []
        classes = len(np.unique(label.iloc[:, 1]))
        valid_num = int(len(label)*0.2)
        os.makedirs(to_dir, exist_ok=True)
        for i in np.unique(label.iloc[:, 1]):
            L = label[label.iloc[:, 1]==i].iloc[:, 0]
            if valid_num>classes:
                for j in range(classes):
                    idx = np.random.choice(L, 1, replace=False)
                    filename_valid.append(idx[0])
                    category_valid.append(i)
                    shutil.move(dir+idx[0], to_dir)
            for j in range()
            if i<valid_num%classes:
                idx = np.random.choice(L, 2, replace=False)
                for j in range(len(idx)):
                    filename_valid.append(idx[j])
                    category_valid.append(i)
                    shutil.move(dir+idx[i], to_dir)
            else:
                idx = np.random.choice(L, 1, replace=False)
                filename_valid.append(idx[0])
                category_valid.append(i)
                shutil.move(dir+idx[0], to_dir)
        sub_valid = pd.DataFrame({
            col_list[0]: filename_valid,
            col_list[1]: category_valid
        })
        sub_valid.to_csv(valid_file, index=False)
        filename_train = label.iloc[[ i for i in range(len(label)) if label.iloc[i, 0] not in filename_valid ], 0]
        category_train = label.iloc[[ i for i in range(len(label)) if label.iloc[i, 0] not in filename_valid ], 1]
        sub_train = pd.DataFrame({
            col_list[0]: filename_train,
            col_list[1]: category_train
        })
        sub_train.to_csv(train_file, index=False)

    def classifier(self, label_path="train.csv", dir="train/", to_dir="train/"):
        label = pd.read_csv(label_path)
        fc = zip(label.iloc[:, 0], label.iloc[:, 1])
        for i, j in fc:
            os.makedirs(to_dir+str(j), exist_ok=True)
            shutil.move(dir+i, to_dir+str(j))

if __name__ == "__main__":
    preprocess = Preprocess()
    preprocess.split(label_path="label.csv", dir="training/", to_dir="train/", train_file="train.csv",
                    valid_file="test.csv", col_list=["filename", "category"])
    preprocess.split(label_path="train_ori.csv", dir="train/", to_dir="valid/", train_file="train.csv",
                    valid_file="valid.csv", col_list=["filename", "category"])
    preprocess.classifier(label_path="train.csv", dir="train/", to_dir="train/")
    preprocess.classifier(label_path="valid.csv", dir="valid/", to_dir="valid/")
#%%
import pandas as pd
import numpy as np

label = pd.read_csv("label.csv")
len(label)*0.2
#%%
import pandas as pd
import numpy as np
import os, shutil

label = pd.read_csv("label.csv")
fc = zip(label.iloc[:, 0], label.iloc[:, 1])
for i, j in fc:
    os.makedirs("train_ori/"+str(j), exist_ok=True)
    shutil.move("train_ori/"+i, "train_ori/"+str(j))
# %%
import pandas as pd
import numpy as np
import os, shutil

label = pd.read_csv("label.csv")
for i in os.listdir("train_ori"):
    L = label[label.category==int(i)].iloc[:, 0]
    idx = np.random.choice(L, 2, replace=False)
    os.makedirs("test_ori/"+i, exist_ok=True)
    for j in idx:
        shutil.move("train_ori/"+i+"/"+j, "test_ori/"+i)
# %%
import pandas as pd
import numpy as np
import os, shutil

for i in os.listdir("train_ori"):
    L = os.listdir("train_ori/"+i)
    os.makedirs("valid_ori/"+i, exist_ok=True)
    if int(i)<131:
        idx = np.random.choice(L, 2, replace=False)
        for j in idx:
            shutil.move("train_ori/"+i+"/"+j, "valid_ori/"+i)
    else:
        idx = np.random.choice(L, 1, replace=False)
        shutil.move("train_ori/"+i+"/"+idx[0], "valid_ori/"+i)
#%%
import pandas as pd
import numpy as np
import os, shutil

label = pd.read_csv("label.csv")
fc = zip(label.iloc[:, 0], label.iloc[:, 1])
for i, j in fc:
    os.makedirs("pre/"+str(j), exist_ok=True)
    shutil.move("pre/"+i, "pre/"+str(j))