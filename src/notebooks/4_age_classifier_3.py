import pandas as pd
import numpy  as np

from keras.models     import Sequential
from keras.layers     import Dense, Dropout, Activation, Flatten
from keras.layers     import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Get the data
datas = []
foldsToRanges = []
start, end = 0, 0
for fold in range(5):
    path = "../data/face_image_project/fold_%d_data.txt" % (fold)
    data_fold = pd.read_csv(filepath_or_buffer=path, sep="\t")
    data_fold = data_fold.sample(frac=1).reset_index(drop=True)    
    datas.append(data_fold)
    
    start = end
    end = start + len(data_fold)
    foldsToRanges.append(slice(start, end))

data = pd.concat(datas, ignore_index=True)

data = data[["user_id","original_image","face_id","gender", "age"]]

data = data[data.gender != "u"]

data.loc[data.gender == "f", "gender"] = 0
data.loc[data.gender == "m", "gender"] = 1

path_template = "../data/face_image_project/aligned/%s/landmark_aligned_face.%d.%s"
data["file_path"] = data[["user_id","face_id","original_image"]].apply(lambda x:  
                                                                   path_template % (x[0],x[1],x[2]),
                                                                   axis=1)

data["age"] = data["age"].replace("13","(8, 13)")
data["age"] = data["age"].replace("(8, 12)","(8, 13)")
data["age"] = data["age"].replace("42","(38, 42)")
data["age"] = data["age"].replace("2","(0, 2)")
data["age"] = data["age"].replace("29","(25, 32)")

data["age"] = data["age"].replace("35","(25, 32)")
data["age"] = data["age"].replace("22","(15, 20)")
data["age"] = data["age"].replace("34","(25, 32)")
data["age"] = data["age"].replace("23","(25, 32)")
data["age"] = data["age"].replace("45","(48, 53)")
data["age"] = data["age"].replace("55","(48, 53)")
data["age"] = data["age"].replace("36","(38, 43)")
data["age"] = data["age"].replace("3","(0, 2)")


data["age"] = data["age"].replace("57","(60, 100)")
data["age"] = data["age"].replace("58","(60, 100)")
data["age"] = data["age"].replace("56","(60, 100)")
data["age"] = data["age"].replace("46","(48, 53)")

data["age"] = data["age"].replace("(38, 42)","(38, 48)")
data["age"] = data["age"].replace("(8, 13)","(8, 23)")
data["age"] = data["age"].replace("32","(25, 32)")
data["age"] = data["age"].replace("(27, 32)","(25, 32)")
data["age"] = data["age"].replace("(38, 43)","(38, 48)")

# Copy original before we drop stuff
data_orig = data

## Drop the None data
data = data[data.age != 'None']

# Building a vainilla CNN
vainilla_cnn = Sequential()
vainilla_cnn.add(Convolution2D(32, 3, 3, input_shape=(227,227,3)))
vainilla_cnn.add(Activation('elu'))
vainilla_cnn.add(MaxPooling2D(pool_size=(2, 2)))
vainilla_cnn.add(Convolution2D(64, 3, 3))
vainilla_cnn.add(Activation('elu'))
vainilla_cnn.add(Convolution2D(64, 3, 3))
vainilla_cnn.add(Activation('elu'))
vainilla_cnn.add(MaxPooling2D(pool_size=(2, 2)))
vainilla_cnn.add(Flatten())
vainilla_cnn.add(Dense(output_dim=200, input_dim=500))
vainilla_cnn.add(BatchNormalization())
vainilla_cnn.add(Activation("elu"))
vainilla_cnn.add(Dense(output_dim=100, input_dim=200))
vainilla_cnn.add(BatchNormalization())
vainilla_cnn.add(Activation("elu"))
vainilla_cnn.add(Dense(output_dim=8))
vainilla_cnn.add(Activation("softmax"))

vainilla_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

data = data[["file_path","gender", "age"]]
data = pd.get_dummies(data, columns=["age"])

age_cols = [u'age_(0, 2)', u'age_(15, 20)',
       u'age_(25, 32)', u'age_(38, 48)', u'age_(4, 6)', u'age_(48, 53)',
       u'age_(60, 100)', u'age_(8, 23)']

import scipy.misc

i_width  = 227
i_height = 227

def train_generator(data):
    while True:
        start, end = 0, 32
        while end < len(data):
            data = data.sample(frac=1).reset_index(drop=True)
            sample  = data[start:end]

            X = pd.DataFrame(sample["file_path"].apply(lambda x:  img_to_array( scipy.misc.imresize(load_img(x), (i_height, i_width) ) ) ))           
            X = X["file_path"].apply(lambda x: x.reshape((1,)+ x.shape))
            X = np.vstack(X)

            Y = sample[age_cols].as_matrix()
            
            yield (X, Y)
            start += 32
            end += 32

data_orig = data_orig[["file_path","gender", "age"]]
data_orig = pd.get_dummies(data_orig, columns=["age"])

fold0to3 = pd.concat([data_orig[foldsToRanges[0]], data_orig[foldsToRanges[1]], data_orig[foldsToRanges[2]], 
                     data_orig[foldsToRanges[3]]])
fold0to3 = fold0to3[fold0to3.age_None != 1]

#history = vainilla_cnn.fit_generator(train_generator(fold0to3), samples_per_epoch=64, nb_epoch=2000)
#vainilla_cnn.save_weights('vainilla_cnn_age_1_trainfold0to3_3.h5')
vainilla_cnn.load_weights('vainilla_cnn_age_1_trainfold0to3_3.h5')

fold4 = data_orig[foldsToRanges[4]]
fold4 = fold4[fold4.age_None != 1]

test_res = vainilla_cnn.evaluate_generator(train_generator(fold4), 32*400)

# save history and test_res
np.save("vainilla_cnn_test_res_3", np.array(test_res))
print dir(history)
#np.save("vainilla_cnn_history_3", history)
