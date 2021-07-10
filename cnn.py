import os , random, numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt



kategoriler = ["Cat","Dog"]
yol = "PetImages"
veri = []
boyut = 180


for kategori in kategoriler:
    kategori_yol = os.path.join(yol,kategori)
    deger = kategoriler.index(kategori)
    
    for resim_yol in tqdm(os.listdir(kategori_yol)):
        resim_ad = os.path.join(kategori_yol,resim_yol)
        deger2 = os.listdir(kategori_yol).index(resim_yol)
        resim = cv2.imread(resim_ad,cv2.IMREAD_GRAYSCALE)
        if(resim is None):
            
           
            print("Bozuk Resim:",resim_ad)
        else:
            resim = cv2.resize(resim,(boyut,boyut))
            veri.append([resim,deger])
        
rastgele = random.randrange(0,1000)    



random.shuffle(veri)
X = []
Y = []

for x,y in veri:
    
    X.append(x)
    Y.append(y)
    
#plt.imshow(cv2.cvtColor(X[rastgele],cv2.COLOR_BGR2RGB),cmap="gray")
#plt.show()


X = np.array(X).reshape(-1,boyut,boyut,3)
Y = np.array(Y).reshape(-1,1)


X = X / 255.0

from tensorflow.keras import optimizers
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, Flatten, Dense, MaxPool2D

model = Sequential()
model.add(Conv2D(filters=128,kernel_size=(3,3),input_shape=X[0].shape,activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(), metrics=['acc'])

mc = ModelCheckpoint('en_iyi_model.h5',save_best_only=True,monitor='val_loss',mode='min')

model.fit(X,Y,batch_size=32,epochs=100,shuffle=True,validation_split=0.2,callbacks=[mc])


plt.figure(figsize=(4,4))
plt.plot(model.history.history['acc'])
plt.plot(model.history.history['val_acc'])
plt.title('Model Doğruluk Oranı')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='lower right')
plt.savefig('sema.png',dpi=300)
print(max(model.history.history['val_acc']))
plt.show()


from tensorflow.keras.models import load_model
yuklenen_model = load_model('en_iyi_model.h5')

print(yuklenen_model.predict(X)[855])

plt.imshow(X[855],cmap='gray')