# MemeNet : Meme Template Classifier

The task at hand is to classify the following meme templates : Spiderman, Drake & Mr. Incredibles, as for the reason? there isn't one really, I did this as a learning exercise to be honest.

![Untitled](https://github.com/afieif/MemeNet/assets/60255809/6b7b0368-0c31-43b4-9cc7-d2f99fa2a742)
![Untitled](https://github.com/afieif/MemeNet/assets/60255809/f7b896d4-67a2-4161-952f-5185bafbbae0)
![Untitled](https://github.com/afieif/MemeNet/assets/60255809/1e1b2b20-757e-4b46-b7c2-815c9e4e157a)

## The model seemed to be recognizing these pretty well so I decided to turn things up a notch by giving it memes that were combinations of two templates

![image](https://github.com/afieif/MemeNet/assets/60255809/76a2cf4f-f88c-41fe-845b-9cc2c3ec2283)
![image](https://github.com/afieif/MemeNet/assets/60255809/0341e5e0-eae1-494b-b455-2ee556cc9d77)


## Finally I threw in some wildcards , just out of curiosity I tested it on some variants that were considerably different from the images in the training set

![image](https://github.com/afieif/MemeNet/assets/60255809/8936ad38-a319-4c95-a933-b1f904c93584)
![image](https://github.com/afieif/MemeNet/assets/60255809/5559dcf7-85f5-40a1-a07c-64f7ea219635)
![image](https://github.com/afieif/MemeNet/assets/60255809/85adda1e-64e1-443b-a011-3797dd3b5d56)



## Now that you've seen the results, let's have a look at the model

```
model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
```

