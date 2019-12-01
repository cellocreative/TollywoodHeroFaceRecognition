def get_hero_name(x):
    from PIL import Image
    import numpy as np
    from keras.models import load_model
    image = Image.open(x)
    image_resize = Image.Image.resize(image,[100,100])
    image_array = (np.array(image_resize))/255
    image_reshape = image_array.reshape(1,100,100,3)
    model = load_model('heropred.h5')
    prediction = model.predict_classes(image_reshape)
    if prediction == 0:
        return 'Chiranjeevi'
    elif prediction == 1:
        return 'Nagarjuna'
    elif prediction == 2:
        return 'Ram Charan'
