from keras.models import load_model


model = load_model("test_sign.model")

# model.save_weights("./save_weights/leNet.ckpt", save_format="tf")
