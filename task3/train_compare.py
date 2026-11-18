import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, f1_score

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

train_per_class = 1000
test_per_class = 200
img_size = 32
batch_size = 128
cnn_epochs = 6
eff_epochs = 4

(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()


def balanced(x, y, n):
    r = np.random.default_rng(seed)
    all_idx = []
    for c in range(10):
        idx = np.where(y == c)[0]
        r.shuffle(idx)
        all_idx.append(idx[:n])
    idx = r.permutation(np.concatenate(all_idx))
    return x[idx], y[idx]


train_x, train_y = balanced(train_x, train_y, train_per_class)
test_x, test_y = balanced(test_x, test_y, test_per_class)

print('train', train_x.shape, 'test', test_x.shape)

val_size = int(len(train_x) * 0.1)
fit_x, fit_y = train_x[:-val_size], train_y[:-val_size]
val_x, val_y = train_x[-val_size:], train_y[-val_size:]


def cnn_model():
    inp = keras.Input((28, 28, 1))
    x = layers.Rescaling(1. / 255)(inp)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(10, activation='softmax')(x)
    return keras.Model(inp, out)


def eff_model():
    base = keras.applications.EfficientNetB0(
        weights='imagenet', include_top=False,
        input_shape=(img_size, img_size, 3))
    base.trainable = False
    inp = base.input
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(10, activation='softmax')(x)
    return keras.Model(inp, out)


def cnn_data(x, y, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(x))
    ds = ds.map(lambda im, lb: (tf.cast(im, tf.float32), lb))
    return ds.batch(batch_size).prefetch(1)


def eff_data(x, y, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(x))

    def prep(im, lb):
        im = tf.expand_dims(im, -1)
        im = tf.image.resize(im, (img_size, img_size))
        im = tf.repeat(im, 3, -1)
        im = keras.applications.efficientnet.preprocess_input(tf.cast(im, tf.float32))
        return im, lb

    ds = ds.map(prep)
    return ds.batch(batch_size).prefetch(1)


print('training cnn')
cnn = cnn_model()
cnn.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
h_cnn = cnn.fit(cnn_data(fit_x, fit_y, True),
                validation_data=cnn_data(val_x, val_y, False),
                epochs=cnn_epochs)

print('training efficientnet')
eff = eff_model()
eff.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
h_eff = eff.fit(eff_data(fit_x, fit_y, True),
                validation_data=eff_data(val_x, val_y, False),
                epochs=eff_epochs)


def test_model(model, ds, y_true):
    loss, acc = model.evaluate(ds, verbose=0)
    pred = model.predict(ds, verbose=0).argmax(axis=1)
    f1 = f1_score(y_true, pred, average='macro')
    cm = confusion_matrix(y_true, pred)
    return loss, acc, f1, cm, pred


cnn_loss, cnn_acc, cnn_f1, cnn_cm, cnn_pred = test_model(
    cnn, cnn_data(test_x, test_y, False), test_y)
eff_loss, eff_acc, eff_f1, eff_cm, eff_pred = test_model(
    eff, eff_data(test_x, test_y, False), test_y)

print('cnn acc', cnn_acc, 'f1', cnn_f1)
print('eff acc', eff_acc, 'f1', eff_f1)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(out, exist_ok=True)

with open(os.path.join(out, 'comparison.csv'), 'w') as f:
    f.write('model,accuracy,loss,f1\n')
    f.write('cnn,%.4f,%.4f,%.4f\n' % (cnn_acc, cnn_loss, cnn_f1))
    f.write('efficientnet,%.4f,%.4f,%.4f\n' % (eff_acc, eff_loss, eff_f1))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(h_cnn.history['accuracy'], label='cnn train')
ax[0].plot(h_cnn.history['val_accuracy'], label='cnn val')
ax[0].plot(h_eff.history['accuracy'], label='eff train')
ax[0].plot(h_eff.history['val_accuracy'], label='eff val')
ax[0].set_title('accuracy')
ax[0].legend()
ax[1].plot(h_cnn.history['loss'], label='cnn train')
ax[1].plot(h_cnn.history['val_loss'], label='cnn val')
ax[1].plot(h_eff.history['loss'], label='eff train')
ax[1].plot(h_eff.history['val_loss'], label='eff val')
ax[1].set_title('loss')
ax[1].legend()
fig.savefig(os.path.join(out, 'training_curves.png'))
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].imshow(cnn_cm, cmap='Blues')
ax[0].set_title('cnn')
ax[1].imshow(eff_cm, cmap='Blues')
ax[1].set_title('efficientnet')
fig.savefig(os.path.join(out, 'confusion_matrices.png'))
plt.close(fig)

fig, ax = plt.subplots(3, 4, figsize=(10, 8))
for i, a in enumerate(ax.flat):
    a.imshow(test_x[i].squeeze(), cmap='gray')
    a.set_title('true %d cnn %d eff %d' % (test_y[i], cnn_pred[i], eff_pred[i]))
    a.axis('off')
fig.savefig(os.path.join(out, 'sample_predictions.png'))
plt.close(fig)
