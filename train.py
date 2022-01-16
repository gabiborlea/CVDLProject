from dataset import *
from utils import *
from model import *
import matplotlib.pyplot as plt

checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=5*BATCH_SIZE)


optimizer = tf.keras.optimizers.Adam(
    learning_rate=LR,
    amsgrad=False,
)
model = DepthEstimationModel()

cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)
model.compile(optimizer, loss=cross_entropy)

data_frame = load_data_frame()

train_loader = DataGenerator(
    data=data_frame[:260].reset_index(drop="true"), batch_size=BATCH_SIZE, dim=(HEIGHT, WIDTH)
)
validation_loader = DataGenerator(
    data=data_frame[260:].reset_index(drop="true"), batch_size=BATCH_SIZE, dim=(HEIGHT, WIDTH)
)
history = model.fit(
    train_loader,
    epochs=EPOCHS,
    callbacks=[cp_callback],
    validation_data=validation_loader,
)

model.summary()

model.save('saved_models/depth_model')
np.save('saved_losses/loss2', history.history['loss'])

plt.plot(history.history['loss'], label='loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='lower right')