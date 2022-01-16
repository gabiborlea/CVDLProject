from vizualization import *
from dataset import *
from utils import *

data_frame = load_data_frame()

model = tf.keras.models.load_model('saved_models/depth_model')

test_loader = next(
    iter(
        DataGenerator(
            data=data_frame[313:].reset_index(drop="true"), batch_size=15, dim=(HEIGHT, WIDTH)
        )
    )
)
visualize_depth_map(test_loader, test=True, model=model, size=15)
