import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_memory_growth(gpus[1], True)

import tensorflow_addons as tfa
from tensorflow.python.ops.gen_dataset_ops import MapDataset
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam

import os
from model.my_model import MyModel
from loss.dice_loss import DiceLoss

LOGS = './logs'
SAVED_MODELS = './saved_models'

IMAGES = './data/images'
ANNOTATIONS = './data/ann'

HEIGHT = 256
WIDTH = 256

LR = 5e-4
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-2
EPOCHS = 10

def configure_for_performance(ds: MapDataset) -> MapDataset:
    ds = ds.cache()
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def get_dataset() -> MapDataset:
    normalization = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

    images = image_dataset_from_directory(
        IMAGES, 
        label_mode=None, 
        image_size=(HEIGHT, WIDTH), 
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    ground_truth = image_dataset_from_directory(
        ANNOTATIONS,
        label_mode=None,
        image_size=(HEIGHT, WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    dataset = tf.data.Dataset.zip((images, ground_truth))
    dataset = dataset.map(lambda low_res, hi_res: (normalization(low_res), normalization(hi_res)))
    dataset = configure_for_performance(dataset)

    return dataset


def data_term_loss(y_true, y_pred):
    y_true = tf.concat((tf.ones_like(y_true) - y_true, y_true), -1)
    matching_indicator = tf.math.reduce_sum(y_true, axis=3)
    matching_penalty = tf.math.square(tf.math.subtract(y_pred[:, :, :, 0], y_true[:, :, :, 1]))
    data_term = tf.math.multiply(matching_indicator, matching_penalty)
    data_term = tf.math.reduce_sum(data_term, axis=2)
    data_term = tf.math.reduce_sum(data_term, axis=1)
    return data_term



def compute_loss(label, matt_alpha):
    per_example_loss = data_term_loss(y_true=label, y_pred=matt_alpha)
    pred_loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=BATCH_SIZE)
    return pred_loss


if __name__ == '__main__':
    if not os.path.exists(LOGS):
        os.mkdir(LOGS)

    if not os.path.exists(SAVED_MODELS):
        os.mkdir(SAVED_MODELS)

    tb_dirs = os.listdir(LOGS)
    if len(tb_dirs) == 0:
        idx = '1'
    else:
        idx = str(int(tb_dirs[-1]) + 1)

    tensorboard_dir = os.path.join(LOGS, idx)
    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)

    num_classes = 1

    tf.debugging.experimental.enable_dump_debug_info(
        tensorboard_dir,
        tensor_debug_mode="FULL_HEALTH", 
        circular_buffer_size=-1)

    train_writer = tf.summary.create_file_writer(tensorboard_dir)

    optimizer = tfa.optimizers.AdamW(learning_rate = LR, weight_decay = WEIGHT_DECAY)
    # optimizer = Adam(learning_rate=LR)
    train_acc_metric = tf.keras.metrics.MeanIoU(
        num_classes=num_classes + 1 if num_classes == 1 else num_classes, name='train_accuracy'
    )


    my_model = MyModel()

    x = tf.keras.Input(shape=(HEIGHT, WIDTH, 3), name='inputs')
    model = my_model.build(x)

    model.summary()
    dice_loss = DiceLoss()

    dataset = get_dataset()

    @tf.function
    def train_step(x, y) -> tuple:
        with tf.GradientTape(persistent=True) as tape:
            logits = model(x, training=True)

            loss_value = dice_loss(y, logits)

        grads = tape.gradient(loss_value, model.trainable_weights)

        # Apply some clipping
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]

        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        train_acc_metric.update_state(y, logits)
        train_acc = train_acc_metric.result()

        return loss_value, train_acc

    total_steps = len(os.listdir(IMAGES)) / BATCH_SIZE
    if isinstance(total_steps, float):
        total_steps = int(total_steps) + 1

    for epoch in range(EPOCHS):
        step = 0
        final_acc = final_loss = 0
        for batch_x, batch_Y in dataset.take(total_steps):
            loss_value, acc = train_step(batch_x, batch_Y)

            print(
                '{:03d}/{:03d} {:04d}/{:04d}, Loss: {:6f}, Accuracy: {:6f}'
                .format(epoch + 1, EPOCHS, step + 1, total_steps, loss_value, acc)
            )

            step += 1

            final_acc += acc
            final_loss += loss_value

        with train_writer.set_as_default():
            tf.summary.scalar('loss', final_loss / step, step=epoch)
            tf.summary.scalar('accuracy', final_acc / step, step=epoch)

        train_acc_metric.reset_states()

    save_path = os.path.join(SAVED_MODELS, idx)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    model.save(save_path)
