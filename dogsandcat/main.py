import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_fscore_support
import os
import sys
import subprocess
import importlib
from pathlib import Path

IN_COLAB = 'google.colab' in sys.modules

if '__file__' in globals():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    # In notebook environments (e.g., Colab), __file__ is not available.
    BASE_DIR = os.getcwd()

DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
TEST_DIR = os.path.join(DATASET_DIR, 'test')
KAGGLE_DATASET_HANDLE = 'bhavikjikadara/dog-and-cat-classification-dataset'
IMG_SIZE = (96, 96)
BATCH_SIZE = 16
EPOCHS = 10
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42


def has_image_files(directory_path: Path) -> bool:
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            if Path(file_name).suffix.lower() in image_exts:
                return True
    return False


def find_class_root(start_dir: Path) -> Path | None:
    # A class root is a directory that directly contains class folders with image files.
    for root, dirs, _ in os.walk(start_dir):
        if len(dirs) < 2:
            continue
        root_path = Path(root)
        class_dirs = [root_path / d for d in dirs]
        if sum(has_image_files(d) for d in class_dirs) >= 2:
            return root_path
    return None


def resolve_data_dirs() -> tuple[str, str | None, str | None]:
    if os.path.isdir(DATASET_DIR):
        return DATASET_DIR, TRAIN_DIR if os.path.isdir(TRAIN_DIR) else None, TEST_DIR if os.path.isdir(TEST_DIR) else None

    if IN_COLAB and os.path.isdir('/content/dataset'):
        colab_train = '/content/dataset/train'
        colab_test = '/content/dataset/test'
        return '/content/dataset', colab_train if os.path.isdir(colab_train) else None, colab_test if os.path.isdir(colab_test) else None

    try:
        kagglehub = importlib.import_module('kagglehub')
    except ImportError as exc:
        if IN_COLAB:
            print('Chưa có kagglehub, đang cài đặt trên Colab...')
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'kagglehub'])
            kagglehub = importlib.import_module('kagglehub')
        else:
            raise FileNotFoundError(
                f'Không tìm thấy thư mục dữ liệu local: {DATASET_DIR}. '
                "Cài kagglehub bằng lệnh: pip install kagglehub"
            ) from exc

    print(f'Tải dataset từ Kaggle: {KAGGLE_DATASET_HANDLE}')
    downloaded_path = Path(kagglehub.dataset_download(KAGGLE_DATASET_HANDLE))
    print(f'Path dataset từ KaggleHub: {downloaded_path}')

    class_root = find_class_root(downloaded_path)
    if class_root is None:
        raise FileNotFoundError(
            'Không tìm được thư mục class trong dataset đã tải. '
            f'Vui lòng kiểm tra cấu trúc trong: {downloaded_path}'
        )

    train_candidate = None
    test_candidate = None
    for child in class_root.iterdir():
        if not child.is_dir():
            continue
        child_name = child.name.lower()
        if 'train' in child_name:
            train_candidate = str(child)
        elif 'test' in child_name:
            test_candidate = str(child)

    if train_candidate and test_candidate:
        return str(class_root), train_candidate, test_candidate

    return str(class_root), None, None

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Sử dụng GPU: {gpus}")
else:
    print("Không có GPU CUDA khả dụng, sẽ train bằng CPU.")

resolved_dataset_dir, resolved_train_dir, resolved_test_dir = resolve_data_dirs()
print(f'Dataset root đang dùng: {resolved_dataset_dir}')

has_train_test_layout = resolved_train_dir is not None and resolved_test_dir is not None

if has_train_test_layout:
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        resolved_train_dir,
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        subset='both',
        seed=SEED,
        shuffle=True,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        resolved_test_dir,
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    class_names = train_ds.class_names

    if class_names != test_ds.class_names:
        raise ValueError(
            "Class train/test không khớp. "
            f"Train: {class_names}, Test: {test_ds.class_names}"
        )
else:
    combined_val_test_split = VAL_SPLIT + TEST_SPLIT
    train_ds, val_test_ds = tf.keras.utils.image_dataset_from_directory(
        resolved_dataset_dir,
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=combined_val_test_split,
        subset='both',
        seed=SEED,
        shuffle=True,
    )

    val_test_batches = tf.data.experimental.cardinality(val_test_ds).numpy()
    if val_test_batches <= 1:
        raise ValueError(
            "Tập dữ liệu quá ít để tách riêng validation và test. "
            "Hãy bổ sung dữ liệu hoặc giảm BATCH_SIZE."
        )

    test_batches = max(1, int(round(val_test_batches * TEST_SPLIT / combined_val_test_split)))
    if test_batches >= val_test_batches:
        test_batches = val_test_batches - 1

    test_ds = val_test_ds.take(test_batches)
    val_ds = val_test_ds.skip(test_batches)
    class_names = train_ds.class_names

num_classes = len(class_names)

train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
val_ds = val_ds.apply(tf.data.experimental.ignore_errors())
test_ds = test_ds.apply(tf.data.experimental.ignore_errors())

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")

data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

model = keras.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.Rescaling(1.0 / 255),
    data_augmentation,
    layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax'),
])

model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
    )
]

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
)

plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

print("\n--- ĐÁNH GIÁ TRÊN TẬP VALIDATION ---")
val_loss, val_accuracy = model.evaluate(val_ds)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

print("\n--- ĐÁNH GIÁ TRÊN TẬP TEST ---")
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predict
y_pred = model.predict(test_ds)
y_pred_classes = np.argmax(y_pred, axis=1)

# True labels
y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in test_ds], axis=0)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true,
    y_pred_classes,
    average='weighted',
    zero_division=0,
)

print("\n--- TỔNG HỢP CHỈ SỐ TEST ---")
print(f"Accuracy : {test_accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# Classification report
print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_true, y_pred_classes, target_names=class_names, zero_division=0))


final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
acc_gap = final_train_acc - final_val_acc

print("\n--- PHÂN TÍCH NHANH ---")
if acc_gap > 0.1:
    print(
        "Mô hình có dấu hiệu overfitting: train_acc cao hơn val_acc đáng kể. "
        "Có thể tăng augmentation hoặc regularization."
    )
elif final_train_acc < 0.7 and final_val_acc < 0.7:
    print(
        "Mô hình có dấu hiệu underfitting: cả train_acc và val_acc đều thấp. "
        "Có thể tăng số epoch hoặc mở rộng mô hình vừa phải."
    )
else:
    print("Mô hình đang fit tương đối ổn, chưa thấy dấu hiệu overfit/underfit rõ rệt.")