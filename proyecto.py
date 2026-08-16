from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import numpy as np
import cv2
import mediapipe as mp


DATA_DIR = Path("../asl_alphabet_train/")


def process_single_image(img_path_str, class_name):
    """Procesa una imagen y extrae los landmarks de la mano."""

    # Configuración de MediaPipe
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

    model_path = "hand_landmarker.task"

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Crear el detector
    with HandLandmarker.create_from_options(options) as hands:

        img = cv2.imread(img_path_str)

        if img is None:
            return None, None

        # OpenCV usa BGR -> MediaPipe necesita RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convertir imagen de OpenCV a MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img_rgb
        )

        # Detectar manos
        results = hands.detect(mp_image)

        if results.hand_landmarks:

            data_aux = []
            x_ = []
            y_ = []

            # Primera mano detectada
            hand_landmarks = results.hand_landmarks[0]

            for lm in hand_landmarks:
                x_.append(lm.x)
                y_.append(lm.y)

            # Normalizar coordenadas
            min_x = min(x_)
            min_y = min(y_)

            for lm in hand_landmarks:
                data_aux.append(lm.x - min_x)
                data_aux.append(lm.y - min_y)

            return data_aux, class_name

    return None, None


def main():

    # Buscar todas las imágenes y utilizar el nombre
    # de la carpeta como etiqueta
    image_tasks = []

    for class_dir in DATA_DIR.iterdir():

        if class_dir.is_dir():

            class_name = class_dir.name

            # JPG
            for img_path in class_dir.glob("*.jpg"):
                image_tasks.append((str(img_path), class_name))

            # PNG
            for img_path in class_dir.glob("*.png"):
                image_tasks.append((str(img_path), class_name))

    print(
        f"Found {len(image_tasks)} images across subdirectories."
    )

    data = []
    labels = []

    # Procesar imágenes en paralelo
    with ProcessPoolExecutor() as executor:

        futures = [
            executor.submit(
                process_single_image,
                path,
                label
            )
            for path, label in image_tasks
        ]

        for idx, future in enumerate(as_completed(futures)):

            try:
                features, label = future.result()

                if features is not None:
                    data.append(features)
                    labels.append(label)

            except Exception as e:
                print(f"Error procesando imagen: {e}")

            if (idx + 1) % 500 == 0:
                print(
                    f"Processed {idx + 1}/{len(image_tasks)} images..."
                )

    # Guardar los datos
    with open("data.pickle", "wb") as f:

        pickle.dump(
            {
                "data": np.array(data),
                "labels": np.array(labels)
            },
            f
        )

    print(
        f"Finished extracting features! "
        f"Successful images: {len(data)}"
    )


if __name__ == "__main__":
    main()

################################################################
import pickle
from collections import Counter

# Load your pickled dataset
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

labels = data_dict['labels']
counts = Counter(labels)

print("--- Sample Count Per Class ---")
for class_name, count in sorted(counts.items()):
    print(f"Class '{class_name}': {count} samples")

# Identify classes with fewer than 2 samples
low_count_classes = [cls for cls, count in counts.items() if count < 2]
print("\nClasses with less than 2 samples:", low_count_classes)
################################################################################
import pickle
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load data
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data_raw = data_dict['data']
labels_raw = data_dict['labels']

# 2. Filter out classes with fewer than 10 samples (removes 'NOTHING')
counts = Counter(labels_raw)
MIN_SAMPLES = 10

filtered_data = []
filtered_labels = []

for features, label in zip(data_raw, labels_raw):
    # Ensure features have expected landmark length (42 values for x,y coordinates)
    if counts[label] >= MIN_SAMPLES and len(features) == 42:
        filtered_data.append(features)
        filtered_labels.append(label)

X = np.array(filtered_data, dtype=np.float32)
y = np.array(filtered_labels)

print(f"Total samples kept: {len(y)}")
print(f"Classes included: {len(set(y))}")
print(f"Excluded classes: {[cls for cls, count in counts.items() if count < MIN_SAMPLES]}")

# 3. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    shuffle=True, 
    stratify=y, 
    random_state=42
)

# 4. Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 5. Evaluate and Save
y_predict = model.predict(X_test)
score = accuracy_score(y_test, y_predict)
print(f"\nModel Accuracy: {score * 100:.2f}%")

with open('asl_model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Saved model as asl_model.p!")


############################################################################################################
import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./asl_model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = prediction[0]

            cv2.putText(frame, predicted_character, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    cv2.imshow('ASL Alphabet Classifier', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

######################################################################################################
import pickle
import cv2
import mediapipe as mp
import numpy as np

# --------------------------------------------------
# Cargar modelo de clasificación
# --------------------------------------------------

model_dict = pickle.load(open('./asl_model.p', 'rb'))
model = model_dict['model']


# --------------------------------------------------
# Configurar cámara
# --------------------------------------------------

cap = cv2.VideoCapture(0)


# --------------------------------------------------
# Configurar MediaPipe Hand Landmarker
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# Crear detector
with HandLandmarker.create_from_options(options) as hands:

    frame_timestamp_ms = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        H, W, _ = frame.shape

        # OpenCV BGR -> RGB
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convertir a MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # Timestamp obligatorio para VIDEO
        frame_timestamp_ms += 33

        # Detectar manos
        results = hands.detect_for_video(
            mp_image,
            frame_timestamp_ms
        )

        # --------------------------------------------------
        # Procesar landmarks
        # --------------------------------------------------

        if results.hand_landmarks:

            for hand_landmarks in results.hand_landmarks:

                data_aux = []

                x_ = [lm.x for lm in hand_landmarks]
                y_ = [lm.y for lm in hand_landmarks]

                for lm in hand_landmarks:

                    data_aux.append(
                        lm.x - min(x_)
                    )

                    data_aux.append(
                        lm.y - min(y_)
                    )

                # --------------------------------------------------
                # Predicción
                # --------------------------------------------------

                prediction = model.predict(
                    [np.asarray(data_aux)]
                )

                predicted_character = prediction[0]

                cv2.putText(
                    frame,
                    predicted_character,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 255, 0),
                    3
                )

        # Mostrar cámara
        cv2.imshow(
            'ASL-Vision-12',
            frame
        )

        # Presionar Q para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# --------------------------------------------------
# Liberar recursos
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()



