The folder named datasets contains one or many datasets.
You can download the folder [here](https://drive.google.com/drive/folders/1uSM2oOa19bDkkcqdDMWQmW9m7-jhoiLd?usp=sharing).

The structure as below:
datasets
├── BRED_dataset
│   ├── game
│   │   └── subject_0 - subject_27
│   │        ├── Anger
│   │        ├── Disgust
│   │        ├── Fear
│   │        ├── Happiness
│   │        ├── Sadness
│   │        └── Surprise
│   ├── pre-game
│   │   └── subject_0 - subject_27
│   │        ├── Anger
│   │        ├── Disgust
│   │        ├── Fear
│   │        ├── Happiness
│   │        ├── Sadness
│   │        └── Surprise
│   ├── annotations.csv
│   ├── face_skeleton.py
│   └── README.md
├── GEMEP
│   ├── crop
│   │   └── {num}{emotion}_GEMEP
│   │        └── frame*.jpg
│   ├── frame
│   │   └── {num}{emotion}_GEMEP
│   │        └── frame*.jpg
│   ├── frame_body_json
│   │   └── {num}{emotion}_GEMEP
│   │        └── {num}_keypoints.json
│   ├── all_body_label.csv
│   ├── all_face_label.csv
│   ├── body_test_label.csv
│   ├── body_train_label.csv
│   ├── face_test_label.csv
│   ├── face_train_label.csv
│   └── gemep_crop.py
├── rafdb
│   ├── EmoLabel
│   │   └── list_patition_label.txt
│   └── Image
│        └── aligned
│             ├── train_{num}_aligned.jpg
│             └── test_{num}_aligned.jpg
├── rafdb_openpose
│   ├── Image
│   │   ├── train_00001-train_12270
│   │   │    └── train_*_keypoints.json
│   │   └── v
│   │   │    └── frame*.jpg
│   ├── label.csv
│   ├── test_label.csv
│   └── train_label.csv
└── readme.txt
