# 🚀 YOLOv8 Projects: Gap Detection, Container Filling, Number Detection

This repository contains 3 advanced real-time object detection projects using **YOLOv8** developed during my internship at **Clustor Computing, Nagpur**.

## 📁 Projects Included

### 1️⃣ GAP DETECTION
- **Objective:** Detects the physical gap between two objects (e.g. a book and a bottle) on a table in real-time.
- **How it works:** A YOLOv8 model is trained on custom images where the gap is visible or absent. The model predicts bounding boxes, and logic is applied to measure the space between detected objects.

### 2️⃣ CONTAINER FILLING DETECTION
- **Objective:** Classify container fill levels into 3 categories: `Filling`, `Filled`, and `Overfilled`.
- **How it works:** Dataset images of containers are labeled with their respective states. YOLOv8 is used to classify and detect the fill level based on the height of the content relative to a marked line on the container.

### 3️⃣ NUMBER DETECTION
- **Objective:** Detect and identify numbers in an image, e.g. digits written or printed on objects.
- **How it works:** A YOLOv8 model trained on number datasets to detect and classify numbers `0-9`.

---

## 🧠 Technologies Used

| Tool / Library | Purpose |
|----------------|---------|
| [Python](https://www.python.org/) | Core programming language |
| [YOLOv8](https://github.com/ultralytics/ultralytics) | Real-time object detection |
| [OpenCV](https://opencv.org/) | Image processing, camera access |
| [LabelImg](https://github.com/tzutalin/labelImg) | Image annotation for training datasets |
| [Tkinter](https://docs.python.org/3/library/tkinter.html) | GUI (used in some sub-projects) |
| [PyTorch](https://pytorch.org/) | Deep learning framework |
| [Ultralytics CLI](https://docs.ultralytics.com/) | YOLOv8 model training and inference |
| [Real-ESRGAN (optional)](https://github.com/xinntao/Real-ESRGAN) | Upscaling CCTV footage (in extended project) |

---


## 🛠️ How to Run

1. **Clone the repository:**

```bash
git clone https://github.com/Solomon-Alexander1/YOLOv8-Projects-Gap-Detection-Container-Filling-Number-Detection.git
cd YOLOv8-Projects-Gap-Detection-Container-Filling-Number-Detection

Install dependencies:
pip install -r requirements.txt
pip install ultralytics opencv-python torch torchvision

Run any script:
python "Gap Detection Main.py"
python "Number Detection Main.py"
python "Container_Filling.py"
```
📺 Demo Videos

- 🔗 [Gap Detection Demo](https://github.com/Solomon-Alexander1/YOLOv8-Projects-Gap-Detection-Container-Filling-Number-Detection-/blob/master/Gap_No_Gap_Detection_Video.mp4)

- 🔗 [Container Filling + Number Detection Demo](https://github.com/Solomon-Alexander1/YOLOv8-Projects-Gap-Detection-Container-Filling-Number-Detection-/blob/master/Container_Filling_Video.mp4)

- 🔗 [Number Detection Demo](https://github.com/Solomon-Alexander1/YOLOv8-Projects-Gap-Detection-Container-Filling-Number-Detection-/blob/master/Number_Detection_Video.mkv)


📦 Model Weights
Pre-trained weights used:
yolov8n.pt – Lightweight model (for real-time)

yolov8s.pt – More accurate small model
You can retrain on your own dataset using the provided train.py script.


📌 Future Enhancements
📈 Add GUI for easier testing

🧠 Integrate CCTV footage and real-time alert system

🌐 Host on web dashboard with monitoring


### 🙋‍♂️ Author

**Solomon Goodwin Alexander**  
📍 Nagpur, India  
🔗 [GitHub Profile](https://github.com/Solomon-Alexander1)  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/solomon-alexander-184733170/)

