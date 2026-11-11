<div align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcXRiaHhhcGQ0YzkxNWxhdnV0djR6NnYyNzRxN2E5aGhzdWRscDdtcyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/L1R1tvI9svkVDsHqaV/giphy.gif" width="150" alt="Cute wink"/>
  <h1 align="center">🌸 DrowsyGuard AI 🌸</h1>

  <p align="center">
    <strong>A real-time, aesthetic & smart Drowsiness Detection System.</strong>
    <br/>
    Built with ❤️ using OpenCV, Dlib, and Streamlit.
  </p>
</div>

---

## ✨ Features

- **Real-Time Detection:** Live webcam feed with facial landmark overlay.
- **Eye Aspect Ratio (EAR):** Mathematically tracks when your eyes close.
- **Deep Learning Ready:** CNN models for drowsiness and yawning detection.
- **Aesthetic UI:** A completely feminine, elegant, and soft UI built on top of Streamlit.
- **Instant Alerts:** Visual and sound alerts (MacOS) if you fall asleep!

## 🎀 Tech Stack

- **Python & OpenCV:** For computer vision and processing the webcam feed.
- **Dlib:** For accurate 68-point facial landmark detection.
- **Keras & TensorFlow:** For training custom CNN models (eye/yawn state).
- **Streamlit:** Fast, elegant, and interactive frontend.

## 🚀 How to Run

1. **Activate the Virtual Environment:**
   ```bash
   source venv/bin/activate
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the Beautiful App!**
   ```bash
   streamlit run app.py
   ```

## 🧠 Training the Models

Want to train the CNN models yourself? It's super easy! 

All training logic is in `train_model.py`.
```bash
python train_model.py
```
This script will split the dataset, build a CNN, use data augmentation, and save the best models automatically in the `./models/` folder.

---

<p align="center">
  <i>Sleep is important, but not while driving! Stay safe. 🩷</i>
</p>