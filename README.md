# 🧠 AI Skin Disease Detection System

A full‑stack deep learning web application that detects skin diseases from images using a CNN model, FastAPI backend, and Next.js frontend.

---

## 🚀 Live Architecture

```
Next.js Frontend → FastAPI Backend → PyTorch CNN → Prediction
                                   ↓
                           Hugging Face Model Hub
```

✅ Model hosted externally (Hugging Face)
✅ Clean and lightweight repository
✅ Production‑ready structure

---

## ✨ Features

* 🧠 CNN‑based skin disease classification
* ⚡ FastAPI high‑performance backend
* 🎨 Modern Next.js frontend UI
* 📤 Image upload with preview
* 📊 Confidence score output
* ☁️ Model auto‑download from Hugging Face
* 🔌 CORS enabled for frontend integration

---

## 🛠️ Tech Stack

**Frontend**

* Next.js (JavaScript)
* Tailwind CSS

**Backend**

* FastAPI
* PyTorch
* Pillow

**Model Hosting**

* Hugging Face Model Hub

---

## 📂 Project Structure

```
skin-disease-app/
├── backend/
│   ├── main.py
│   ├── model.py
│   ├── requirements.txt
│   └── (model auto-downloads)
│
└── frontend/
    ├── app/
    └── package.json
```

---

## ⚙️ Backend Setup (FastAPI)

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

API docs:

```
http://localhost:8000/docs
```

---

## 💻 Frontend Setup (Next.js)

```bash
cd frontend
npm install
npm run dev
```

