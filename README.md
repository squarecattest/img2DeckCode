Master Duel Deck Recognizer
===
## 🚀 Overview

This tool automatically converts Yu-Gi-Oh! Master Duel deck images into structured deck information and a Neuron deck code. 
You can upload deck photos and receive a complete deck list, including card names, quantities and Neuron deck code.

The application can be deployed locally as a web service, providing a browser-based interface for image upload and result viewing. 
It also exposes an API that allows external clients, such as Discord bots, to submit images and retrieve deck data programmatically.

## 🛠️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/BernieTv/ElevenLabs-Clone.git
```

### 2️⃣ Navigate to Project Directory

```bash
cd img2DeckCode
```

### 3️⃣ Install Python 🐍

Ensure Python 3.10 or above is installed. If not, download it:  
👉 [Download Python](https://www.python.org/downloads/)

## 📦 Install Dependencies

It is recommended to run this project inside a Python virtual environment (venv).
```bash
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## ▶️ How to Run

```bash
python ./web/main.py
```

```bash
cd web
ngrok http 8000
```

Use the generated ngrok URL to access the web interface or call the API remotely.
