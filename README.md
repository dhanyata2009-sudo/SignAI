# SignAI – AI-Based Sign Language Recognition Website

## 📌 Project Overview

SignAI is an AI-based web application designed to recognize hand gestures and help bridge the communication gap between people who use sign language and those who may not understand it.

The system uses real-time hand tracking and landmark detection to identify supported hand gestures through a computer's webcam. It provides an interactive platform for recognizing signs and managing gesture training data.

The project aims to make sign language recognition more accessible through a simple, browser-based interface.

## 🎯 Objectives

- Develop a browser-based sign language recognition system.
- Detect and track hand movements in real time.
- Recognize supported sign language gestures using hand landmark information.
- Provide an interactive interface for gesture recognition.
- Support adding and updating custom gesture training data.

## ✨ Key Features

- **Real-Time Hand Detection:** Detects hands using the device camera.
- **Hand Landmark Tracking:** Identifies key points on the hand for gesture analysis.
- **Gesture Recognition:** Recognizes supported hand gestures.
- **Training Management:** Allows authorized users to add new gesture training samples.
- **Update Existing Signs:** Supports updating existing gesture data without creating duplicate entries.
- **Browser-Based Application:** Runs directly in a web browser.
- **Local Data Storage:** Uses browser LocalStorage to maintain custom training data.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| HTML5 | Webpage structure |
| CSS3 | Styling and user interface |
| JavaScript | Application logic |
| Vite | Development environment and build tool |
| MediaPipe Tasks Vision | Hand detection and landmark tracking |
| Hand Landmarker | Detecting hand landmarks |
| LocalStorage | Storing custom gesture training data |

## ⚙️ How It Works

1. **Camera Initialization:** The application requests access to the device camera.
2. **Hand Detection:** MediaPipe detects the hand in the camera feed.
3. **Landmark Extraction:** The system identifies hand landmarks.
4. **Gesture Processing:** Extracted hand information is processed to identify supported gestures.
5. **Recognition Output:** The application displays the recognized sign.
6. **Training Management:** Authorized users can add or update gesture training samples.

## 🚀 Installation and Setup

Follow these steps to run SignAI locally.

**1. Clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/SignAI.git
```

**2. Navigate to the project folder**

```bash
cd SignAI
```

**3. Install dependencies**

```bash
npm install
```

**4. Start the development server**

```bash
npm run dev
```

**5. Open the application**

Open the local URL displayed in your terminal, usually:

```text
http://localhost:5173
```

**Note:** Allow camera permissions when prompted. Ensure the required Hand Landmarker model file is available at the configured model path.

## 📂 Project Structure

```text
SignAI/
│
├── public/
│   └── models/
│       └── hand_landmarker.task
│
├── src/
│   ├── JavaScript files
│   ├── CSS files
│   └── Application modules
│
├── index.html
├── package.json
├── package-lock.json
├── vite.config.js
├── .gitignore
└── README.md
```

*Note: The structure above is illustrative. Adjust it to match the actual project folders.*

## 🔮 Future Scope

- Expand the gesture recognition vocabulary.
- Improve recognition accuracy under different lighting conditions.
- Support additional sign language gestures.
- Explore sentence-level sign recognition.
- Add speech and text conversion capabilities.

## 👩‍💻 Project Information

**Project Name:** SignAI – AI-Based Sign Language Recognition Website

**Project Type:** Academic / Personal Project

**Domain:** Artificial Intelligence and Web Development

**Developed By:** Dhanyata Kate

## 📌 Important Note

This project is designed to recognize its supported set of gestures. It is a prototype and should not be considered a complete sign language translation system.
