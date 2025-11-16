![GesturiX Logo](frontend/assets/images/GesturiX-readme.png)

Gesturix is a modern sign language learning companion that helps you learn, translate, and communicate using interactive tools and real-time sign recognition. Built with love and purpose to make learning sign language accessible for everyone.

## Features

- Real time language translation
- Learn through videos and tutorials
- Customizable profile
- etc. (future updates)

## Tech Stack

| Layer            | Technology                                    | Purpose / Usage                                                 |
| ---------------- | --------------------------------------------- | --------------------------------------------------------------- |
| Frontend         | [React Native](https://reactnative.dev/)      | Mobile app framework for iOS and Android                        |
|                  | [NativeWind](https://www.nativewind.dev/)     | Tailwind-style utility classes for styling React Native apps    |
|                  | [Expo](https://expo.dev/)                     | Development environment, build & deployment tools               |
|                  | [TypeScript](https://www.typescriptlang.org/) | Type safety and better code structure                           |
| Backend          | [FastAPI](https://fastapi.tiangolo.com/)      | Python framework for building APIs                              |
|                  | [Python](https://www.python.org/)             | Core language for backend logic and ML integration              |
| Machine Learning | [Mediapipe](https://mediapipe.dev/)           | Real-time gesture detection & tracking                          |
|                  | [PyTorch](https://pytorch.org/)               | Deep learning framework for model inference                     |
| Database         | [Supabase](https://supabase.com/)             | Backend-as-a-Service for authentication and database management |

## Prerequisites

| Category         | Tool / Technology         | Purpose / Notes / Installation                                   |
| ---------------- | ------------------------- | ---------------------------------------------------------------- |
| System           | Windows 10+ / Android 15+ | Windows 10+ for development, Android 15+ for testing on devices. |
| Node.js          | Node.js                   | JavaScript runtime for React Native & Expo. v18+ recommended.    |
| Package Manager  | npm / yarn                | npm comes with Node.js; yarn optional.                           |
| Expo CLI         | Expo CLI                  | For running and building React Native apps.                      |
| Python           | Python 3.11+              | For backend and ML integration.                                  |
| pip              | pip                       | Python package manager.                                          |
| Python Env       | venv / virtualenv         | Isolated Python environment for dependencies.                    |
| Android Emulator | Android Studio            | For testing Android builds locally.                              |

## Cloning and Initialization

**Step 1:** Clone the repository

```bash
  git clone https://github.com/GoldenSnek/GesturiX
```

**Step 2:** Navigate into the frontend folder

```bash
  cd GesturiX/frontend
```

**Step 3:** Install dependencies (frontend)

```bash
  npm install
```

**Step 4:** Navigate into the backend folder

```bash
  cd GesturiX/backend
```

**Step 5:** Create venv folder

```bash
  py -3.11 -m venv venv
```

**Step 6:** Activate venv

```bash
  venv\Scripts\Activate.ps1
```

**Step 7:** Install dependencies (backend)

```bash
  pip install -r requirements.txt
```

## Run Locally

**Step 1:** Navigate into the backend folder

```bash
  cd GesturiX/backend
```

**Step 2:** Activate backend server

```bash
  py main.py
```

**Step 3:** Download the latest GesturiX build as of _Nov 11, 2025_ -> https://expo.dev/accounts/gesturix/projects/frontend/builds/2da9af85-7a0d-4e3e-9846-c45b2641ab2f

**Step 4:** Navigate into the frontend folder

```bash
  cd GesturiX/frontend
```

**Step 5:** Run expo

```bash
  npx expo start --dev-client
```

**Step 6:** Using the recently downloaded GesturiX.apk, scan the QR code or manually type the URL found in the frontend terminal.

## Developer Profiles

- **Jesnar T. Tindogan** << Project Manager | Backend Developer >>

  [![github](https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Jasner13)

- **John Michael A. Nave** << Lead Developer | DevOps Engineer >>

  [![github](https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/GoldenSnek)

- **John Michael B. Villamor** << ML Engineer | Quality Assurance>>

  [![github](https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Villamormike)
