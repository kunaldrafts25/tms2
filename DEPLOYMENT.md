# 🚀 TMS2 Streamlit Cloud Deployment Guide

## 📋 Overview

This guide explains how to deploy the TMS2 Traffic Management System dashboard to Streamlit Cloud.

## 🔧 Pre-Deployment Setup

### 1. Repository Configuration

The repository is already configured with the following deployment files:

- `requirements.txt` - Streamlit Cloud optimized dependencies
- `runtime.txt` - Python 3.11 specification
- `packages.txt` - System dependencies for OpenCV
- `.streamlit/config.toml` - Streamlit configuration
- `.python-version` - Python version specification

### 2. Main Application File

The main dashboard file is located at:
```
src/dashboard/quick_dashboard.py
```

## 🌐 Streamlit Cloud Deployment

### Step 1: Connect Repository

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository: `tms2`
5. Set the main file path: `src/dashboard/quick_dashboard.py`
6. Click "Deploy!"

### Step 2: Configuration

The app will automatically use:
- **Python Version**: 3.11 (from `runtime.txt`)
- **Dependencies**: Minimal set from `requirements.txt`
- **System Packages**: OpenCV dependencies from `packages.txt`

### Step 3: Fallback Mode

The dashboard automatically detects the cloud environment and runs in fallback mode:

- ✅ **Simulation Mode**: Uses realistic traffic simulation data
- ✅ **No Heavy ML**: TensorFlow and PyTorch are disabled
- ✅ **Lightweight**: Only essential packages are loaded
- ✅ **Full UI**: All dashboard features remain functional

## 🎯 Features Available in Cloud Deployment

### ✅ Working Features:
- 📊 Real-time traffic simulation dashboard
- 🚦 Traffic signal visualization with countdown timers
- 📈 Traffic analytics and charts
- 🎛️ 4-way intersection analysis (simulation mode)
- 📱 Responsive design for mobile devices
- 🌙 Dark theme optimized interface
- 📋 Session reporting and data export

### ⚠️ Simulation Mode Features:
- 🎭 Realistic traffic pattern simulation
- 🤖 AI decision simulation (no actual ML models)
- 📹 Video upload interface (processes with simulation)
- 🚨 Emergency mode simulation

### ❌ Disabled for Cloud:
- 🎥 Live camera feeds (requires local hardware)
- 🧠 Actual ML model inference (TensorFlow/PyTorch)
- 📊 Real-time video processing

## 🔍 Troubleshooting

### Common Issues:

1. **Dependency Errors**
   - The `requirements.txt` is optimized for cloud deployment
   - Heavy ML packages are commented out

2. **Import Errors**
   - The dashboard has robust fallback handling
   - Missing packages trigger simulation mode

3. **Performance Issues**
   - Cloud deployment uses lightweight simulation
   - No heavy video processing in cloud mode

### Logs to Check:
```
SUCCESS: Fallback systems imported successfully
WARNING: Model components not available: [expected in cloud]
INFO: Running in simulation mode due to cloud constraints
```

## 🎨 Customization

### Theme Configuration
Edit `.streamlit/config.toml` to customize:
- Primary colors
- Background colors
- Text colors

### Simulation Parameters
Edit `src/utils/model_fallback.py` to adjust:
- Traffic patterns
- Simulation realism
- AI confidence levels

## 📱 Access Your Deployed App

Once deployed, your app will be available at:
```
https://[your-app-name].streamlit.app
```

## 🔄 Updates

To update your deployed app:
1. Push changes to your GitHub repository
2. Streamlit Cloud will automatically redeploy
3. Changes appear within 1-2 minutes

## 🆘 Support

If you encounter issues:
1. Check the Streamlit Cloud logs
2. Verify all files are committed to GitHub
3. Ensure the main file path is correct: `src/dashboard/quick_dashboard.py`

---

**Note**: The cloud deployment runs in simulation mode for demonstration purposes. For full functionality with real cameras and ML models, run locally with the complete requirements.
