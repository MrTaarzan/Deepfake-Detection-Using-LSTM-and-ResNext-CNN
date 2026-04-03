# DeepFake Detection System

This project implements a DeepFake detection system using ResNext CNN and LSTM networks, deployed through a modern web interface.

## Project Overview

The system uses a combination of:
- ResNext CNN for spatial feature extraction
- LSTM for temporal sequence analysis
- Modern web interface for easy interaction with the model

## Features

- Real-time deepfake detection from video uploads
- Support for multiple video formats
- Interactive web interface with real-time feedback
- Detailed analysis reports
- High accuracy detection using state-of-the-art deep learning models

## Project Structure

```
src/
├── models/         # Neural network model implementations
├── data/          # Data handling and preprocessing
├── utils/         # Utility functions and helpers
├── web/           # Web application
    ├── static/    # Static files (CSS, JS, images)
    └── templates/ # HTML templates
tests/             # Unit tests
docs/              # Documentation
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory and add necessary configurations.

## Usage

1. Start the web server:
```bash
python src/web/app.py
```

2. Open a web browser and navigate to `http://localhost:5000`

## Model Architecture

- **Feature Extraction**: ResNext-50 CNN
- **Temporal Analysis**: LSTM network
- **Classification**: Fully connected layers with sigmoid activation

## Development

To run tests:
```bash
pytest tests/
```

## License

MIT License

## Contributors

Abhinav G Pankaj  
Anusha K  
Drisya V S  
Gaanasri M S   