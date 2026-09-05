# Use an official lightweight Python image with GIS system dependencies
FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

# Install system dependencies required for GDAL, rasterio, and OpenCV/Folium
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin \
    libgdal-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose default Streamlit port
EXPOSE 8080

# Launch Streamlit bound to Google Cloud Run's required port (8080)
CMD ["streamlit", "run", "MadaHydroLab.py", "--server.port=8080", "--server.address=0.0.0.0"]
