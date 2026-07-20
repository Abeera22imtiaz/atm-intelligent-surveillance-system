# 1. Use an official, lightweight Python stable base image
FROM python:3.10-slim

# 2. Fix mirrors to use reliable global archives and install essential system dependencies
RUN sed -i 's/deb.debian.org/ftp.us.debian.org/g' /etc/apt/sources.list && \
    sed -i 's/security.debian.org/ftp.us.debian.org/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory inside the container for structured execution
WORKDIR /app

# 4. Copy the requirements file first to optimize Docker layer caching for dependencies
COPY requirements.txt .

# 5. Install Python dependencies without caching the installer files to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy core modular scripts directory (Contains config, tracker, trainer, etc.)
COPY src/ ./src

# 7. Copy root level configurations, execution entry scripts, and verified weights
COPY data.yaml .
COPY xVision_main.py .
COPY yolov8s.pt .

# 8. Define the default runtime command to execute the main tracking application via CLI flags
CMD ["python", "xVision_main.py", "--task", "track"]