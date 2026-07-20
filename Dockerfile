# 1. Use an official Python image built on Ubuntu (busted/bookworm standard) which has very stable network mirrors
FROM python:3.10-bookworm

# 2. Install essential system dependencies required for OpenCV directly
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory inside the container for structured execution
WORKDIR /app

# 4. Copy the requirements file first to optimize Docker layer caching for dependencies
COPY requirements.txt .


# 5. Upgrade pip and install Python dependencies safely
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

# 6. Copy core modular scripts directory (Contains config, tracker, trainer, etc.)
COPY src/ ./src

# 7. Copy root level configurations, execution entry scripts, and verified weights
COPY data.yaml .
COPY xVision_main.py .
COPY yolov8s.pt .

# 8. Define the default runtime command to execute the main tracking application via CLI flags
CMD ["python", "xVision_main.py", "--task", "track"]