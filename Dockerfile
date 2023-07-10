
FROM python:3

# Set working directory
WORKDIR /workspace

# Copy requirements.txt and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Copy the rest of the files
COPY . .

# Run bash by default
CMD ["python3", "main.py"] 