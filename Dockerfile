FROM ubuntu:22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and build dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Install Python dependencies
COPY acm/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set Python path to find the 'acm' package
ENV PYTHONPATH=/app

# Expose the API port
EXPOSE 8000

# Run the FastAPI application with uvicorn
CMD ["python3", "-m", "uvicorn", "acm.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
