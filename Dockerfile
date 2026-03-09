FROM python:3.10-slim

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Set environment variables for Hugging Face Spaces
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    TRANSFORMERS_CACHE=/tmp/cache \
    HF_HOME=/tmp/cache \
    NUMBA_CACHE_DIR=/tmp/cache \
    MPLCONFIGDIR=/tmp/cache \
    PORT=7860

# Install system dependencies required for OpenCV and Audio processing
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create cache directory and grant permissions to the user
RUN mkdir -p /tmp/cache && chown -R user:user /tmp/cache

# Switch to the "user" user
USER user

# Set the working directory
WORKDIR $HOME/app

# Copy the requirements file and install dependencies
COPY --chown=user:user backend/requirements.txt $HOME/app/
# Add gunicorn to requirements inline if it doesn't exist
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# Copy the rest of the application code
COPY --chown=user:user backend/ $HOME/app/backend/
COPY --chown=user:user frontend/ $HOME/app/frontend/
COPY --chown=user:user docs/ $HOME/app/docs/

# Expose the port
EXPOSE 7860

# Start the application using Gunicorn (1 worker + 4 threads to share ML model memory)
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "--workers", "1", "--threads", "4", "backend.app:app"]
