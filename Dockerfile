# 1. Base image
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements first to leverage Docker cache
COPY requirements.txt .

# 4. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the app files
COPY . .

# 6. Expose port
EXPOSE 5000

RUN mkdir -p /tmp/yf_cache_disabled && chmod -R 777 /tmp/yf_cache_disabled /app/data

# 7. Use Gunicorn to serve Flask for production
#    - Bind to 0.0.0.0 so external connections can reach the container
#    - 4 workers is a decent default for CPU-bound apps
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
