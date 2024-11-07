
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libhdf5-dev \
    git \
    cron \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app
# Install Python dependencies using pip
COPY requirements.txt .
RUN pip install --upgrade pip  # Ensure pip is up to date
RUN pip install -r requirements.txt

# Set up the working directory

# Copy the project files
COPY . /app

RUN echo "*/1 * * * * /usr/bin/python3 /app/monitor_file.py >> /var/log/cron.log 2>&1" > /etc/cron.d/monitor-cron
RUN chmod 0644 /etc/cron.d/monitor-cron

RUN crontab /etc/cron.d/monitor-cron
RUN touch /var/log/cron.log

EXPOSE 5000

# Set the environment variable to disable virtual environments
ENV MLFLOW_MLFLOW_BACKEND_CONFIG '{"disable_env_creation": true}'

# Run the MLflow project inside the Docker container with no virtual env
#CMD /bin/sh -c "cron && mlflow run . --env-manager=local"
CMD ["gunicorn", "-w", "3", "-b", "0.0.0.0:5000", "app:app"]
# CMD ["python","app.py"]





