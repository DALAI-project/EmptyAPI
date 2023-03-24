FROM python:3.7

# Set the current working directory to /app
WORKDIR /app

# Copy requirements.txt
COPY ./requirements.txt /app/requirements.txt

# Install requirements
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy model files
COPY ./model /app/model

# Copy app file
COPY ./api.py /app/

# Expose port
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
