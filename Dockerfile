FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port 5000 (or 7860 for HF Spaces)
EXPOSE 7860

# Run with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "app:app"]
