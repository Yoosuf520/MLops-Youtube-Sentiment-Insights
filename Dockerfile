# base image
FROM python:3.10-slim-bookworm

# install uv
RUN pip install uv

# workdir
WORKDIR /app

# copy
COPY . /app

# install dependencies using uv
RUN uv pip install -r requirements.txt --system

# port
EXPOSE 8000

# command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]