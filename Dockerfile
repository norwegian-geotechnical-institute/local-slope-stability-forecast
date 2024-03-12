FROM mcr.microsoft.com/azure-functions/python:4-python3.11 as requirements-stage

# Install Poetry
RUN python -m pip install poetry==1.6.1

# Create requirements file for package dependencies
COPY poetry.lock /
COPY pyproject.toml /
RUN poetry export -f requirements.txt -o requirements.txt


FROM mcr.microsoft.com/azure-functions/python:4-python3.11

# Install dependencies
COPY --from=requirements-stage /requirements.txt /
RUN pip install -r requirements.txt --no-deps

# Copy script to run
COPY fos_forecast.py /
COPY .env /
COPY tmp_models /models
COPY tmp_params /params
