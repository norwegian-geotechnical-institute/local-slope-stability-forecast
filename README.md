# Running locally
Python 3.11 or higher is required to run. Dependencies and virtual environment is managed using
[Poetry](https://python-poetry.org/.)

Install dependencies into virtual environment
```bash
poetry install
```
Create a `.env` file by copying from `.env.template`:
```
cat .env.template > .env
```
Change any parameters in .env to configure your specific run. Note: the `.env` file is intended for local setup and 
might include secrets. It should not be checked into git. Also, make sure that no secrets are added to the 
`.env.template` file.

Run
```bash
poetry run python .\fos_forecast.py
```

Optionally, specify the time as a command line argument
```bash
poetry run python .\fos_forecast.py --timestamp '2023-12-15T00:00:00+01'
```


## Testing requirements file for cloud environment
To test whether the script is runnable in the cloud environment, try to run
the script in an azure functions docker container.

Build image
```
docker build -t tmpslopes .
```
Run image in interactive mode
```
docker run -it tmpslopes bash
```
Inside image, run script
```
python fos_forecast.py
```
Note that this container does not have access to blob storage or the P drive. It uses
local temporary input files downloaded from blob storage in a previous local run.

# Manually triggering azure timed function
Get the function url and _master key from the Azure portal "App Keys" pane for the azure function, then run:
```
curl -kv -XPOST "<function_url>/admin/functions/TimedTrigger" -H "x-functions-key: <master_key>" -H "Content-Type: application/json" -d '{"input": ""}'
```
A 202 response denotes a successful request.
