import logging
import azure.functions as func
from datetime import datetime, timezone
from API_trial_LSTM_RF import run

app = func.FunctionApp()

# Note that HTTP triggers have a 230-second limit
# Ref.: https://learn.microsoft.com/en-us/azure/azure-functions/functions-scale#timeout
@app.function_name(name="HttpTrigger")
@app.route(route="http_trigger")
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    run(datetime.now(timezone.utc))
    logging.info('Python HTTP trigger function done processing request.')
    return func.HttpResponse(f"Run completed successfully!")


@app.function_name(name="TimedTrigger")
@app.schedule(schedule="0 0 * * *",
              arg_name="mytimer",
              run_on_startup=False)
def test_function(mytimer: func.TimerRequest) -> None:
    logging.info(f"Python timer trigger function triggered.")
    run(datetime.now(timezone.utc))
    logging.info(f"Python timed trigger done processing request.")

