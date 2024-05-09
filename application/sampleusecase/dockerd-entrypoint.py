import os
import shlex
import subprocess
import sys
from subprocess import CalledProcessError

from retrying import retry
from sagemaker_inference import model_server


def _retry_if_error(exception):
    """Determines if a retry should be attempted based on the exception type.

    Args:
        exception (Exception): The exception thrown by the retrying function.

    Returns:
        bool: True if a retry should be attempted, False otherwise.
    """
    return isinstance(exception, (CalledProcessError, OSError))


@retry(stop_max_delay=1000 * 50, retry_on_exception=_retry_if_error)
def _start_mms():
    """Starts the model server with configured environment variables.

    Sets the number of workers for the model server and starts it using a specified handler.
    """
    # Default number of workers per model is 1, configurable via environment variable
    os.environ["SAGEMAKER_MODEL_SERVER_WORKERS"] = "5"
    model_server.start_model_server(handler_service="/home/model-server/model_handler.py:handle")


def main():
    """Main function to handle script execution.

    Supports command-line arguments to control server operation. Defaults to keeping the Docker container alive.
    """
    if sys.argv[1] == "serve":
        _start_mms()
    else:
        subprocess.check_call(shlex.split(" ".join(sys.argv[1:])))

    # Prevent docker exit by keeping a non-terminating process running
    subprocess.call(["tail", "-f", "/dev/null"])


if __name__ == "__main__":
    main()
