import traceback
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
import requests
import os
import time
import uuid
import logging
import json
from tabulate import tabulate
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()


# Function to check if the app is running inside Docker
def is_running_in_docker():
    """Check if the app is running inside Docker by checking for specific environment variables or Docker files."""
    return os.path.exists("/.dockerenv") or os.getenv("DOCKER") == "true"


# Logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("watsonx-openai-api")

# Mapping of valid regions
valid_regions = ["us-south", "eu-gb", "jp-tok", "eu-de"]

# Get region from env variable
region = os.getenv("WATSONX_REGION")

api_version = os.getenv("WATSONX_VERSION") or "2025-02-06"
on_prem = os.getenv("WATSONX_ON_PREM")
cpd_url = os.getenv("CPD_URL")

# Handle behavior based on environment (Docker vs. Interactive mode)
if not region or region not in valid_regions:
    if is_running_in_docker():
        # In Docker, raise an error if WATSONX_REGION is missing or invalid
        logger.error(
            f"WATSONX_REGION key is not set or invalid. Supported regions are: {', '.join(valid_regions)}."
        )
        raise SystemExit(
            f"WATSONX_REGION is required. Supported regions are: {', '.join(valid_regions)}."
        )
    else:
        # In interactive mode, prompt the user for the region
        print("Please select a region from the following options:")
        for idx, reg in enumerate(valid_regions, start=1):
            print(f"{idx}. {reg}")

        choice = input("Enter the number corresponding to your region: ")

        try:
            region = valid_regions[int(choice) - 1]
        except (IndexError, ValueError):
            raise ValueError(
                "Invalid region selection. Please restart and select a valid option."
            )

# Mapping of each region to URL
WATSONX_URLS = {
    "us-south": "https://us-south.ml.cloud.ibm.com",
    "eu-gb": "https://eu-gb.ml.cloud.ibm.com",
    "jp-tok": "https://jp-tok.ml.cloud.ibm.com",
    "eu-de": "https://eu-de.ml.cloud.ibm.com",
}

# IBM Cloud IAM URL for fetching the token
IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"

# Token url for ON_PREM
if on_prem == "1":
    CPD_AUTH_URL = f"{cpd_url}/icp4d-api/v1/authorize"
    USERNAME = os.getenv("USERNAME")
    WATSONX_MODELS_URL = f"{cpd_url}/ml/v1/foundation_model_specs"
    WATSONX_URL = f"{cpd_url}/ml/v1/text/generation?version={api_version}"
    WATSONX_URL_CHAT = f"{cpd_url}/ml/v1/text/chat?version={api_version}"
    WATSONX_CUSTOM_MODELS_URL = (
        f"{cpd_url}/ml/v4/custom_foundation_models?version=2024-05-01"
    )
else:
    WATSONX_MODELS_URL = (
        f"{WATSONX_URLS.get(region)}/ml/v1/foundation_model_specs?version={api_version}"
    )
    # Construct Watsonx URLs with the version parameter
    WATSONX_URL = (
        f"{WATSONX_URLS.get(region)}/ml/v1/text/generation?version={api_version}"
    )
    WATSONX_URL_CHAT = (
        f"{WATSONX_URLS.get(region)}/ml/v1/text/chat?version={api_version}"
    )

# Load IBM API key, Watsonx URL, and Project ID from environment variables
IBM_API_KEY = os.getenv("WATSONX_IAM_APIKEY")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")

if not IBM_API_KEY:
    logger.error(
        "IBM API key is not set. Please set the WATSONX_IAM_APIKEY environment variable."
    )
    raise SystemExit("IBM API key is required.")

if not PROJECT_ID:
    logger.error(
        "Watsonx.ai project ID is not set. Please set the WATSONX_PROJECT_ID environment variable."
    )
    raise SystemExit("Watsonx.ai project ID is required.")

# Global variables to cache the IAM token and expiration time
cached_token = None
token_expiration = 0


# Function to fetch the IAM token (cloud)
def get_iam_token():
    global cached_token, token_expiration
    current_time = time.time()

    if cached_token and current_time < token_expiration:
        logger.debug("Using cached IAM token.")
        return cached_token

    logger.debug("Fetching new IAM token from IBM Cloud...")

    try:
        response = requests.post(
            IAM_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": IBM_API_KEY,
            },
        )
        response.raise_for_status()
        token_data = response.json()
        cached_token = token_data["access_token"]
        expires_in = token_data["expires_in"]

        token_expiration = current_time + expires_in - 600
        logger.debug(f"IAM token fetched, expires in {expires_in} seconds.")

        return cached_token
    except requests.exceptions.RequestException as err:
        logger.error(f"Error fetching IAM token: {err}")
        raise HTTPException(status_code=500, detail=f"Error fetching IAM token: {err}")


# Function to fetch the token (on-prem)
def get_onprem_token():
    global cached_token, token_expiration
    current_time = time.time()

    if cached_token and current_time < token_expiration:
        logger.debug("Using cached on-prem token.")
        return cached_token

    logger.debug("Fetching new token from CPD...")

    try:
        response = requests.post(
            CPD_AUTH_URL,
            headers={"Content-Type": "application/json"},
            json={
                "username": f"{USERNAME}",
                "api_key": f"{IBM_API_KEY}",
            },
            verify=False,
        )
        response.raise_for_status()
        token_data = response.json()
        cached_token = token_data["token"]

        token_expiration = current_time + 3600 - 60
        logger.debug(
            "token fetched, expires at %s",
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(token_expiration)),
        )

        return cached_token
    except requests.exceptions.RequestException as err:
        logger.error(f"Error fetching token: {err}")
        raise HTTPException(status_code=500, detail=f"Error fetching token: {err}")


def format_debug_output(request_data):
    headers = ["by API", "Parameter", "API Value", "Default Value", "Explanation"]
    table = []

    parameters = [
        (
            "Model ID",
            request_data.get("model", "ibm/granite-20b-multilingual"),
            "ibm/granite-20b-multilingual",
            "ID of the model to use for completion",
        ),
        (
            "Max Tokens",
            request_data.get("max_tokens", 2000),
            2000,
            "Maximum number of tokens to generate in the completion. The total tokens, prompt + completion.",
        ),
        (
            "Temperature",
            request_data.get("temperature", 0.2),
            0.2,
            "Controls the randomness of the generated output. Higher values make the output more random.",
        ),
        (
            "Presence Penalty",
            request_data.get("presence_penalty", 1),
            1,
            "Penalizes new tokens based on whether they appear in the text so far.",
        ),
        (
            "top_p",
            request_data.get("top_p", 1),
            1,
            "Nucleus sampling parameter.",
        ),
        (
            "best_of",
            request_data.get("best_of", 1),
            1,
            "Generates multiple completions server-side, returning the 'best' one.",
        ),
        (
            "echo",
            request_data.get("echo", False),
            False,
            "If True, echoes the prompt back along with the completion.",
        ),
        (
            "n",
            request_data.get("n", 1),
            1,
            "Number of completions to generate for each prompt.",
        ),
        (
            "seed",
            request_data.get("seed", None),
            None,
            "If specified, ensures deterministic outputs.",
        ),
        (
            "stop",
            request_data.get("stop", None),
            None,
            "Up to 4 sequences where the model will stop generating further tokens.",
        ),
        (
            "logit_bias",
            request_data.get("logit_bias", None),
            None,
            "Adjusts the likelihood of specified tokens appearing in the completion.",
        ),
        (
            "logprobs",
            request_data.get("logprobs", None),
            None,
            "Includes log probabilities on the most likely tokens.",
        ),
        (
            "stream",
            request_data.get("stream", False),
            False,
            "If True, streams back partial progress.",
        ),
        (
            "suffix",
            request_data.get("suffix", None),
            None,
            "Specifies a suffix that comes after the generated text.",
        ),
    ]

    green = "\033[92m"
    yellow = "\033[93m"
    reset = "\033[0m"

    for param, api_value, default_value, explanation in parameters:
        if api_value is not None:
            color = yellow
            provided_by_api = f"{yellow}X{reset}"
        else:
            color = green
            provided_by_api = ""

        table.append(
            [
                provided_by_api,
                f"{color}{param}{reset}",
                f'{color}"{api_value}"{reset}'
                if isinstance(api_value, str)
                else f"{color}{api_value}{reset}",
                f"{color}{default_value}{reset}",
                f"{color}{explanation}{reset}",
            ]
        )

    return tabulate(
        table,
        headers,
        tablefmt="pretty",
        colalign=("center", "left", "center", "center", "left"),
    )


def get_watsonx_token():
    logger.info(f"On prem value: {on_prem} (on_prem == '1'? {on_prem == '1'})")
    if on_prem == "1":
        logger.info("Getting on-prem token")
        token = get_onprem_token()
    else:
        logger.info("Getting IAM token")
        token = get_iam_token()
    return token


# Fetch the models from Watsonx
def get_watsonx_models():
    try:
        token = get_watsonx_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        params = {
            "version": api_version,
            "filters": "function_text_generation,!lifecycle_withdrawn:and",
            "limit": 200,
        }
        response = requests.get(
            WATSONX_MODELS_URL, headers=headers, params=params, verify=False
        )

        if response.status_code == 404:
            logger.error("404 Not Found: The endpoint or version might be incorrect.")
            raise HTTPException(
                status_code=404, detail="Watsonx Models API endpoint not found."
            )

        response.raise_for_status()
        models_data = response.json()

        logger.debug(f"Count models: {models_data.get('total_count')}")

        # Optionally: get custom models (on-prem only)
        if on_prem == "1":
            try:
                custom_resp = requests.get(
                    WATSONX_CUSTOM_MODELS_URL, headers=headers, verify=False
                )
                if custom_resp.status_code == 404:
                    logger.warning(
                        "Custom models endpoint not found, skipping custom models."
                    )
                else:
                    custom_resp.raise_for_status()
                    custom_models_data = custom_resp.json()
                    logger.debug(
                        f"Count custom models: {custom_models_data.get('total_count')}"
                    )

                    models_data["total_count"] = models_data.get(
                        "total_count", 0
                    ) + custom_models_data.get("total_count", 0)
                    models_data["resources"] = models_data.get(
                        "resources", []
                    ) + custom_models_data.get("resources", [])
            except requests.exceptions.RequestException as err:
                logger.warning(f"Error fetching custom models: {err}")

        logger.debug(f"Combined models: {models_data}")

        return models_data
    except requests.exceptions.RequestException as err:
        logger.error(f"Error fetching models from Watsonx.ai: {err}")
        raise HTTPException(
            status_code=500, detail=f"Error fetching models from Watsonx.ai: {err}"
        )


# Convert Watsonx models to OpenAI-like format
def convert_watsonx_to_openai_format(watsonx_data):
    openai_models = []

    for model in watsonx_data.get("resources", []):
        openai_model = {
            "id": model["model_id"],
            "object": "model",
            "created": int(time.time()),
            "owned_by": f"{model.get('provider')} / {model.get('source')}",
            "description": f"{model.get('short_description', '')} Supports tasks like {', '.join(model.get('task_ids', []))}.",
        }
        openai_models.append(openai_model)

    return {"data": openai_models}


# FastAPI route for /v1/models
@app.get("/models")
async def fetch_models():
    try:
        models = get_watsonx_models()
        logger.debug(f"Available models: {models}")
        openai_like_models = convert_watsonx_to_openai_format(models)
        return openai_like_models
    except Exception as err:
        logger.error(f"Error fetching models: {err}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching models: {err}")


# FastAPI route for /v1/models/{model_id}
@app.get("/models/{model_id}")
async def fetch_model_by_id(model_id: str):
    try:
        models = get_watsonx_models()
        model = next(
            (m for m in models.get("resources", []) if m.get("model_id") == model_id),
            None,
        )

        if model:
            openai_model = {
                "id": model["model_id"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": f"{model.get('provider')} / {model.get('source')}",
                "description": f"{model.get('short_description', '')} Supports tasks like {', '.join(model.get('task_ids', []))}.",
                "max_tokens": model.get("model_limits", {}).get("max_output_tokens"),
                "token_limits": {
                    "max_sequence_length": model.get("model_limits", {}).get(
                        "max_sequence_length"
                    ),
                    "max_output_tokens": model.get("model_limits", {}).get(
                        "max_output_tokens"
                    ),
                },
            }
            return {"data": [openai_model]}

        raise HTTPException(
            status_code=404, detail=f"Model with ID {model_id} not found."
        )

    except Exception as err:
        logger.error(f"Error fetching model by ID: {err}")
        raise HTTPException(
            status_code=500, detail=f"Error fetching model by ID: {err}"
        )


# ------------------------ /v1/completions ------------------------


@app.post("/completions")
async def watsonx_completions(request: Request):
    logger.info("Received a Watsonx completion request.")

    try:
        request_data = await request.json()
    except Exception as e:
        logger.error(f"Error parsing request: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON request body")

    prompt = request_data.get("prompt", "")

    if isinstance(prompt, list):
        prompt = " ".join(prompt)
    elif not isinstance(prompt, str):
        logger.error(
            f"Invalid type for 'prompt': {type(prompt)}. Expected a string or list of strings."
        )
        raise HTTPException(
            status_code=400,
            detail="Invalid type for 'prompt'. Expected a string or list of strings.",
        )

    model_id = request_data.get("model", "ibm/granite-3-8b-instruct")
    max_tokens = request_data.get("max_tokens", 2000)
    temperature = request_data.get("temperature", 0.2)
    best_of = request_data.get("best_of", 1)
    n = request_data.get("n", 1)
    presence_penalty = request_data.get("presence_penalty", 1)
    echo = request_data.get("echo", False)
    logit_bias = request_data.get("logit_bias", None)
    logprobs = request_data.get("logprobs", None)
    stop = request_data.get("stop", None)
    suffix = request_data.get("suffix", None)
    stream = request_data.get("stream", False)
    seed = request_data.get("seed", None)
    top_p = request_data.get("top_p", 1)

    logger.debug("Parameter source debug:")
    logger.debug("\n" + format_debug_output(request_data))

    iam_token = get_watsonx_token()
    logger.debug("Bearer token:\n%s", iam_token)

    watsonx_payload = {
        "input": prompt,
        "parameters": {
            "decoding_method": "sample",  # Greedy not supported
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_k": 50,
            "top_p": top_p,
            "random_seed": seed,
            "repetition_penalty": presence_penalty,
        },
        "model_id": model_id,
        "project_id": PROJECT_ID,
    }

    if stop:
        watsonx_payload["parameters"]["stop_sequences"] = stop
    if logit_bias:
        watsonx_payload["parameters"]["logit_bias"] = logit_bias

    formatted_payload = json.dumps(watsonx_payload, indent=4, ensure_ascii=False)
    logger.debug(f"Sending request to Watsonx.ai (completion): {formatted_payload}")

    headers = {
        "Authorization": f"Bearer {iam_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    try:
        response = requests.post(
            WATSONX_URL, json=watsonx_payload, headers=headers, verify=False
        )
        response.raise_for_status()
        watsonx_data = response.json()
        logger.debug(
            "Received response from Watsonx.ai: %s",
            json.dumps(watsonx_data, indent=4),
        )
    except requests.exceptions.HTTPError as err:
        error_message = response.text
        logger.error(f"HTTPError: {err}, Response: {error_message}")
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Error from Watsonx.ai: {error_message}",
        )
    except requests.exceptions.RequestException as err:
        logger.error(f"RequestException: {err}")
        raise HTTPException(status_code=500, detail=f"Error calling Watsonx.ai: {err}")

    results = watsonx_data.get("results", [])
    if results and "generated_text" in results[0]:
        generated_text = results[0]["generated_text"]
        logger.debug("Generated text from Watsonx.ai: \n%s", generated_text)
    else:
        generated_text = "\n\nNo response available."
        logger.warning("No generated text found in Watsonx.ai response.")

    openai_response = {
        "id": f"cmpl-{str(uuid.uuid4())[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_id,
        "system_fingerprint": f"fp_{str(uuid.uuid4())[:12]}",
        "choices": [
            {
                "text": generated_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": results[0].get("stop_reason", "length")
                if results
                else "length",
            }
        ],
        "usage": {
            "prompt_tokens": results[0].get("input_token_count", 0) if results else 0,
            "completion_tokens": results[0].get("generated_token_count", 0)
            if results
            else 0,
            "total_tokens": (
                (results[0].get("input_token_count", 0) if results else 0)
                + (results[0].get("generated_token_count", 0) if results else 0)
            ),
        },
    }

    # non-stream 모드
    if not stream:
        logger.debug(
            "Returning OpenAI-compatible completion response: %s",
            json.dumps(openai_response, indent=4),
        )
        return openai_response

    # stream 모드: SSE로 래핑
    logger.debug("Returning STREAMING OpenAI-compatible completion response")

    async def event_generator():
        chunk1 = {
            "id": openai_response["id"],
            "object": "text_completion",
            "created": openai_response["created"],
            "model": openai_response["model"],
            "choices": [
                {
                    "text": openai_response["choices"][0]["text"],
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
        }

        chunk2 = {
            "id": openai_response["id"],
            "object": "text_completion",
            "created": openai_response["created"],
            "model": openai_response["model"],
            "choices": [
                {
                    "text": "",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": openai_response["choices"][0]["finish_reason"],
                }
            ],
        }

        yield f"data: {json.dumps(chunk1)}\n\n"
        yield f"data: {json.dumps(chunk2)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ------------------------ /v1/chat/completions ------------------------


@app.post("/chat/completions")
async def watsonx_chat_completions(request: Request):
    logger.info("Received a Watsonx chat completion request.")

    try:
        request_data = await request.json()
        logger.debug(f"Received request data: {json.dumps(request_data, indent=4)}")
    except Exception as e:
        logger.error(f"Error parsing request: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON request body")

    model_id = request_data.get("model", "ibm/granite-3-8b-instruct")
    max_tokens = request_data.get("max_tokens", 1024)
    temperature = request_data.get("temperature", 1.0)
    n = request_data.get("n", 1)
    logit_bias = request_data.get("logit_bias", None)
    logprobs = request_data.get("logprobs", False)
    stop = request_data.get("stop", None)
    seed = request_data.get("seed", None)
    top_p = request_data.get("top_p", 1.0)
    messages = request_data.get("messages", [])
    frequency_penalty = request_data.get("frequency_penalty", 0.0)
    presence_penalty = request_data.get("presence_penalty", 0.0)
    stream = request_data.get("stream", False)

    logger.debug("Parameter source debug:")
    logger.debug("\n" + format_debug_output(request_data))

    iam_token = get_watsonx_token()

    watsonx_payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "model_id": model_id,
        "project_id": PROJECT_ID,
        "n": n,
        "seed": seed,
        "logprobs": logprobs,
    }

    if stop is not None:
        watsonx_payload["stop"] = stop
    # NOTE: chat 엔드포인트에는 parameters 블록이 없어서 logit_bias는 일단 무시
    # if logit_bias:
    #     watsonx_payload["logit_bias"] = logit_bias

    formatted_payload = json.dumps(watsonx_payload, indent=4, ensure_ascii=False)
    logger.debug(f"Sending request to Watsonx.ai (chat): {formatted_payload}")

    headers = {
        "Authorization": f"Bearer {iam_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    try:
        response = requests.post(
            WATSONX_URL_CHAT, json=watsonx_payload, headers=headers, verify=False
        )
        response.raise_for_status()
        watsonx_data = response.json()
        logger.debug(
            "Received response from Watsonx.ai: %s",
            json.dumps(watsonx_data, indent=4),
        )
    except requests.exceptions.HTTPError as err:
        error_message = response.text
        logger.error(f"HTTPError: {err}, Response: {error_message}")
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Error from Watsonx.ai: {error_message}",
        )
    except requests.exceptions.RequestException as err:
        logger.error(f"RequestException: {err}")
        raise HTTPException(status_code=500, detail=f"Error calling Watsonx.ai: {err}")

    # watsonx chat 응답을 OpenAI 형식으로 맞추기
    openai_like = {
        "id": watsonx_data.get("id", f"chatcmpl-{str(uuid.uuid4())[:12]}"),
        "object": "chat.completion",
        "created": watsonx_data.get("created", int(time.time())),
        "model": watsonx_data.get("model", model_id),
        "system_fingerprint": f"fp_{str(uuid.uuid4())[:12]}",
        "choices": watsonx_data.get("choices", []),
        "usage": watsonx_data.get("usage", {}),
    }

    # non-stream 모드
    if not stream:
        logger.debug(
            "Returning OpenAI-compatible chat response: %s",
            json.dumps(openai_like, indent=4),
        )
        return openai_like

    # stream 모드: SSE로 래핑
    logger.debug("Returning STREAMING OpenAI-compatible chat response")

    async def event_generator():
        if openai_like["choices"]:
            first_choice = openai_like["choices"][0]
        else:
            first_choice = {
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }

        message = first_choice.get("message", {})
        content = message.get("content", "")
        role = message.get("role", "assistant")
        finish_reason = first_choice.get("finish_reason", "stop")

        chunk1 = {
            "id": openai_like["id"],
            "object": "chat.completion.chunk",
            "created": openai_like["created"],
            "model": openai_like["model"],
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": role,
                        "content": content,
                    },
                    "finish_reason": None,
                }
            ],
        }

        chunk2 = {
            "id": openai_like["id"],
            "object": "chat.completion.chunk",
            "created": openai_like["created"],
            "model": openai_like["model"],
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }
            ],
        }

        yield f"data: {json.dumps(chunk1)}\n\n"
        yield f"data: {json.dumps(chunk2)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
