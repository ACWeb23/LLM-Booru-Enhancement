"""
This script implements a FastAPI application that provides an API for image tagging using a pre-trained ONNX model. The application allows users to send images in base64 format and receive tags based on the content of the images.
It also supports different modes of operation, including using only Booru tags, combining Booru tags with a language model (LLM), or using only the LLM for captioning.
To caption with a local LLM it connects to a separate LLM API running on a specified port. Such as llama.cpp or kobold.cpp. The LLM API should be running and accessible for the captioning to work in the "BooruTag+LLM" or "LLM" modes.
"""

import base64
import io
import os
from typing import List, Union, Optional

import numpy as np
import onnxruntime as rt
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import argparse

# ------------------------------------------------------------------
# Command Line Arguments
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Image Tagger API")
parser.add_argument("--Mode", type=str, default="BooruTag+LLM", choices=["BooruTag", "BooruTag+LLM", "LLM"], help="Image Captioning Mode")
parser.add_argument("--Port", type=int, default=8001, help="Port to run the API on")
parser.add_argument("--GeneralThreshold", type=float, default=0.50, help="General Tag Percentage Threshold")
parser.add_argument("--CharacterThreshold", type=float, default=0.85, help="Character Tag Percentage Threshold")
parser.add_argument("--LLM_Port", type=int, default=5001, help="Port to Connect to the LLM API on")

args = parser.parse_args()

# ------------------------------------------------------------------
# Local model files & Constants
# ------------------------------------------------------------------
MODEL_PATH = r"Models\model.onnx"
TAGS_PATH = r"Models\selected_tags.csv"

kaomojis = [
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", 
    ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||"
]


def load_labels(df):
    names = df["name"].map(
        lambda x: x.replace("_", " ") if x not in kaomojis else x
    ).tolist()

    rating = list(np.where(df["category"] == 9)[0])
    general = list(np.where(df["category"] == 0)[0])
    character = list(np.where(df["category"] == 4)[0])

    return names, rating, general, character


class Predictor:
    def __init__(self, general_threshold=0.50, character_threshold=0.85):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        if not os.path.exists(TAGS_PATH):
            raise FileNotFoundError(f"Tags CSV not found: {TAGS_PATH}")

        tags = pd.read_csv(TAGS_PATH)

        (
            self.tag_names,
            self.rating_indexes,
            self.general_indexes,
            self.character_indexes,
        ) = load_labels(tags)

        self.model = rt.InferenceSession(MODEL_PATH)

        _, h, _, _ = self.model.get_inputs()[0].shape
        self.target_size = h
        
        self.general_threshold = general_threshold
        self.character_threshold = character_threshold

    def prepare_image(self, image):
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        w, h = image.size
        size = max(w, h)

        padded = Image.new("RGB", (size, size), (255, 255, 255))
        padded.paste(image, ((size - w) // 2, (size - h) // 2))

        if size != self.target_size:
            padded = padded.resize(
                (self.target_size, self.target_size),
                Image.BICUBIC,
            )

        arr = np.asarray(padded, dtype=np.float32)
        # RGB -> BGR
        arr = arr[:, :, ::-1]

        return np.expand_dims(arr, 0)

    def predict(self, image):
        image = self.prepare_image(image)

        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name

        preds = self.model.run(
            [output_name],
            {input_name: image},
        )[0][0]

        labels = list(zip(self.tag_names, preds.astype(float)))

        general = [
            labels[i]
            for i in self.general_indexes
            if labels[i][1] > self.general_threshold
        ]

        character = [
            labels[i]
            for i in self.character_indexes
            if labels[i][1] > self.character_threshold
        ]

        tags = sorted(
            general + character,
            key=lambda x: x[1],
            reverse=True,
        )

        return ", ".join(tag for tag, _ in tags)


# Initialize Predictor with dynamic command line thresholds
predictor = Predictor(
    general_threshold=args.GeneralThreshold, 
    character_threshold=args.CharacterThreshold
)

app = FastAPI(title="Image Tagger API")

# ------------------------------------------------------------------
# Request Pydantic Schemas
# ------------------------------------------------------------------
class TextContent(BaseModel):
    type: str
    text: str


class ImageURL(BaseModel):
    url: str


class ImageContent(BaseModel):
    type: str
    image_url: Union[ImageURL, str]


ContentItem = Union[TextContent, ImageContent]


class Message(BaseModel):
    role: str
    content: Union[str, List[ContentItem]]  # Support both raw strings and object lists


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]


# ------------------------------------------------------------------
# Local LLM Communication Helper
# ------------------------------------------------------------------
def query_local_llm(image_b64: str, user_prompt: str, booru_tags: Optional[str] = None, system_prompt: Optional[str] = None) -> str:
    """Sends the base64 image, dynamic prompts, and tags to the local OpenAI-compliant LLM API."""
    url = f"http://localhost:{args.LLM_Port}/v1/chat/completions"
    
    # Inject Booru tags into the prompt text if available
    if booru_tags:
        full_user_prompt = f"{user_prompt}\n\n[Guidance Tags: {booru_tags}]"
    else:
        full_user_prompt = user_prompt

    # Construct standard OpenAI format payload
    messages = []
    
    # Pass along system instructions if they were supplied by the host application
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
        
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": full_user_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}"
                }
            }
        ]
    })

    payload = {
        "model": "local-vision-model",
        "messages": messages
    }
    
    try:
        response = requests.post(url, json=payload, timeout=90)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to communicate with local LLM on port {args.LLM_Port}. Error: {str(e)}"
        )


# ------------------------------------------------------------------
# Router API Endpoint
# ------------------------------------------------------------------
@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    image_b64 = None
    user_prompt = "Describe this image."  # Default fallback text instruction
    system_prompt = None

    # Parse incoming messages array dynamically
    for message in request.messages:
        # 1. Handle incoming System prompts
        if message.role == "system":
            if isinstance(message.content, str):
                system_prompt = message.content
            elif isinstance(message.content, list):
                for item in message.content:
                    if getattr(item, "type", None) == "text":
                        system_prompt = item.text
            continue

        # 2. Handle User messages containing prompts and images
        if message.role == "user":
            if isinstance(message.content, str):
                user_prompt = message.content
            elif isinstance(message.content, list):
                for item in message.content:
                    if getattr(item, "type", None) == "text":
                        user_prompt = item.text
                    elif getattr(item, "type", None) == "image_url":
                        if isinstance(item.image_url, str):
                            image_b64 = item.image_url
                        else:
                            image_b64 = item.image_url.url

    if image_b64 is None:
        raise HTTPException(status_code=400, detail="No image supplied.")

    # Strip Data URI scheme if present
    if image_b64.startswith("data:"):
        image_b64_raw = image_b64.split(",", 1)[1]
    else:
        image_b64_raw = image_b64

    # --- Mode Evaluation Logic ---
    if args.Mode == "BooruTag":
        try:
            image_bytes = base64.b64decode(image_b64_raw)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image.")
        
        final_output = predictor.predict(image)

    elif args.Mode == "BooruTag+LLM":
        try:
            image_bytes = base64.b64decode(image_b64_raw)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image.")
        
        # Pull tags from ONNX first, then forward everything to backend LLM runner
        tags = predictor.predict(image)
        final_output = query_local_llm(image_b64_raw, user_prompt, booru_tags=tags, system_prompt=system_prompt)

    elif args.Mode == "LLM":
        # Straight passthrough to the vision LLM using host instructions
        final_output = query_local_llm(image_b64_raw, user_prompt, booru_tags=None, system_prompt=system_prompt)

    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": 0,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": final_output
                }
            }
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": args.Mode,
        "model": MODEL_PATH,
        "tags": TAGS_PATH
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "App:app",
        host="0.0.0.0",
        port=args.Port,
    )