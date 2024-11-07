


import os
import ray
from ray import serve
from fastapi import FastAPI, Request
import logging

from llama_cpp import Llama



logger = logging.getLogger("ray.serve")

app = FastAPI()

# Define the deployment
@serve.deployment(name="LLamaCPPDeployment")
@serve.ingress(app)
class LLamaCPPDeployment:

    def __init__(self):
        # Initialize the LLamaCPP model
        self.model_id = os.getenv("MODEL_ID")
        self.n_ctx = int(os.getenv("N_CTX"))
        self.n_batch = int(os.getenv("N_BATCH"))

        self.llama_cpp = Llama(model_path=MODEL_ID, n_ctx=self.n_ctx, n_batch=self.n_batch)

    @app.post("/v1/chat/completion")
    async def call_llama(self, request: Request):
        body = await request.json()
        print (body)
        prompt = body.get("prompt", "")

        max_tokens=256
        temperature=0.1
        top_p=0.5
        echo=False
        stop=["#"]

        output = self.llama_cpp(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=echo,
            stop=stop,
        )
        output_text = output["choices"][0]["text"].strip()
        return {"output": output_text}


model = LLamaCPPDeployment.bind()
