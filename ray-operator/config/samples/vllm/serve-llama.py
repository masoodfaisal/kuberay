

import os
import logging
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from ray import serve

from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse
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
        self.model_id = os.getenv("MODEL_ID", default="Qwen/Qwen2-0.5B-Instruct-GGUF")
        # self.n_ctx = int(os.getenv("N_CTX"))
        # self.n_batch = int(os.getenv("N_BATCH"))
        # self.llama_cpp = Llama(model_path=MODEL_ID, n_ctx=self.n_ctx, n_batch=self.n_batch)
        self.llm = Llama.from_pretrained(repo_id=self.model_id,filename="*q8_0.gguf")
        #"hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF",
        print("__init__ Complete")

    @app.post("/v1/chat/completion")
    async def call_llama(self, request: Request):
        body = await request.json()
        print (body)
        prompt = body.get("prompt", "")

        # max_tokens=256
        # temperature=0.1
        # top_p=0.5
        # echo=False
        # stop=["#"]

        # output = self.llama_cpp(
        #     prompt,
        #     max_tokens=max_tokens,
        #     temperature=temperature,
        #     top_p=top_p,
        #     echo=echo,
        #     stop=stop,
        # )
        # output_text = output["choices"][0]["text"].strip()
        # return {"output": output_text}

        # output_text = self.llm.create_chat_completion(
        #     messages = [
        #         {
        #             "role": "user",
        #             "content": prompt
        #         }
        #     ]
        # )        
        output = self.llm(
            "Q: " + prompt + " A: ", # Prompt
            max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
            stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
            echo=True # Echo the prompt back in the output
        )        
        return JSONResponse(content={"output": output})




model = LLamaCPPDeployment.bind()
