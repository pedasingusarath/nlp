# Academic Question Response Generator

A student mini-project based on the assignment brief. It uses pretrained DistilGPT2,
PyTorch, a TorchServe handler, a Streamlit interface, and a documented SageMaker
deployment path.

## 1. Project structure

- `local_inference.py` — downloads and runs DistilGPT2 locally with PyTorch.
- `handler.py` — TorchServe custom handler (validation, tokenization, generation).
- `config.properties` — TorchServe server configuration.
- `app.py` — Streamlit interface.
- `build_model_archive.ps1` — downloads model files and packages a `.mar` archive.
- `sagemaker/` — deployment notes and container integration guidance.

## 2. Requirements

Use Python 3.10–3.12 for the most straightforward compatibility with the
TorchServe/PyTorch ecosystem. Create a virtual environment:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If TorchServe installation fails on your OS/Python version, use a supported
Linux environment (for example, Ubuntu or WSL2) and check the current TorchServe
compatibility matrix. TorchServe is in limited maintenance, as the assignment notes.

## 3. Run local model first

```powershell
python local_inference.py
```

Example prompt: `Explain the working of CNN`

The first run downloads DistilGPT2 from Hugging Face and may take time.

## 4. Run the interface

```powershell
streamlit run app.py
```

The default interface uses local PyTorch inference so that the UI can be tested
without AWS. Do not put AWS access keys in Streamlit code or a public repository.

## 5. Package and run TorchServe (Linux/WSL2 recommended)

Run the archive script from the project root:

```bash
bash build_model_archive.sh
```

For Windows PowerShell, inspect and run `build_model_archive.ps1`. Then start:

```bash
torchserve --start --ncs --model-store model_store --models academic_qa=academic_qa.mar --ts-config config.properties
```

Test the prediction API:

```bash
curl -X POST http://127.0.0.1:8080/predictions/academic_qa \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain the working of CNN","max_new_tokens":50}'
```

Stop TorchServe:

```bash
torchserve --stop
```

## 6. API contract

Request:
```json
{"prompt":"Explain CNN","max_new_tokens":50,"temperature":0.8}
```

Response:
```json
[{"generated_text":"..."}]
```

Invalid request:
```json
{"error":"prompt is required"}
```

The exact error envelope depends on TorchServe's HTTP error handling; record the
actual response from your running version in the report.

## 7. SageMaker deployment

SageMaker does not automatically run TorchServe in every deployment. The
assignment requires explicitly configuring the intended model server. Use a
faculty-approved AWS account, package the tested TorchServe application in a
compatible inference container, push the image to ECR, upload model artifacts
to S3, create a SageMaker model and endpoint configuration, deploy an endpoint,
and invoke it from a secure backend.

The included `sagemaker/DEPLOYMENT.md` gives the deployment checklist. A working
cloud endpoint depends on the AWS region, IAM role, ECR image, S3 model artifact,
instance type, and tested container compatibility; these cannot be pre-created
without your AWS account.

## 8. Minimum tests for the assignment

1. Explain the working of CNN.
2. What is an RNN?
3. Explain tokenization in NLP.
4. What is attention in Transformers?
5. What is machine learning?
6. Explain supervised learning.
7. What is an operating system?
8. Explain database normalization.
9. What is a compiler?
10. Explain cloud computing.

Also test empty input, a prompt longer than 2,000 characters, and two values of
`max_new_tokens`. Compare local PyTorch, local TorchServe, and SageMaker outputs,
response formats, and latency.

## 9. Important limitations

DistilGPT2 is a general text-generation model, not a verified academic question
answering model. It may generate irrelevant, incomplete, repetitive, or incorrect
content. The application must display the disclaimer and students should verify
outputs against course material.

SageMaker endpoints and related AWS resources can incur charges. Delete endpoints,
endpoint configurations, unused models, and other resources after the demonstration.
