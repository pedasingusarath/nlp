# SageMaker deployment checklist

This is a deployment plan, not a claim that an AWS endpoint has already been
created. Follow your faculty's approved AWS account and region.

1. Validate the model archive and TorchServe API locally first.
2. Choose a SageMaker-compatible inference container strategy that explicitly
   starts TorchServe. Confirm PyTorch, Java, TorchServe, and container versions
   are compatible.
3. Build a container image that:
   - installs the tested TorchServe runtime and dependencies;
   - copies `config.properties`, `handler.py`, and `academic_qa.mar`;
   - starts TorchServe with the model loaded;
   - exposes the inference health and invocation routes expected by SageMaker.
4. Push the image to Amazon ECR.
5. Upload the model artifact to an S3 bucket.
6. Create a SageMaker Model referencing the ECR image, S3 artifact, and an
   execution IAM role.
7. Create an endpoint configuration using a faculty-approved instance type.
8. Create and wait for the endpoint to become InService.
9. Invoke it using the SageMaker Runtime API from a trusted backend. Do not
   expose AWS credentials in Streamlit/browser code.
10. Record endpoint name (redact account details), region, instance type,
    model-server configuration, request/response, and latency.
11. Delete the endpoint and unused endpoint configuration/model resources after
    the demonstration.

## Important integration note

SageMaker custom inference containers expect health and invocation behavior
compatible with its hosting contract (commonly `/ping` and `/invocations`).
TorchServe's default prediction route is `/predictions/academic_qa`. A production
deployment therefore needs a tested adapter/proxy or a SageMaker-compatible
TorchServe integration that maps the SageMaker routes to TorchServe. Do not assume
that exposing port 8080 alone is sufficient. Verify this end-to-end in your
chosen container before creating a paid endpoint.
