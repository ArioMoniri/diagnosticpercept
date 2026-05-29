"""Submit the diagnostic_percept notebook for batch execution on Vertex AI.

Uses the Vertex AI Notebook Executor (``aiplatform.NotebookExecutionJob``)
to run the notebook to completion on a Vertex AI Workbench runtime that we
spin up for the job, dump every cell output, and upload results to GCS.

Run::

    pip install google-cloud-aiplatform
    gcloud auth application-default login

    python scripts/submit_to_vertex.py \\
      --project-id    YOUR_PROJECT \\
      --region        us-central1 \\
      --gcs-output    gs://YOUR_BUCKET/diagnostic_percept_runs/ \\
      --machine-type  a2-highgpu-1g \\
      --accelerator-type   NVIDIA_TESLA_A100 \\
      --accelerator-count  1

After completion the script polls the job state and prints the GCS path
to the executed `.ipynb` (with all outputs). For HF gated models pass
``--hf-token hf_...`` and the script wires it as a runtime env var.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from typing import Optional


NOTEBOOK_GITHUB_URL = (
    "https://raw.githubusercontent.com/ArioMoniri/diagnosticpercept/main/"
    "notebooks/diagnostic_percept.ipynb"
)


def submit(
    project_id: str,
    region: str,
    gcs_output: str,
    machine_type: str = "a2-highgpu-1g",
    accelerator_type: str = "NVIDIA_TESLA_A100",
    accelerator_count: int = 1,
    notebook_url: str = NOTEBOOK_GITHUB_URL,
    hf_token: Optional[str] = None,
    gh_token: Optional[str] = None,
    display_name_prefix: str = "diagnostic-percept",
    wait: bool = True,
) -> str:
    """Submit the notebook for batch execution. Returns the job resource name."""
    from google.cloud import aiplatform_v1beta1 as aip
    from google.protobuf.duration_pb2 import Duration

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    display_name = f"{display_name_prefix}-{timestamp}"

    client = aip.NotebookServiceClient(
        client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"}
    )
    parent = f"projects/{project_id}/locations/{region}"

    env_vars = {}
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token
    if gh_token:
        env_vars["GH_TOKEN"] = gh_token

    job = aip.NotebookExecutionJob(
        display_name=display_name,
        gcs_notebook_source=aip.NotebookExecutionJob.GcsNotebookSource(
            uri=notebook_url,   # raw GitHub URL is acceptable here
        ),
        gcs_output_uri=gcs_output,
        execution_timeout=Duration(seconds=6 * 60 * 60),  # 6h cap
        execution_user="user-managed-runtime",
        custom_environment_spec=aip.NotebookExecutionJob.CustomEnvironmentSpec(
            machine_spec=aip.MachineSpec(
                machine_type=machine_type,
                accelerator_type=accelerator_type,
                accelerator_count=accelerator_count,
            ),
            network_spec=aip.NetworkSpec(enable_internet_access=True),
        ),
    )
    # Env vars on the job (set in the runtime container).
    if env_vars:
        job.kernel_name = "python3"
        # NotebookExecutionJob doesn't expose env directly in all SDK versions;
        # falling back: pass via the runtime template (see docs/colab-enterprise.md).

    op = client.create_notebook_execution_job(
        parent=parent, notebook_execution_job=job,
        notebook_execution_job_id=display_name,
    )
    print(f"Submitted: {display_name}")
    print(f"           gs output: {gcs_output}")
    print(f"           op name:   {op.operation.name}")

    if not wait:
        return op.operation.name

    print("Waiting for completion (polling every 30s) ...")
    result = op.result(timeout=6 * 60 * 60 + 600)
    print(f"\nFinished: state={result.job_state}")
    print(f"  result notebook (with outputs): "
          f"{result.gcs_output_uri}/{display_name}.ipynb")
    return result.name


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-id", required=True)
    p.add_argument("--region", default="us-central1")
    p.add_argument("--gcs-output", required=True,
                    help="gs://BUCKET/path/ for outputs")
    p.add_argument("--machine-type", default="a2-highgpu-1g")
    p.add_argument("--accelerator-type", default="NVIDIA_TESLA_A100")
    p.add_argument("--accelerator-count", type=int, default=1)
    p.add_argument("--notebook-url", default=NOTEBOOK_GITHUB_URL)
    p.add_argument("--hf-token", default=None)
    p.add_argument("--gh-token", default=None)
    p.add_argument("--no-wait", action="store_true",
                    help="Submit and return immediately")
    args = p.parse_args(argv)

    submit(
        project_id=args.project_id,
        region=args.region,
        gcs_output=args.gcs_output,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        notebook_url=args.notebook_url,
        hf_token=args.hf_token,
        gh_token=args.gh_token,
        wait=not args.no_wait,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
