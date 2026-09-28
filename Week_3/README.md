# Lab 3 — Serve a Model with FastAPI, Docker, and a Kubernetes Canary
**Module 3: Model Serving Strategies**

## Objective
Package a trained classifier as a REST service, containerize it, and roll out a new
model version to Kubernetes as a **canary** (about 10% of traffic) — then decide whether to promote it.

Covers **LO1** (deploy models with various serving architectures).

## Dataset and Models
`load_wine` (scikit-learn toy dataset: 178 samples, 13 features, 3 classes).
`train_model.py` trains two versions of a Random Forest:
- **v1** (stable): 30 trees
- **v2** (candidate): 200 trees

## Files
| File | Purpose |
|---|---|
| `train_model.py` | Trains and saves `models/model_v1.joblib` and `models/model_v2.joblib` |
| `app.py` | FastAPI server: `GET /health`, `POST /predict` |
| `Dockerfile` | Builds one image per model version (`--build-arg MODEL_VERSION=v1`) |
| `wine-classifier.yaml` | Deployments for v1 (9 replicas) and v2 (1 replica) behind one Service |
| `loadtest.py` | Sends requests, reports which version answered plus p50/p95/p99 latency |
| `requirements.txt` / `requirements-serve.txt` | Local environment / container image dependencies |

## Part A — Serve Locally (tested)
```powershell
# Install the project's Python dependencies.
pip install -r requirements.txt

# Train both model versions and save their artifacts.
python train_model.py

# Terminal 1: stable model
# .env provides MODEL_PATH=models/model_v1.joblib and MODEL_VERSION=v1.
# Start the v1 API on port 8001.
uvicorn app:app --port 8001
```

In a second PowerShell terminal, start the candidate model:
```powershell
# Override the v1 defaults from .env for this terminal.
# Point this terminal at the v2 model file.
$env:MODEL_PATH = 'models/model_v2.joblib'

# Label responses from this server as v2.
$env:MODEL_VERSION = 'v2'

# Start the v2 API on port 8002.
uvicorn app:app --port 8002
```

Check the v1 health endpoint and make a sample prediction from a third PowerShell terminal:
```powershell
# Check that the v1 server is healthy.
Invoke-RestMethod http://localhost:8001/health

# Build the sample prediction request body.
$body = @{ features = @(13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050) } | ConvertTo-Json -Compress

# Send the sample to v1.
Invoke-RestMethod -Uri http://localhost:8001/predict -Method Post -ContentType 'application/json' -Body $body

# Check that the v2 server is healthy.
Invoke-RestMethod http://localhost:8002/health

# Send the sample to v2.
Invoke-RestMethod -Uri http://localhost:8002/predict -Method Post -ContentType 'application/json' -Body $body

# Measure latency for 200 requests to v1.
python loadtest.py --url http://localhost:8001 --n 200

# Measure latency for 200 requests to v2.
python loadtest.py --url http://localhost:8002 --n 200
```
Sending the wrong number of features returns HTTP 422.

**Reference results from a test run (your numbers will differ by machine):**

| | v1 (30 trees) | v2 (200 trees) |
|---|---|---|
| Test accuracy | 1.000 | 1.000 |
| p50 latency | 3.7 ms | 10.5 ms |
| p95 latency | 4.3 ms | 11.1 ms |

## Part B — Containerize and Canary on Kubernetes
Make sure Docker Desktop is running and Minikube is started with `minikube start --driver=docker`.
The commands below use Minikube's bundled `kubectl`, so a separate kubectl installation is not required.

**1. Build both images** (from the lab folder, after `train_model.py`):
```powershell
# Build the v1 container image.
docker build --build-arg MODEL_VERSION=v1 -t wine-classifier:v1 .

# Build the v2 container image.
docker build --build-arg MODEL_VERSION=v2 -t wine-classifier:v2 .
```
**2. Test the Docker containers locally:**
Use different host ports from the local Uvicorn servers. These commands start both images as containers:
```powershell
# Start v1, mapping host port 8101 to container port 8000.
docker run --rm -d --name wine-v1 -p 8101:8000 wine-classifier:v1

# Start v2, mapping host port 8102 to container port 8000.
docker run --rm -d --name wine-v2 -p 8102:8000 wine-classifier:v2
```
Check both health endpoints and send a sample prediction:
```powershell
# Check the v1 container health endpoint.
Invoke-RestMethod http://localhost:8101/health

# Check the v2 container health endpoint.
Invoke-RestMethod http://localhost:8102/health

# Build the sample prediction request body.
$body = @{ features = @(13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050) } | ConvertTo-Json -Compress

# Send the sample to the v1 container.
Invoke-RestMethod -Uri http://localhost:8101/predict -Method Post -ContentType 'application/json' -Body $body

# Send the sample to the v2 container.
Invoke-RestMethod -Uri http://localhost:8102/predict -Method Post -ContentType 'application/json' -Body $body

# Show the last 20 log lines from v1.
docker logs wine-v1 --tail 20

# Show the last 20 log lines from v2.
docker logs wine-v2 --tail 20

# Stop both test containers.
docker stop wine-v1 wine-v2
```
The health responses should report `v1` and `v2` respectively. The prediction response should include the corresponding `model_version`.

**3. Load both images into Minikube:**
```powershell
# Load the v1 image into Minikube.
minikube image load wine-classifier:v1

# Load the v2 image into Minikube.
minikube image load wine-classifier:v2

```

**4. Deploy:**
```powershell
# Create or update the Deployments and Service.
minikube kubectl -- apply -f .\wine-classifier.yaml

# List the classifier pods and their status.
minikube kubectl -- get pods -l app=wine-classifier

# Wait for the v1 Deployment to become ready.
minikube kubectl -- rollout status deployment/wine-classifier-v1

# Wait for the v2 Deployment to become ready.
minikube kubectl -- rollout status deployment/wine-classifier-v2

```

**5. Verify and test the Kubernetes pods:**
Check that the pods are `Running` and `1/1` Ready, then list the v1 and v2 pod names:
```powershell
# Show all classifier pods with node and IP details.
minikube kubectl -- get pods -l app=wine-classifier -o wide

# List only the v1 pods.
minikube kubectl -- get pods -l app=wine-classifier,version=v1

# List only the v2 pods.
minikube kubectl -- get pods -l app=wine-classifier,version=v2

```

Test the Service by forwarding its port. Keep this command running in one terminal:
```powershell
# Forward local port 8080 to the Service; leave running while testing.
minikube kubectl -- port-forward service/wine-classifier 8080:80

```
In another PowerShell terminal, check health and make a prediction. The health response identifies which model served the request:
```powershell
# Check the Service health and identify the responding model.
Invoke-RestMethod http://localhost:8080/health

# Build the sample prediction request body.
$body = @{ features = @(13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050) } | ConvertTo-Json -Compress

# Send a prediction through the Service.
Invoke-RestMethod -Uri http://localhost:8080/predict -Method Post -ContentType 'application/json' -Body $body

```

To inspect a specific pod's logs, replace `POD_NAME` with a name from `get pods`:
```powershell
# Show the last 20 log lines from the selected pod.
minikube kubectl -- logs pod/POD_NAME --tail=20
kubectl logs pod/POD_NAME --tail=20

```

To test a specific pod directly, replace `POD_NAME` with its name from `get pods`. Run this in one terminal; use local port `8081` for a v1 pod or `8082` for a v2 pod:
```powershell
# Forward local port 8081 to the selected pod; leave running while testing.
minikube kubectl -- port-forward pod/POD_NAME 8081:8000

```
Keep the port-forward running. In a second terminal, test its health and prediction endpoints (use `8082` in the URL when forwarding a v2 pod):
```powershell
# Check the selected pod's health (use port 8082 for a v2 pod).
Invoke-RestMethod http://localhost:8081/health

# Build the sample prediction request body.
$body = @{ features = @(13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050) } | ConvertTo-Json -Compress

# Send a prediction to the selected pod.
Invoke-RestMethod -Uri http://localhost:8081/predict -Method Post -ContentType 'application/json' -Body $body

```
Press Ctrl+C in the port-forward terminal when finished.

**6. Measure the traffic split from inside the cluster** (each curl opens a fresh connection):
```powershell
# Send 200 prediction requests to the Service from inside the Kubernetes network.
# Each curl invocation opens a new connection so the Service can select a pod.
# Prepare a shell script that sends 200 fresh-connection predictions.
$probeScript = 'for i in $(seq 1 200); do curl -sS -X POST http://wine-classifier/predict -H "Content-Type: application/json" -d ''{"features":[13.2,1.78,2.14,11.2,100,2.65,2.76,0.26,1.28,4.38,1.05,3.4,1050]}''; echo; done'


# Base64 keeps PowerShell from changing quotes while passing the script to Minikube.
# Encode the script to preserve its quoting.
$encodedProbe = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($probeScript))

# Decode the script inside the curl container, then run it with the container's shell.
# Set the container command to decode and execute the probe.
$containerCommand = "echo $encodedProbe | base64 -d | sh"

# Create a temporary curl pod, stream its output, and remove it when it finishes.
# The full executable path avoids Minikube PATH issues in older PowerShell terminals.
# Run the probe pod and capture its responses.
$results = & 'C:\Program Files\Kubernetes\Minikube\minikube.exe' kubectl -- run probe --rm -i --quiet --restart=Never --image=curlimages/curl -- sh -c $containerCommand

# Extract each model version from the JSON responses and count how often it appeared.
$results | Select-String -AllMatches -Pattern '"model_version":"v[0-9]"' | ForEach-Object { $_.Matches.Value } | Group-Object

```
Expect roughly 90% v1 and 10% v2. Balancing is random, so a few points of variation is normal.

**7. Advance or roll back:**
```powershell
# Scale v1 to five replicas.
minikube kubectl -- scale deployment wine-classifier-v1 --replicas=5

# Scale v2 to five replicas to compare an even split.
minikube kubectl -- scale deployment wine-classifier-v2 --replicas=5

# Roll back by scaling the v2 canary to zero replicas.
minikube kubectl -- scale deployment wine-classifier-v2 --replicas=0

# Delete both Deployments while keeping the Service in place.
minikube kubectl -- delete deployment wine-classifier-v1 wine-classifier-v2

```
Replica ratios only approximate a split. Exact percentages need an ingress controller or service mesh.

## Guardrail Exercise
Use the Part A latency numbers and these guardrails from the Module 3 canary slide:
**p95 may rise by at most +15%**, and error rate by at most +0.10 pp.
1. Compute v2's p95 change versus v1. (With the reference numbers: 11.1 ÷ 4.3 ≈ 2.6x, a +158% increase.)
2. Does v2 pass the latency guardrail? Should it be promoted?
3. v2's accuracy equals v1's on this toy test set. What would you need to justify the extra latency?

## Discussion Questions
1. Why does the Service select only on `app` and not on `version`?
2. What breaks if a canary pod has no readiness probe?
3. Which would you choose for this model: blue-green or canary? Why?
4. How would autoscaling interact with a 9:1 replica split?

### Add minikube in path
# Add Minikube's installation directory to this terminal's PATH.
$env:Path += ';C:\Program Files\Kubernetes\Minikube'
