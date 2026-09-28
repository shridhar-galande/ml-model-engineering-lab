# Docker Desktop and Minikube Installation (Windows)

This guide sets up Docker Desktop and Minikube on Windows and starts a local Kubernetes cluster using Docker Desktop as the Minikube driver.

## Prerequisites

- Windows 10 or 11, with virtualization enabled.
- Administrator access for installation.
- An internet connection.

On Windows Home, Docker Desktop uses the WSL 2 backend. If prompted during setup, install or update WSL 2 and restart Windows.

## Install Docker Desktop

1. Follow the [Docker Desktop for Windows installation guide](https://docs.docker.com/desktop/setup/install/windows-install/).
2. Launch Docker Desktop and wait until it reports that the engine is running.
3. Make sure Docker Desktop is using Linux containers. In PowerShell, check:

   ```powershell
   docker info --format '{{.OSType}}'
   ```

   The command should print `linux`. If it fails to connect, Docker Desktop's engine is not ready; start or restart Docker Desktop and check again.

## Install Minikube

Use the [official Minikube Windows installation guide](https://minikube.sigs.k8s.io/docs/start/?arch=%2Fwindows%2Fx86-64%2Fstable%2F.exe+download#Ingress), or install it with WinGet in PowerShell:

```powershell
winget install --exact --id Kubernetes.minikube
```

After installation, open a new PowerShell terminal so it loads the updated `PATH`. Verify the command is available:

```powershell
minikube version
```

## Start and Verify the Cluster

With Docker Desktop running and its Linux engine responding, run:

```powershell
minikube start --driver=docker
minikube status
minikube kubectl -- get nodes
```

The status should show the Minikube host, Kubernetes control plane, and `kubelet` as running. The node command should list a node in the `Ready` state.

## Stop or Remove the Cluster

Stop the cluster while keeping its data:

```powershell
minikube stop
```

Delete the cluster and its data:

```powershell
minikube delete
```

## Basic Command Reference

### Docker

Run these in PowerShell. Replace example names, ports, and paths with your own values.

| Command | What it does |
|---|---|
| `docker version` | Show Docker client and engine versions. |
| `docker info` | Show engine details and confirm Docker is running. |
| `docker ps` | List running containers. |
| `docker ps -a` | List all containers, including stopped ones. |
| `docker images` | List local images. |
| `docker pull nginx:latest` | Download an image. |
| `docker run --name web -d -p 8080:80 nginx:latest` | Create and start a container in the background, mapping host port 8080 to container port 80. |
| `docker logs -f web` | Follow a container's logs; press Ctrl+C to stop following. |
| `docker exec -it web sh` | Open a shell inside a running Linux container; type `exit` to leave. |
| `docker stop web` / `docker start web` | Stop or restart a container. |
| `docker rm -f web` | Remove a container, stopping it first if needed. |
| `docker rmi nginx:latest` | Remove a local image that is not in use. |
| `docker build -t my-app:dev .` | Build an image from the Dockerfile in the current directory. |
| `docker inspect web` | Show detailed container configuration. |
| `docker stats` | Show live resource usage for running containers. |
| `docker system df` | Show Docker disk usage. |

`docker system prune` removes unused Docker data. Review its prompt carefully; adding `-a` also removes unused images.

### Minikube

| Command | What it does |
|---|---|
| `minikube start --driver=docker` | Create or start the cluster using Docker Desktop. |
| `minikube status` | Show the cluster component status. |
| `minikube stop` / `minikube start` | Stop or restart the cluster while keeping its data. |
| `minikube pause` / `minikube unpause` | Pause or resume Kubernetes workloads. |
| `minikube delete` | Delete the current cluster and its data. |
| `minikube addons list` | List available addons and their status. |
| `minikube addons enable ingress` | Enable an addon, such as Ingress. |
| `minikube dashboard` | Open the Kubernetes Dashboard. |
| `minikube image load my-app:dev` | Load a locally built image into Minikube. |
| `minikube image build -t my-app:dev .` | Build an image using Minikube's image tooling. |
| `minikube service SERVICE_NAME --url` | Print a URL for accessing a NodePort service. |
| `minikube profile list` | List Minikube profiles (clusters). |

On Windows with the Docker driver, use `minikube service SERVICE_NAME --url` to access a NodePort service rather than relying on the Minikube node IP. For a LoadBalancer service, run `minikube tunnel` in a separate, elevated PowerShell window and leave it running while using the service.

### kubectl

Minikube can run `kubectl` for you, even when it is not installed separately. Prefix Kubernetes commands like this:

```powershell
minikube kubectl -- get nodes
```

The examples below use `kubectl` directly. If the command is unavailable, replace `kubectl` with `minikube kubectl --` (for example, `minikube kubectl -- get pods -A`).

| Command | What it does |
|---|---|
| `kubectl version --client` | Show the kubectl client version. |
| `kubectl cluster-info` | Show Kubernetes control-plane endpoints. |
| `kubectl get nodes` | List cluster nodes and readiness. |
| `kubectl get namespaces` | List namespaces. |
| `kubectl get pods -A` | List pods in all namespaces. |
| `kubectl get deployments,services` | List deployments and services in the current namespace. |
| `kubectl get pods -n default -o wide` | Show pods in a namespace with node and IP details. |
| `kubectl describe pod POD_NAME` | Show detailed pod status and events. |
| `kubectl logs POD_NAME` | Print a pod's logs. |
| `kubectl logs -f deployment/DEPLOYMENT_NAME` | Follow logs from a deployment. |
| `kubectl exec -it POD_NAME -- sh` | Open a shell in a pod that has a shell available. |
| `kubectl apply -f .\manifest.yaml` | Create or update resources from a YAML manifest. |
| `kubectl delete -f .\manifest.yaml` | Delete resources declared in a manifest. |
| `kubectl create deployment nginx --image=nginx` | Create a sample deployment. |
| `kubectl expose deployment nginx --port=80 --type=NodePort` | Create a NodePort service for the deployment. |
| `kubectl scale deployment nginx --replicas=3` | Set the deployment's replica count. |
| `kubectl rollout status deployment/nginx` | Wait for a deployment rollout to complete. |
| `kubectl rollout history deployment/nginx` | Show rollout revisions. |
| `kubectl rollout undo deployment/nginx` | Roll back to the previous deployment revision. |
| `kubectl port-forward service/nginx 8080:80` | Forward local port 8080 to service port 80; leave the command running while using it. |
| `kubectl config current-context` | Show the active Kubernetes context. |

Commands that name a pod or deployment require its real name. Use `kubectl get pods` or `kubectl get deployments` to find names first. Add `-n NAMESPACE` to namespaced commands when the resource is not in the current namespace.