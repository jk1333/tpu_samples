#!/bin/bash

# ==============================================================================
# 설정 변수
# ==============================================================================
# 프로젝트 ID 자동 가져오기
PROJECT_ID=$(gcloud config get-value project)
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")

# Cloud Storage 설정
BUCKET_NAME="gs://${PROJECT_ID}"  # 프로젝트 ID를 버킷 이름으로 사용
BUCKET_REGION="us-central1"       # 버킷 생성 리전

# TPU 설정
TPU_NAME="my-tpu-spot-vm"
TPU_CREATED=false

# Workbench 설정
WB_NAME="my-workbench-g2"
WB_ZONE="us-central1-a"
WB_MACHINE_TYPE="g2-standard-4" # NVIDIA L4 1장이 포함된 머신 타입

# GKE 설정 (vLLM용)
GKE_CLUSTER_NAME="vllm-cluster"
GKE_ZONE="us-central1-a"
GKE_MACHINE_TYPE="g2-standard-16"
GKE_ACCELERATOR="type=nvidia-l4,count=1,gpu-driver-version=LATEST"
GKE_NUM_NODES=1

# V6e 시도할 리전 목록
REGIONS_V6E=(
    "us-central1-b"
    "us-east1-d"
    "us-east5-a"
    "us-east5-b"
    "europe-west4-a"
    "asia-northeast1-b"
    "southamerica-west1-a"
)

# V5e 시도할 리전 목록
REGIONS_V5E=(
    "us-central1-a"
    "us-south1-a"
    "us-west1-c"
    "us-west4-a"
    "europe-west4-b"
)

# ==============================================================================
# 0. 필수 API 활성화 및 IAM 권한 설정
# ==============================================================================
echo "----------------------------------------------------------------"
echo "Enabling APIs and Setting Permissions..."
echo "----------------------------------------------------------------"

# API 활성화 (GKE용 container.googleapis.com 추가)
gcloud services enable tpu.googleapis.com \
    notebooks.googleapis.com \
    compute.googleapis.com \
    aiplatform.googleapis.com \
    iam.googleapis.com \
    container.googleapis.com

# Default Compute Engine Service Account 가져오기
DEFAULT_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
echo "Default Service Account: $DEFAULT_SA"

# Storage Owner(Admin) 권한 부여
echo "Granting Storage Admin role to Default Service Account..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${DEFAULT_SA}" \
    --role="roles/storage.admin" \
    --condition=None \
    --quiet

echo "✅ IAM permissions updated."

# ==============================================================================
# 1. Cloud Storage 버킷 생성 (NEW)
# ==============================================================================
echo "----------------------------------------------------------------"
echo "🪣  Creating Cloud Storage Bucket ($BUCKET_NAME)..."
echo "----------------------------------------------------------------"

# 버킷 존재 여부 확인 (gcloud storage 명령어 사용)
if gcloud storage buckets describe $BUCKET_NAME --project=$PROJECT_ID >/dev/null 2>&1; then
    echo "ℹ️  Bucket '$BUCKET_NAME' already exists. Skipping."
else
    # 버킷 생성 (uniform-bucket-level-access 권장)
    gcloud storage buckets create $BUCKET_NAME \
        --project=$PROJECT_ID \
        --location=$BUCKET_REGION \
        --uniform-bucket-level-access \
        --quiet

    if [ $? -eq 0 ]; then
        echo "✅ SUCCESS: Bucket '$BUCKET_NAME' created."
    else
        echo "❌ FAILED: Failed to create bucket."
        exit 1
    fi
fi

# ==============================================================================
# 2. Cloud TPU VM 생성 (재시도 로직 포함)
# ==============================================================================

# TPU 생성 함수
try_create_tpu() {
    local ZONE=$1
    local TYPE=$2
    local VERSION=$3

    echo "----------------------------------------------------------------"
    echo "Trying to create TPU SPOT VM in $ZONE ($TYPE)..."
    echo "----------------------------------------------------------------"

    gcloud compute tpus tpu-vm create $TPU_NAME \
        --zone=$ZONE \
        --accelerator-type=$TYPE \
        --version=$VERSION \
        --spot \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --project=$PROJECT_ID \
        --quiet

    if [ $? -eq 0 ]; then
        echo "✅ SUCCESS: TPU VM '$TPU_NAME' created in $ZONE ($TYPE)!"
        return 0
    else
        echo "❌ FAILED: Could not create in $ZONE. Cleaning up..."
        gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE --project=$PROJECT_ID --quiet 2>/dev/null
        return 1
    fi
}

# TPU 생성 루프
while [ "$TPU_CREATED" = false ]; do
    
    # [우선순위 1] V6e 리전 순회
    for ZONE in "${REGIONS_V6E[@]}"; do
        try_create_tpu "$ZONE" "v6e-1" "v2-alpha-tpuv6e"
        if [ $? -eq 0 ]; then TPU_CREATED=true; break 2; fi
    done

    # [우선순위 2] V5e 리전 순회
    if [ "$TPU_CREATED" = false ]; then
        for ZONE in "${REGIONS_V5E[@]}"; do
            try_create_tpu "$ZONE" "v5e-1" "v2-alpha-tpuv5-lite"
            if [ $? -eq 0 ]; then TPU_CREATED=true; break 2; fi
        done
    fi

    if [ "$TPU_CREATED" = false ]; then
        echo "⚠️  All TPU regions failed. Retrying in 10 seconds..."
        sleep 10
    fi
done

# ==============================================================================
# 3. Vertex AI Workbench 생성
# ==============================================================================
echo "----------------------------------------------------------------"
echo "Creating Vertex AI Workbench Instance ($WB_NAME)..."
echo "----------------------------------------------------------------"

# Workbench 존재 여부 확인
gcloud workbench instances describe $WB_NAME --location=$WB_ZONE --project=$PROJECT_ID >/dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "ℹ️  Workbench '$WB_NAME' already exists. Skipping creation."
else
    # [수정] --boot-disk-type 옵션 추가 (G2 인스턴스 필수)
    gcloud workbench instances create $WB_NAME \
        --project=$PROJECT_ID \
        --location=$WB_ZONE \
        --machine-type=$WB_MACHINE_TYPE \
        --boot-disk-type=PD_BALANCED \
        --boot-disk-size=150 \
        --data-disk-size=100 \
        --data-disk-type=PD_BALANCED \
        --install-gpu-driver \
        --quiet

    if [ $? -eq 0 ]; then
        echo "✅ SUCCESS: Workbench Instance '$WB_NAME' created."
    else
        echo "❌ FAILED: Failed to create Workbench Instance."
        exit 1
    fi
fi

# ==============================================================================
# 4. GKE 클러스터 생성 (vLLM)
# ==============================================================================
echo "----------------------------------------------------------------"
echo "Creating GKE Cluster ($GKE_CLUSTER_NAME)..."
echo "----------------------------------------------------------------"

# 클러스터 존재 여부 확인
if gcloud container clusters describe $GKE_CLUSTER_NAME --zone=$GKE_ZONE --project=$PROJECT_ID >/dev/null 2>&1; then
    echo "ℹ️  GKE Cluster '$GKE_CLUSTER_NAME' already exists. Skipping creation."
else
    gcloud container clusters create $GKE_CLUSTER_NAME \
        --project=$PROJECT_ID \
        --zone=$GKE_ZONE \
        --machine-type=$GKE_MACHINE_TYPE \
        --accelerator=$GKE_ACCELERATOR \
        --num-nodes=$GKE_NUM_NODES \
        --quiet

    if [ $? -eq 0 ]; then
        echo "✅ SUCCESS: GKE Cluster '$GKE_CLUSTER_NAME' created."
    else
        echo "❌ FAILED: Failed to create GKE Cluster."
        exit 1
    fi
fi

# ==============================================================================
# 5. 방화벽 규칙 및 마무리
# ==============================================================================
echo "----------------------------------------------------------------"
echo "Finalizing Network Settings..."
echo "----------------------------------------------------------------"

gcloud compute firewall-rules describe jupyter --project=$PROJECT_ID >/dev/null 2>&1
if [ $? -ne 0 ]; then
    gcloud compute --project=$PROJECT_ID firewall-rules create jupyter \
        --direction=INGRESS \
        --priority=1000 \
        --network=default \
        --action=ALLOW \
        --rules=tcp:8080 \
        --source-ranges=0.0.0.0/0
fi

echo "🎉 All Done! TPU, Workbench, and GKE setup complete."