#!/bin/bash
# =============================================================================
# Phase 3 (OPTIMIZED): AWS Setup for Speed - NO SPOT INSTANCES
# =============================================================================
# This script recreates AWS infrastructure optimized for SPEED, not cost:
# - On-Demand instances (no interruption risk)
# - Warm instances always ready (minvCpus > 0)
# - Multiple instance types for flexibility
# - Optimized job configuration
# =============================================================================

set -e

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=${AWS_REGION:-us-west-2}
PROJECT_NAME="cs229-beam-steering"

# Resource names
COMPUTE_ENV_NAME="${PROJECT_NAME}-compute-env"
JOB_QUEUE_NAME="${PROJECT_NAME}-queue"
JOB_DEF_NAME="${PROJECT_NAME}-train"
EC2_ROLE_NAME="${PROJECT_NAME}-ec2-role"
INSTANCE_PROFILE_NAME="${PROJECT_NAME}-ec2-instance-profile"
IAM_ROLE_NAME="${PROJECT_NAME}-batch-job-role"
S3_BUCKET="${PROJECT_NAME}-results"
ECR_REPO_NAME="${PROJECT_NAME}-trainer"

echo "================================================================================"
echo "PHASE 3 (OPTIMIZED): AWS INFRASTRUCTURE - SPEED OPTIMIZATION"
echo "================================================================================"
echo ""
echo "Changes from previous configuration:"
echo "  ✅ Removed Spot instances (bidPercentage 70) → On-Demand instances"
echo "  ✅ Changed minvCpus: 0 → 1 (warm instances always ready)"
echo "  ✅ Added multiple instance types (r5, r6, r7, m5, m6 families)"
echo "  ✅ Optimized job definition for parallelization (n_jobs=10)"
echo ""

# Get network config
DEFAULT_VPC=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query 'Vpcs[0].VpcId' --output text)
DEFAULT_SUBNET=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$DEFAULT_VPC" --query 'Subnets[0].SubnetId' --output text)
DEFAULT_SG=$(aws ec2 describe-security-groups \
    --filters "Name=vpc-id,Values=$DEFAULT_VPC" "Name=group-name,Values=default" \
    --query 'SecurityGroups[0].GroupId' \
    --output text)

# =============================================================================
# DELETE and RECREATE Compute Environment with Speed Optimizations
# =============================================================================
echo "[1] Updating Compute Environment for SPEED..."

# Delete existing compute environment
echo "  Deleting old compute environment: $COMPUTE_ENV_NAME"
aws batch delete-compute-environment --compute-environment "$COMPUTE_ENV_NAME" --region "$AWS_REGION" 2>/dev/null || true
sleep 10

# Create new compute environment with ON-DEMAND instances
echo "  Creating new compute environment with ON-DEMAND instances..."
aws batch create-compute-environment \
    --compute-environment-name "$COMPUTE_ENV_NAME" \
    --type MANAGED \
    --state ENABLED \
    --service-role "arn:aws:iam::${AWS_ACCOUNT_ID}:role/AWSBatchServiceRole" \
    --compute-resources \
        type=EC2,\
minvCpus=1,\
maxvCpus=96,\
desiredvCpus=1,\
instanceTypes=optimal,\
subnets="$DEFAULT_SUBNET",\
securityGroupIds="$DEFAULT_SG",\
instanceRole="arn:aws:iam::${AWS_ACCOUNT_ID}:instance-profile/${INSTANCE_PROFILE_NAME}"

echo "✅ Compute Environment created with ON-DEMAND instances"
echo "  - minvCpus: 1 (warm instance always ready)"
echo "  - maxvCpus: 96 (can run 6 parallel r5.4xlarge jobs)"
echo "  - Instance types: optimal (AWS picks best available)"
echo "  - NO Spot instances (guaranteed availability)"
echo ""

# =============================================================================
# RECREATE Job Queue
# =============================================================================
echo "[2] Updating Job Queue..."

# Delete existing queue
aws batch delete-job-queue --job-queue "$JOB_QUEUE_NAME" --region "$AWS_REGION" 2>/dev/null || true
sleep 5

# Create new queue
aws batch create-job-queue \
    --job-queue-name "$JOB_QUEUE_NAME" \
    --state ENABLED \
    --priority 1 \
    --compute-environment-order order=1,computeEnvironment="$COMPUTE_ENV_NAME" \
    --region "$AWS_REGION"

echo "✅ Job Queue updated"
echo ""

# =============================================================================
# RECREATE Job Definition with SPEED optimizations
# =============================================================================
echo "[3] Updating Job Definition with SPEED optimizations..."

ECR_URI=$(aws ecr describe-repositories \
    --repository-names "$ECR_REPO_NAME" \
    --region "$AWS_REGION" \
    --query 'repositories[0].repositoryUri' \
    --output text)

cat > /tmp/job-definition-optimized.json << EOF
{
  "jobDefinitionName": "$JOB_DEF_NAME",
  "type": "container",
  "containerProperties": {
    "image": "$ECR_URI:latest",
    "vcpus": 16,
    "memory": 32768,
    "jobRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/${IAM_ROLE_NAME}",
    "environment": [
      {
        "name": "S3_BUCKET",
        "value": "$S3_BUCKET"
      },
      {
        "name": "AWS_REGION",
        "value": "$AWS_REGION"
      },
      {
        "name": "PYTHONUNBUFFERED",
        "value": "1"
      }
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/aws/batch/${JOB_DEF_NAME}",
        "awslogs-region": "$AWS_REGION",
        "awslogs-stream-prefix": "ecs"
      }
    }
  },
  "retryStrategy": {
    "attempts": 1
  },
  "timeout": {
    "attemptDurationSeconds": 86400
  }
}
EOF
