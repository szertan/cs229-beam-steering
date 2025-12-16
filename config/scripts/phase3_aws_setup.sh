#!/bin/bash
# =============================================================================
# Phase 3: AWS Setup for CS229 Beam Steering ES Training
# =============================================================================
# This script sets up all AWS infrastructure needed for deploying the training
# pipeline. Tasks include:
# - S3 bucket for outputs
# - IAM role for Batch jobs
# - ECR repository for Docker images
# - Batch Compute Environment, Job Queue, Job Definition
# =============================================================================

set -e  # Exit on error

# Configuration
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=${AWS_REGION:-us-west-2}
PROJECT_NAME="cs229-beam-steering"

# Resource names
S3_BUCKET="${PROJECT_NAME}-results"
IAM_ROLE_NAME="${PROJECT_NAME}-batch-job-role"
ECR_REPO_NAME="${PROJECT_NAME}-trainer"
COMPUTE_ENV_NAME="${PROJECT_NAME}-compute-env"
JOB_QUEUE_NAME="${PROJECT_NAME}-queue"
JOB_DEF_NAME="${PROJECT_NAME}-train"

echo "================================================================================"
echo "PHASE 3: AWS INFRASTRUCTURE SETUP"
echo "================================================================================"
echo "Account: $AWS_ACCOUNT_ID"
echo "Region: $AWS_REGION"
echo ""

# =============================================================================
# Task 3.1: Verify AWS CLI Configuration
# =============================================================================
echo "[3.1] Verifying AWS CLI configuration..."
aws sts get-caller-identity
echo "✅ AWS CLI configured"
echo ""

# =============================================================================
# Task 3.2: Create S3 Bucket
# =============================================================================
echo "[3.2] Creating S3 bucket: $S3_BUCKET"
if aws s3 ls "s3://$S3_BUCKET" 2>/dev/null; then
    echo "  ℹ️  Bucket already exists"
else
    if [ "$AWS_REGION" = "us-east-1" ]; then
        aws s3 mb "s3://$S3_BUCKET" --region "$AWS_REGION"
    else
        aws s3 mb "s3://$S3_BUCKET" --region "$AWS_REGION" --create-bucket-configuration LocationConstraint="$AWS_REGION"
    fi
    echo "✅ Created S3 bucket"
fi
echo ""

# =============================================================================
# Task 3.3: Create IAM Role for Batch Jobs
# =============================================================================
echo "[3.3] Creating IAM role: $IAM_ROLE_NAME"

# Create trust policy document
cat > /tmp/batch-trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create or update role
if aws iam get-role --role-name "$IAM_ROLE_NAME" 2>/dev/null; then
    echo "  ℹ️  Role already exists"
else
    aws iam create-role \
        --role-name "$IAM_ROLE_NAME" \
        --assume-role-policy-document file:///tmp/batch-trust-policy.json
    echo "✅ Created IAM role"
fi

# Attach S3 policy
echo "  Attaching S3FullAccess policy..."
aws iam attach-role-policy \
    --role-name "$IAM_ROLE_NAME" \
    --policy-arn "arn:aws:iam::aws:policy/AmazonS3FullAccess"

# Attach CloudWatch logs policy
echo "  Attaching CloudWatch Logs policy..."
aws iam attach-role-policy \
    --role-name "$IAM_ROLE_NAME" \
    --policy-arn "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"

echo "✅ Configured IAM role with S3 and CloudWatch policies"
echo ""

# =============================================================================
# Task 3.4: Create ECR Repository
# =============================================================================
echo "[3.4] Creating ECR repository: $ECR_REPO_NAME"

if aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$AWS_REGION" 2>/dev/null; then
    echo "  ℹ️  Repository already exists"
    ECR_URI=$(aws ecr describe-repositories \
        --repository-names "$ECR_REPO_NAME" \
        --region "$AWS_REGION" \
        --query 'repositories[0].repositoryUri' \
        --output text)
else
    aws ecr create-repository \
        --repository-name "$ECR_REPO_NAME" \
        --region "$AWS_REGION"
    echo "✅ Created ECR repository"
    
    ECR_URI=$(aws ecr describe-repositories \
        --repository-names "$ECR_REPO_NAME" \
        --region "$AWS_REGION" \
        --query 'repositories[0].repositoryUri' \
        --output text)
fi

echo "  ECR URI: $ECR_URI"
echo ""

# =============================================================================
# Task 3.5a: Create Batch Compute Environment
# =============================================================================
echo "[3.5a] Creating Batch Compute Environment: $COMPUTE_ENV_NAME"

# Get default VPC and subnet
DEFAULT_VPC=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query 'Vpcs[0].VpcId' --output text)
DEFAULT_SUBNET=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$DEFAULT_VPC" --query 'Subnets[0].SubnetId' --output text)

# Get default security group
DEFAULT_SG=$(aws ec2 describe-security-groups \
    --filters "Name=vpc-id,Values=$DEFAULT_VPC" "Name=group-name,Values=default" \
    --query 'SecurityGroups[0].GroupId' \
    --output text)

echo "  Default VPC: $DEFAULT_VPC"
echo "  Default Subnet: $DEFAULT_SUBNET"
echo "  Default Security Group: $DEFAULT_SG"
