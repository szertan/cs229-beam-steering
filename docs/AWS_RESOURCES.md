================================================================================
AWS RESOURCES FOR CS229 BEAM STEERING ES TRAINING
================================================================================
Created: 2025-12-13

ACCOUNT INFORMATION:
  AWS Account ID: 745854319073
  Region: us-west-2

S3 BUCKET:
  Name: cs229-beam-steering-results
  Purpose: Store training outputs (best_rho.npy, metadata.json, etc.)

IAM ROLES:
  Job Execution Role: cs229-beam-steering-batch-job-role
    - Permissions: AmazonS3FullAccess, CloudWatchLogsFullAccess
    - Used by: Batch jobs (ECS tasks)
  
  EC2 Role: cs229-beam-steering-ec2-role
    - Permissions: AmazonEC2ContainerServiceforEC2Role
    - Used by: EC2 instances in Compute Environment

ECR REPOSITORY:
  Name: cs229-beam-steering-trainer
  URI: 745854319073.dkr.ecr.us-west-2.amazonaws.com/cs229-beam-steering-trainer
  Purpose: Store Docker image for Batch jobs

AWS BATCH INFRASTRUCTURE:
  Compute Environment: cs229-beam-steering-compute-env
    - Type: Managed EC2
    - Instance Type: r5.4xlarge (16 vCPU, 128 GB RAM)
    - Min vCPU: 0 (scales down when no jobs)
    - Max vCPU: 64 (enough for multiple parallel jobs)
    - Spot Instances: Enabled (70% cost savings)
  
  Job Queue: cs229-beam-steering-queue
    - Priority: 1
    - Compute Environment: cs229-beam-steering-compute-env
  
  Job Definition: cs229-beam-steering-train (revision 1)
    - Image: 745854319073.dkr.ecr.us-west-2.amazonaws.com/cs229-beam-steering-trainer:latest
    - vCPU: 16 (one r5.4xlarge has 16 vCPU, so 1 job per instance)
    - Memory: 32768 MB (32 GB, typical for r5.4xlarge)
    - Job Role: cs229-beam-steering-batch-job-role
    - Environment:
      - S3_BUCKET: cs229-beam-steering-results
      - AWS_REGION: us-west-2

NEXT STEPS (PHASE 4: Push Docker to ECR):
  1. Tag the local Docker image with ECR URI
  2. Login to ECR
  3. Push image to ECR
  
Commands for Phase 4:
  ECR_URI="745854319073.dkr.ecr.us-west-2.amazonaws.com/cs229-beam-steering-trainer"
  docker tag cs229-trainer:latest $ECR_URI:latest
  aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ECR_URI
  docker push $ECR_URI:latest

================================================================================
