#!/bin/bash
# Monitor AWS Batch training jobs every 20 seconds

# ES-Single jobs (0deg, 90deg, 180deg) + ES-Multi (parallel v2) + ES-NN full
JOB_IDS="4c77510c-dc9a-4f79-b535-575de376a466 265bddcd-dc2c-43bd-bb21-a76610584cec 5188eeef-9298-45d4-9f16-7797feacb9a9 1dda00bc-0414-4e5d-a893-18ccc1e059f0 50c08eb7-35c7-4bfb-86df-fed677a46448"
REGION="us-west-2"
S3_BUCKET="cs229-beam-steering-results"

echo "=========================================="
echo "AWS BATCH JOB MONITOR"
echo "Started: $(date)"
echo "=========================================="
echo ""

while true; do
    echo "--- $(date) ---"
    
    # Get job statuses
    aws batch describe-jobs --jobs $JOB_IDS --region $REGION \
        --query 'jobs[*].{name:jobName,status:status}' \
        --output table 2>/dev/null
    
    # Count checkpoints in S3
    echo ""
    echo "S3 Progress (checkpoints):"
    for prefix in 0deg 90deg 180deg multi nn; do
        count=$(aws s3 ls s3://$S3_BUCKET/ --recursive 2>/dev/null | grep "${prefix}_" | grep "checkpoint_" | grep "metadata.json" | wc -l | tr -d ' ')
        latest=$(aws s3 ls s3://$S3_BUCKET/ --recursive 2>/dev/null | grep "${prefix}_" | grep "checkpoint_" | grep "metadata.json" | tail -1 | awk '{print $4}' | sed 's/.*checkpoint_/checkpoint_/' | sed 's/\/.*//')
        if [[ "$count" -gt 0 ]]; then
            echo "  $prefix: $count (latest: ${latest:-none})"
        fi
    done
    
    # Show current iteration from logs
    echo ""
    echo "Current Iteration (from logs):"
    for job_id in $JOB_IDS; do
        job_name=$(aws batch describe-jobs --jobs $job_id --region $REGION --query 'jobs[0].jobName' --output text 2>/dev/null)
        status=$(aws batch describe-jobs --jobs $job_id --region $REGION --query 'jobs[0].status' --output text 2>/dev/null)
        log_stream=$(aws batch describe-jobs --jobs $job_id --region $REGION --query 'jobs[0].container.logStreamName' --output text 2>/dev/null)
        
        if [[ "$status" == "SUCCEEDED" ]] || [[ "$status" == "FAILED" ]]; then
            echo "  $job_name: $status"
        elif [[ "$log_stream" != "None" ]] && [[ -n "$log_stream" ]]; then
            # Try to get iteration from log - handle both /1000 and /10 formats
            iteration=$(aws logs get-log-events --log-group-name /aws/batch/job --log-stream-name "$log_stream" --region $REGION --limit 20 --query 'events[*].message' --output text 2>/dev/null | grep -oE '\[\s*[0-9]+/[0-9]+\]' | tail -1 | tr -d '[] ')
            if [[ -n "$iteration" ]]; then
                echo "  $job_name: $iteration"
            else
                echo "  $job_name: running (no iteration yet)"
            fi
        else
            echo "  $job_name: $status"
        fi
    done
    
    # Check if all jobs are done
    statuses=$(aws batch describe-jobs --jobs $JOB_IDS --region $REGION --query 'jobs[*].status' --output text 2>/dev/null)
    if [[ ! "$statuses" =~ "RUNNABLE" ]] && [[ ! "$statuses" =~ "STARTING" ]] && [[ ! "$statuses" =~ "RUNNING" ]] && [[ ! "$statuses" =~ "SUBMITTED" ]]; then
        echo ""
        echo "=========================================="
        echo "ALL JOBS COMPLETED at $(date)"
        echo "Final statuses: $statuses"
        echo "=========================================="
        break
    fi
    
    echo ""
    echo "(Next update in 20 seconds... Ctrl+C to stop)"
    echo ""
    sleep 20
done
