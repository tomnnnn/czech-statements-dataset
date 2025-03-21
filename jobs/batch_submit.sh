#!/bin/bash

PARAM_FILE="params.json"
export DATA="/storage/brno2/home/tomn/czech-statements-dataset"
export LOG_DIR="$DATA/logs"
TO_REPLACE='$OUTPUT,$MODEL,$DATA,$PROMPT_PATH,$ALLOWED_LABELS,$MODEL,$BATCH_SIZE,$SAMPLE_PORTION,$MAX_INDEX,$EXAMPLE_COUNT,$EXPLANATION,$DATASET_PATH,$EVIDENCE_SOURCE,$NAME,$MODEL_API,$MODEL_FILE'

rm -rf .jobs
mkdir .jobs

# Read JSON and process each job
jq -c '.[]' "$PARAM_FILE" | while read -r json_line; do
    # Parse JSON into variables
    eval "$(echo "$json_line" | jq -r 'to_entries | map("export " + .key + "=\"" + (.value | tostring) + "\"") | .[]')"

    JOB_UID=$(tr -dc 'a-zA-Z0-9' < /dev/urandom | fold -w 6 | head -n 1)
    JOB_SCRIPT=".jobs/${NAME}_${JOB_UID}.pbs"

    # Handle PBS_ARRAY_INDEX for single-job case
    if [[ "$MAX_INDEX" -eq 1 ]]; then
        export PBS_ARRAY_INDEX=0
    fi

    # Use envsubst to substitute variables in the template
    envsubst $TO_REPLACE < job_template.pbs > "$JOB_SCRIPT"

    # Construct qsub command
    if [[ "$MAX_INDEX" -eq 1 ]]; then
        queue_cmd="qsub -l walltime=${WALLTIME} -N ${NAME} -o $LOG_DIR ${JOB_SCRIPT}"
    else
        queue_cmd="qsub -J 1-${MAX_INDEX} -l walltime=${WALLTIME} -N ${NAME} -o $LOG_DIR ${JOB_SCRIPT}"
    fi

    echo "$queue_cmd"
    $queue_cmd
done
