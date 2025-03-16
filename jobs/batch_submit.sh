#!/bin/bash
PARAM_FILE="params.txt"

export DATA="/storage/brno2/home/tomn/czech-statements-dataset"
export LOG_DIR=$DATA/logs
TO_REPLACE='$MODEL,$DATA,$PROMPT_PATH,$ALLOWED_LABELS,$MODEL,$BATCH_SIZE,$SAMPLE_PORTION,$MAX_INDEX,$EXAMPLE_COUNT,$EXPLANATION,$DATASET_PATH,$EVIDENCE_SOURCE,$NAME,$MODEL_API,$MODEL_FILE'
export LOG_DIR=$DATA/logs

rm -rf .jobs
mkdir .jobs

# Read each line from the parameter file
while IFS= read -r line || [ -n "$line" ]; do
	# skip commented lines
	if [[ -z "$line" || "$line" =~ ^# ]]; then
		continue
	fi

	# if $MODEL_API is empty, set it to the default value
	if [[ -z "$MODEL_API" ]]; then
		export MODEL_API="transformers"
	fi

	# Extract parameter values from the line
	eval export "$line"  # This sets variables dynamically

	# Generate a unique job script name
	JOB_SCRIPT=".jobs/${NAME}.pbs"

	# Substitute variables in the template using 'envsubst'
	envsubst $TO_REPLACE < job_template.pbs > "$JOB_SCRIPT"

	# Run job
	queue_cmd="qsub -J 1-${MAX_INDEX} -l walltime=${WALLTIME} -N ${NAME} -o $LOG_DIR ${JOB_SCRIPT}"
	echo $queue_cmd
	$queue_cmd

done < "$PARAM_FILE"
