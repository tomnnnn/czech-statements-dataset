#!/bin/bash
PARAM_FILE="params.txt"

export MODEL=meta-llama/Llama-3.2-3B-Instruct
export DATA="/storage/brno2/home/tomn/czech-statements-dataset"
export LOG_DIR=$DATA/logs
export EXAMPLES_PATH=datasets/examples_demagog.json
TO_REPLACE='$EXAMPLES_PATH,$DATA,$EVIDENCE_PATH,$STATEMENTS_PATH,$PROMPT_PATH,$ALLOWED_LABELS,$MODEL,$BATCH_SIZE,$SAMPLE_PORTION,$MAX_INDEX,$EXAMPLE_COUNT,$NAME,$EXPLANATION'

rm -rf .jobs
mkdir .jobs

# Read each line from the parameter file
while IFS= read -r line || [ -n "$line" ]; do
	# skip commented lines
	if [[ -z "$line" || "$line" =~ ^# ]]; then
		continue
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
