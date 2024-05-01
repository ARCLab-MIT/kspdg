import sys
import os
import openai
import time
from dotenv import load_dotenv

if __name__ == '__main__':
    # Load configuration from .env
    load_dotenv()
    openai.api_key = os.environ['OPENAI_API_KEY']
#    openai.api_key = os.environ['OPENAI_API_KEY_MITM']

    if len(sys.argv) < 2:
        print("Use: python check_fine_tuning_job.py <job_id>")
        sys.exit(1)

    # Retrieve job
    jobId = sys.argv[1]
    job = openai.FineTuningJob.retrieve(jobId)
    print (job)

