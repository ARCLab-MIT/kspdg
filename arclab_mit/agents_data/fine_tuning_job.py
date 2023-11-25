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
        print("Use: python fine_tuning_job.py <training_file> [<validation_file>]")
        sys.exit(1)

    # Upload training file
    training_filename = sys.argv[1]
    train_full_response_file = openai.File.create(
        file=open(training_filename,'rb'),
        purpose='fine-tune'
    )
    print(f'training file id: {train_full_response_file.id}')

    # Upload validation file
    validation_file_id = None
    if (len(sys.argv) > 2):
        validation_filename = sys.argv[2]
        validation_full_response_file = openai.File.create(
            file=open(validation_filename,'rb'),
            purpose='fine-tune'
        )
        validation_file_id = validation_full_response_file.id
        print(f'validation file id: {validation_file_id}')

    # Create a fine-tuning job
    response = openai.FineTuningJob.create(training_file=train_full_response_file.id,
                                           validation_file=validation_file_id,
                                           model="gpt-3.5-turbo-1106",
                                           suffix='KSPGPT',
#                                           hyperparameters={'n_epochs': '1'})
                                           hyperparameters={'n_epochs': 'auto'})
    jobId = response.id

    print ("Created fine tunning job: " + jobId)
    print (response)

    with open('result_' + jobId + '.txt', 'w') as result:
        result.write ("Created fine tunning job:")
        result.write (str(response))
        status_completion_list = ['succeeded', 'failed', 'cancelled']
        while True:
            job = openai.FineTuningJob.retrieve(jobId)
            if job.status in status_completion_list:
                print ("\nJob terminated")
                print (job)
                result.write ("\n\nJob result:")
                result.write (str(job))
                break
            time.sleep(10)

