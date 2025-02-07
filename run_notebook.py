import os
import subprocess
import sys

print("Starting notebook execution...")

try:
    # Execute the notebook using jupyter nbconvert
    cmd = ["jupyter", "nbconvert", "--to", "notebook", 
           "--execute", "--inplace", "peft_tutorial.ipynb"]

    # Run the command and capture output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

    # Stream output in real-time
    while True:
        output = process.stdout.readline()
        error = process.stderr.readline()

        if output == '' and error == '' and process.poll() is not None:
            break

        if output:
            print(output.strip())
        if error:
            print(error.strip(), file=sys.stderr)

    # Get the return code
    return_code = process.poll()

    if return_code == 0:
        print("Notebook execution completed successfully!")
    else:
        print(f"Notebook execution failed with return code: {return_code}")
        sys.exit(return_code)

except Exception as e:
    print(f"Error executing notebook: {str(e)}")
    raise