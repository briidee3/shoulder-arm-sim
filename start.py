import subprocess

 

def run_script(script_name):
    """ Helper function to run a python script """
    print(f"Running {script_name}...")
    subprocess.run(['python', script_name], check=True)

try:
    # Run the start_up_sequence.py script
    run_script('start_up_sequence.py')

    # If start_up_sequence.py completes successfully, run gui.py
    run_script('gui.py')

except subprocess.CalledProcessError as e:
    print(f"Error occurred while running scripts: {e}")