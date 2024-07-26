import argparse
import re
import subprocess
import h5py
import numpy as np
import datetime
from scipy import stats
import matplotlib.pyplot as plt

# Constants
ENERGY = 1  # Incident energy
ENERGY_UNIT = "GeV"
EVENTS = 10000  # Number of events per simulation
FILE_PATH = "./common_settings.mac"
EXAMPLE_PAR04_PATH = "./examplePar04.mac"


def run_linux_command_verbose(command):
    """Run Linux command and print verbose output."""
    try:
        result = subprocess.run(command, shell=True, check=True, universal_newlines=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command '{command}': {e}")

def run_linux_command(command):
    """Run Linux command."""
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        print("Output:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command '{command}': {e}")

def modify_mac_file(file_path, params, energy, energy_unit):
    """Modify the given .mac file."""
    patterns = {
        "setSizeOfZCells": r"/Par04/mesh/setSizeOfZCells (\d+\.\d+ \w+)",
        "setNbOfRhoCells": r"/Par04/mesh/setNbOfRhoCells (\d+)",
        "setSizeOfRhoCells": r"/Par04/mesh/setSizeOfRhoCells (\d+\.\d+ \w+)",
        "setNbOfPhiCells": r"/Par04/mesh/setNbOfPhiCells (\d+)",
        "setNbOfZCells": r"/Par04/mesh/setNbOfZCells (\d+)"
    }
    with open(file_path, 'r') as file:
        content = file.read()
    for param, value in params.items():
        pattern = patterns.get(param)
        if pattern:
            content = re.sub(pattern, fr"/Par04/mesh/{param} {value}", content)
    content = re.sub(r"/gun/energy (\d+) (\w+)", fr"/gun/energy {energy} {energy_unit}", content)
    with open(file_path, 'w') as file:
        file.write(content)

def filter_array(arr, min_val, max_val):
    """Filter array based on minimum and maximum values."""
    filtered_arr = arr[(arr >= min_val) & (arr <= max_val)]
    return filtered_arr

def calculate_energy_loss(hdf5_file, energy, events):
    """Calculate energy loss."""
    energy_losses = []
    for i in range(events):
        energy_loss = (energy * 1000 - np.sum(hdf5_file['showers'][i])) * 100 / (energy * 1000)
        energy_losses.append(energy_loss)
    energy_losses = np.array(energy_losses)
    mu, sigma = stats.norm.fit(energy_losses)
    plt.hist(energy_losses, bins=100, density=True, alpha=0.6, color='g', label='Original Distribution')
    x = np.linspace(min(energy_losses), max(energy_losses), 100)
    p = stats.norm.pdf(x, mu, sigma)
    plt.plot(x, p, 'k', linewidth=2, label='First Gaussian (mean={:.4f}, std={:.4f})'.format(mu, sigma))
    filtered_energy_losses = filter_array(energy_losses, mu - 2 * sigma, mu + 2 * sigma)
    mu_filtered, sigma_filtered = stats.norm.fit(filtered_energy_losses)
    x_filtered = np.linspace((mu - 2 * sigma), (mu + 2 * sigma), filtered_energy_losses.size)
    p_filtered = stats.norm.pdf(x_filtered, mu_filtered, sigma_filtered)
    plt.plot(x_filtered, p_filtered, 'r--', linewidth=2, label='Second Gaussian (mean={:.4f}, std={:.4f})'.format(mu_filtered, sigma_filtered))
    plt.xlabel('Energy Losses (%)')
    plt.ylabel('Probability Density')
    plt.title('Energy Losses Distribution')
    plt.legend()
    plt.grid(True)
    plt.savefig('energy_losses_distribution.png')
    return mu_filtered

def edit_examplePar04(examplePar04_path, energy, events, current_datetime, energy_unit):
    """Edit examplePar04.mac file."""
    with open(examplePar04_path, 'r') as f:
        content = f.read()
    file_name = f"{energy}{energy_unit}_{events}events_fullsim_{current_datetime}.root"
    content = re.sub(r'/analysis/setFileName\s+\S+\.root', f'/analysis/setFileName {file_name}', content)
    content = re.sub(r'/run/beamOn\s+\d+', f'/run/beamOn {events}', content)
    with open(examplePar04_path, 'w') as f:
        f.write(content)

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--sizeOfZCells', type=str, help='Size of Z Cells')
    parser.add_argument('--nbOfRhoCells', type=int, help='Number of Rho Cells')
    parser.add_argument('--sizeOfRhoCells', type=str, help='Size of Rho Cells')
    parser.add_argument('--nbOfPhiCells', type=int, help='Number of Phi Cells')
    parser.add_argument('--nbOfZCells', type=int, help='Number of Z Cells')
    args = parser.parse_args()

    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    sizeOfZCells_mm = args.sizeOfZCells + " mm" if args.sizeOfZCells else None
    sizeOfRhoCells_mm = args.sizeOfRhoCells + " mm" if args.sizeOfRhoCells else None
    parameters_to_change = {
        "setSizeOfZCells": sizeOfZCells_mm,
        "setNbOfRhoCells": args.nbOfRhoCells,
        "setSizeOfRhoCells": sizeOfRhoCells_mm,
        "setNbOfPhiCells": args.nbOfPhiCells,
        "setNbOfZCells": args.nbOfZCells
    }

    modify_mac_file(FILE_PATH, parameters_to_change, ENERGY, ENERGY_UNIT)
    edit_examplePar04(EXAMPLE_PAR04_PATH, ENERGY, EVENTS, current_datetime, ENERGY_UNIT)
    command = "./examplePar04 -m examplePar04.mac"
    run_linux_command(command)
    run_linux_command_verbose(command)
    file_name = f"{ENERGY}{ENERGY_UNIT}_{EVENTS}events_fullsim_{current_datetime}"
    command = f"python3 ../training/root2h5.py --inputFile {file_name}.root --numR {args.nbOfRhoCells} --numZ {args.nbOfZCells} --numPhi {args.nbOfPhiCells}" 
    run_linux_command(command)
    run_linux_command_verbose(command)
    print("Running: energy calculation")
    file_path = f"./{file_name}.h5"
    hdf5_file = h5py.File(file_path, 'r')
    print("Incident energy: ", hdf5_file['incident_energy'][0], " Energy conserved: ", np.sum(hdf5_file['showers'][0]))
    energy_value = ENERGY / 1000 if ENERGY_UNIT == "MeV" else ENERGY
    average_energy_loss = calculate_energy_loss(hdf5_file, energy_value, EVENTS)
    #command = f"rm ./{file_name}.h5 && rm ./{file_name}.root"
    #run_linux_command(command)
    #print("Deleted .root & .h5 files")
    print("Energy Lost:", average_energy_loss)

if __name__ == "__main__":
    main()
