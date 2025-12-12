"""
Imports and processes data from SPRKKR outputs for DOS calculations and Plotting

Author: David Redka, Nicolas Piwek
Date: 2025.12.10
"""
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import re
import pandas as pd

#CONSTANTS
# Map orbital quantum numbers to their respective labels (s, p, d, etc.)
BANDS = {1: 's', 2: 'p', 3: 'd', 4: 'f', 5: 'g', 6: 'h', 7: 'i'}
DEFAULT_COMB=True
# Adjust these limits to zoom in/out of your plot by default
XLIM = (-10, 5)   # Energy range (eV)
TITLE_FONT_SIZE = 12
LABEL_FONT_SIZE = 10

# Your preferred colors
COLORS = ['k', 'r', 'b', 'g', 'm', 'c', 'orange', 'purple', 'brown', 'gray']

#SPRKKR_DOS_parser.py
def file_select()->str:
    root = tk.Tk()
    root.withdraw()

    file_filters=[
        ("DOS Files", "*.dos"),
        ("All Files", "*.*")
    ]

    fname = filedialog.askopenfilename(
        initialdir=".", 
        title="Select a file",
        filetypes=file_filters
    )

    if fname:
        print(f"Selected file: {fname}")
    else:
        print("No file selected. Exiting.")
        exit()

    return fname

def load_data(fname:str)->list:
    with open(Path(fname)) as file:
        file_data=file.readlines()
    print("Reading lines from file and prepare parsing...")

    # Find the line where the DOS data starts, marked by 'DOS-FMT:  OLD-SPRKKR'
    # This determines where the header ends and raw data begins
    index = next(i for i, line in enumerate(file_data) if 'DOS-FMT:  OLD-SPRKKR' in line)

    # Split the file into header and raw data
    # Header contains metadata; raw data holds the numerical DOS information
    data_header, data_raw = file_data[:index], file_data[index+1:]
    return data_header, data_raw

def header_parser(header):
    # Extract the number of energy points (NE) from the header
    # Look for the line starting with 'NE' and extract the number that follows
    NE = int(next(line.split()[1] for line in header if line.startswith("NE")))
    print(f"Number of energy points (NE): {NE}")

    # Extract all data from header for further parsing
    # Define a helper function to extract values from the header based on keywords
    def extract_header_value(keyword, data_type):
        line = next((line for line in header if keyword in line), None)
        if line:
            return data_type(line.split()[1])
        return None

    # Extract specific values from the header
    NQ_eff = extract_header_value('NQ_eff', int)
    print("NQ_eff:", NQ_eff, "--> Number of different atoms")

    NT_eff = extract_header_value('NT_eff', int)
    print("NT_eff:", NT_eff, "--> Number of different elements")

    EFERMI = extract_header_value('EFERMI', float)
    print("EFERMI:", EFERMI, "--> Fermi energy")

    IREL = extract_header_value('IREL', int)
    print("IREL:", IREL, "--> ???")

    # Locate lines containing 'IQ NLQ' and 'IT TXT_T CONC NAT IQAT'
    iq_nlq_line = next((i for i, line in enumerate(header) if 'IQ' in line and 'NLQ' in line), None)
    it_txt_line = next((i for i, line in enumerate(header) if 'IT' in line and 'TXT_T' in line), None)

    # Extract lines between 'IQ NLQ' and 'IT TXT_T CONC NAT IQAT' into separate 1D arrays
    if iq_nlq_line is not None and it_txt_line is not None and iq_nlq_line < it_txt_line:
        iq_data = []
        nlq_data = []
        for line in header[iq_nlq_line + 1:it_txt_line]:
            if line.strip():  # Ignore empty lines
                iq, nlq = map(int, line.split())  # Split and convert to integers
                iq_data.append(iq)
                nlq_data.append(nlq)
        IQ = np.array(iq_data, dtype=int)
        NLQ = np.array(nlq_data, dtype=int)

        print("IQ:", IQ, "--> IDs of different atoms")
        if (IQ[-1] / NQ_eff) == 1:
            print("IQ_eff and IQ are consistent")
        else:
            print("Please check, IQ_eff and IQ are not consistent!")
        print("NLQ:", NLQ, "--> Number of bands for atoms IQ")
        print("Highest band is:", BANDS[NLQ[-1]])
    else:
        IQ = NLQ = None
        print("Error! Unable to extract IQ and NLQ data...")

    # Extract lines between 'IT TXT_T CONC NAT IQAT' and end of header into structured arrays
    if it_txt_line is not None:
        it_txt_data = [
            line.split()  # Split each line into components
            for line in header[it_txt_line + 1:]  # Lines from IT TXT_T onward
            if line.strip()  # Ignore empty lines
        ]

        # Parse components into separate arrays
        IT = np.array([int(entry[0]) for entry in it_txt_data], dtype=int)
        TXT_T = np.array([entry[1] for entry in it_txt_data], dtype=str)
        CONC = np.array([float(entry[2]) for entry in it_txt_data], dtype=float)
        NAT = np.array([int(entry[3]) for entry in it_txt_data], dtype=int)
        IQAT = np.array([int(entry[4]) for entry in it_txt_data], dtype=int)

        print("IT:", IT, "--> IDs of different elements (including same site)")
        print("TXT_T:", TXT_T, "--> Types of different elements (including same site)")
        print("CONC:", CONC, "--> Concentrations of different elements (including same site)")
        print("NAT:", NAT, "--> ???")
        print("IQAT:", IQAT, "--> Atom ID of different elements")
    else:
        IT = TXT_T = CONC = NAT = IQAT = None
        print("Unable to extract IT, TXT_T, CONC, NAT, IQAT data.")

    # Add initial columns "E" and "???"
    header_new = []
    header_new.append("E")
    header_new.append("???")

    for n in range(np.size(IQAT)):
        print("Element number: ", n + 1)
        # Find index of corresponding NLQ for the current IQAT
        index_NLQ = np.where(IQ == IQAT[n])[0][0]
        print("Atom number: ", IT[index_NLQ])
        print("Lmax:", NLQ[index_NLQ])

        for m in range(2):  # Loop for spin states (up and down)
            spin = "up" if m == 0 else "dn"

            for l in range(NLQ[index_NLQ]):  # Loop over angular momentum quantum numbers
                label = f"{TXT_T[n]} {BANDS[l+1]} {spin}"
                print(label)  # this string should go into the header array
                header_new.append(label)

        if n == np.size(IQAT) - 1:
            print("\nDone...")
        else:
            print("Next element... \n")

    HEADER = np.array(header_new, dtype=str)

    return HEADER, NE, NQ_eff, NT_eff, EFERMI, IREL, IT, TXT_T, CONC, NAT, IQAT, NLQ, IQ

def data_parser(data, NE, IT, NLQ):
    # Calculate the number of lines per block in the raw data
    # This helps in identifying the structure of the data blocks
    block_lines = len(data) // NE

    # Parse all blocks and store the data in a 2D NumPy array
    # Each block corresponds to one energy point, with multiple associated values
    data_parsed = np.array([
        [
            float(line[i:i+10].strip())  # Extract fixed-width values (10 characters per value)
            for line in data[block_start:block_start + block_lines]  # Loop through lines in the block
            for i in range(0, len(line), 10)  # Extract values in chunks of 10 characters
            if line[i:i+10].strip()  # Ignore empty or whitespace-only chunks
        ]
        for block_start in range(0, len(data), block_lines)  # Iterate through all blocks
    ])
    
    # Check if reading DOS data and header infos are consistent
    if (np.size(data_parsed, 1) - 2 - IT[-1] * NLQ[-1] * 2) == 0:
        print('DOS data and header infos are consistent. Start parsing...')
    else:
        print('Please check data, seems to be wrong. Parsing will be broken...')

    return data_parsed

# Helper function to get valid input from the user for unit conversion
def get_valid_input(prompt, valid_options):
    while True:
        user_input = input(prompt).strip()
        if user_input in valid_options:
            return user_input
        print(f"Invalid input. Please choose one of the following: {', '.join(valid_options)}")

def unit_fixes(data, EFERMI, CONC, TXT_T, HEADER, IT):
    # Shift energy values relative to Fermi energy
    data[:, 0] = data[:, 0] - EFERMI

    # Prompt user for preferred energy units (Ry or eV)
    opt_unit = get_valid_input("\nProceed in units of Ry or eV? (Ry/eV). Default is eV: ", ["Ry", "eV", ""])
    if opt_unit == "eV" or opt_unit=="":
        opt_unit="eV"
        conversion_factor = 13.605693122994  # Conversion factor from Ry to eV
        print("Converting energies and DOS to units of eV.")
    else:
        conversion_factor = 1.0  # No conversion needed
        print("Keeping energies and DOS in units of Ry.")

    # Apply unit conversion to energy values and DOS data
    data[:, 0] = data[:, 0] * conversion_factor
    data[:, 2:] = data[:, 2:] / conversion_factor
    EFERMI = EFERMI * conversion_factor

    # Ask user if concentration correction should be applied
    opt_conc = get_valid_input(
        "\nDo you want to apply concentration correction for element-resolved DOS? (y/n). Default is y: ",
        ["y", "n", ""]
    )

    if opt_conc == "y" or opt_conc=="":
        print("Applying concentration correction to element-resolved DOS...")
        opt_conc="y"
        print("\n")

        # Apply concentration correction to DOS
        for n in range(np.size(IT)):
            conc_factor = CONC[n]  # Concentration of the element at this site
            print(f"Element {TXT_T[n]} with concentration {conc_factor}")

            # Find all columns in HEADER that correspond to the current element
            relevant_columns = [
                idx for idx, label in enumerate(HEADER)
                if label.startswith(TXT_T[n] + " ")  # Ensure exact match by checking prefix
            ]

            if not relevant_columns:
                print(f"Warning: No columns found for element {TXT_T[n]}")
                continue

            # Apply correction factor to the relevant columns
            data[:, relevant_columns] *= conc_factor
            print(f"Corrected columns for {TXT_T[n]}: {relevant_columns}")

        print("Concentration correction applied successfully.")
    else:
        print("Concentration correction skipped.")

    return data, EFERMI, opt_unit, opt_conc

def data_processor(data, TXT_T, IQ, IQAT, NLQ, HEADER, CONC, EFERMI):
    # Step 1: Initialize the `data_tot` array with the energy column
    data_tot = data[:, [0]]  # First column: energy

    # Step 2: Create the new header `HEADER_tot`
    HEADER_tot = ["E"]  # Start with "E" for energy

    # Step 3: Process the elements and their bands (split by spin)
    for n in range(len(TXT_T)):
        # Find the corresponding NLQ value and starting index for the current element
        index_NLQ = np.where(IQ == IQAT[n])[0][0]  # Match the atom ID (IQAT) with IQ
        lmax = NLQ[index_NLQ]  # Maximum number of bands (orbitals) for the current atom
        del index_NLQ

        # Initialize lists to store indices of spin-up and spin-down columns for this element
        spin_up_columns = []
        spin_dn_columns = []
        for l in range(lmax):
            # Locate the indices of spin-up and spin-down columns in the HEADER
            up_col = np.where(HEADER == f"{TXT_T[n]} {BANDS[l+1]} up")[0][0]
            dn_col = np.where(HEADER == f"{TXT_T[n]} {BANDS[l+1]} dn")[0][0]
            spin_up_columns.append(up_col)  # Append spin-up column index
            spin_dn_columns.append(dn_col)  # Append spin-down column index

        # Compute the sum over all bands for each spin
        spin_up_sum = np.sum(data[:, spin_up_columns], axis=1)  # Sum spin-up columns
        spin_dn_sum = np.sum(data[:, spin_dn_columns], axis=1)  # Sum spin-down columns
        total_sum = spin_up_sum + spin_dn_sum  # Total DOS for this element

        # Add the computed spin-resolved and total columns to `data_tot`
        data_tot = np.column_stack((data_tot, spin_up_sum, spin_dn_sum, total_sum))

        # Add header entries for the current element
        HEADER_tot.append(f"{TXT_T[n]} up")  # Spin-up entry for this element
        HEADER_tot.append(f"{TXT_T[n]} dn")  # Spin-down entry for this element
        HEADER_tot.append(f"{TXT_T[n]} tot")  # Total entry for this element

    # Step 4: Process atoms (summed DOS per atom)
    # Create dictionaries to map atoms to their total, up, and dn contributions
    atom_totals = {}
    atom_up_totals = {}
    atom_dn_totals = {}

    for n in range(len(TXT_T)):
        # Extract the atom ID from IQAT[n]
        atom_id = IQAT[n]

        # Find the column indices for the current element in HEADER_tot
        up_col_index = HEADER_tot.index(f"{TXT_T[n]} up")
        dn_col_index = HEADER_tot.index(f"{TXT_T[n]} dn")
        tot_col_index = HEADER_tot.index(f"{TXT_T[n]} tot")

        # Initialize atom totals if not already present
        if atom_id not in atom_totals:
            atom_totals[atom_id] = np.zeros(data_tot.shape[0])
            atom_up_totals[atom_id] = np.zeros(data_tot.shape[0])
            atom_dn_totals[atom_id] = np.zeros(data_tot.shape[0])

        # Accumulate contributions to the atom totals
        atom_up_totals[atom_id] += data_tot[:, up_col_index]
        atom_dn_totals[atom_id] += data_tot[:, dn_col_index]
        atom_totals[atom_id] += data_tot[:, tot_col_index]

    # Append atom contributions to `data_tot` and `HEADER_tot`
    for atom_id in sorted(atom_totals.keys()):
        # Append total, up, and dn contributions for each atom
        data_tot = np.column_stack((data_tot, atom_up_totals[atom_id], atom_dn_totals[atom_id], atom_totals[atom_id]))
        HEADER_tot.append(f"ATOM_{atom_id} up")
        HEADER_tot.append(f"ATOM_{atom_id} dn")
        HEADER_tot.append(f"ATOM_{atom_id} tot")

    # Step 5: Verify totals by integrating up to the Fermi energy
    print("\n--- Integration Results for Total DOS (up to Fermi Energy) ---")

    # Process elements
    for n in range(len(TXT_T)):
        # Find the column index of the `tot` column for the current element
        tot_col_index = HEADER_tot.index(f"{TXT_T[n]} tot")

        # Extract energy and total DOS for the current element
        energy = data_tot[:, 0]
        tot_dos = data_tot[:, tot_col_index]

        # Integrate the DOS up to the Fermi energy (EFERMI)
        below_fermi_indices = energy <= 0.0
        integral = np.trapz(tot_dos[below_fermi_indices], energy[below_fermi_indices])

        # Calculate pure atom properties
        pure_atom_property = integral / CONC[n]

        # Print the result for the current element
        print(f"Element {TXT_T[n]}: Total electrons (up to EFERMI = {EFERMI:.2f} eV) = {integral:.4f}, "
        f"pure atom = {pure_atom_property:.4f}")

    # Process atoms
    print("\n--- Integration Results for Atoms (up to Fermi Energy) ---")

    for atom_id in range(1, max(IQAT) + 1):
        # Locate the `ATOM_X tot` column
        atom_tot_col_name = f"ATOM_{atom_id} tot"
        if atom_tot_col_name in HEADER_tot:
            atom_tot_col_index = HEADER_tot.index(atom_tot_col_name)
            del atom_tot_col_name
            # Extract energy and total DOS for the current atom
            atom_tot_dos = data_tot[:, atom_tot_col_index]
            del atom_tot_col_index

            # Integrate the DOS up to the Fermi energy (EFERMI)
            below_fermi_indices = energy <= 0.0
            atom_integral = np.trapz(atom_tot_dos[below_fermi_indices], energy[below_fermi_indices])
            del below_fermi_indices

            # Print the result for the current atom
            print(f"Atom {atom_id}: Total electrons (up to EFERMI = {EFERMI:.2f}) = {atom_integral:.4f} ")
            del atom_integral, atom_tot_dos
        else:
            print(f"Atom {atom_id}: No 'ATOM_{atom_id} tot' column found in HEADER_tot.")

    print("\nIntegration complete (simple Trapz, only for fast check). Verify the results above.")
    return data_tot, HEADER_tot


def save_parsed(data, fname, HEADER, opt_conc, opt_unit)->None:
    # Export data into an dat file
    # Determine suffix for concentration option
    # If opt_conc is "y", append "_abs_conc" to filenames; otherwise, no additional suffix
    conc_suffix = "_abs_conc" if opt_conc == "y" else ""

    # Construct filenames dynamically
    # For `data_parsed`, use the "bands" keyword
    fname_bands = f"{fname}_bands_{opt_unit}{conc_suffix}.dat"

    # For `data_tot`, use the "tot" keyword
    fname_tot = f"{fname}_tot_{opt_unit}{conc_suffix}.dat"

    # Export `data_parsed` to a file
    # The file will contain comma-separated values (CSV format) and include a header row
    np.savetxt(
        fname_bands,           # File name for the output
        data,                   # Data to export (2D NumPy array)
        delimiter=",",         # Use a comma as the separator for CSV format
        header=",".join(HEADER),  # Convert the list of headers into a comma-separated string
        comments=""            # Ensure no "#" is prepended to the header (default behavior of savetxt)
    )
    print(f"Exported data_parsed to {fname_bands}")
    return fname_bands

def save_tot(data_tot, fname, HEADER_tot, opt_conc, opt_unit)->None:
    # Export data into an dat file
    # Determine suffix for concentration option
    # If opt_conc is "y", append "_abs_conc" to filenames; otherwise, no additional suffix
    conc_suffix = "_abs_conc" if opt_conc == "y" else ""

    # For `data_tot`, use the "tot" keyword
    fname_tot = f"{fname}_tot_{opt_unit}{conc_suffix}.dat"
    
    # Export `data_tot` to a file
    # This file will also be in CSV format and include a header row
    np.savetxt(
        fname_tot,             # File name for the output
        data_tot,                  # Data to export (2D NumPy array)
        delimiter=",",         # Use a comma as the separator for CSV format
        header=",".join(HEADER_tot),  # Convert the list of headers into a comma-separated string
        comments=""            # Ensure no "#" is prepended to the header
    )
    print(f"Exported data_tot to {fname_tot}")
    return fname_tot



#SPRKKR_DOS_export.py

# Function to query yes/no from the user
def get_user_choice(prompt):
    while True:
        choice = input(f"{prompt} (y/n): ").strip().lower()
        if choice in ['y', 'n', ""]:
            return choice == 'y' or choice == ""
        print("Invalid input! Please enter only 'y' or 'n'.")

def sort_dos_data_from_file(fname, 
                            pattern = re.compile(r'([A-Za-z]+)(?:_([0-9]+))?\s+([spdf])\s+(up|dn)'),
                            default=False):
    # Extract header line
    with open(fname, 'r', encoding='utf-8') as f:
        header_line = f.readline().strip()

    # Convert header to list (comma-separated)
    header = np.array(header_line.split(','))

    # Structure: Sites -> Elements -> Orbitals -> Spins -> Columns
    column_map = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'up': [], 'dn': []})))

    # Parse header and assign columns. Default site is "1" if missing.
    for i, col in enumerate(header):
        match = pattern.match(col)
        if match:
            element, site, orbital, spin = match.groups()
            if site is None:
                site = "1"
            column_map[site][element][orbital][spin].append(i)
    
    # Ask which categories to combine
    if default == True: 
        combine_sites = False
        combine_elements = False
        combine_orbitals = False
        combine_spin = False
    else: 
        combine_sites = get_user_choice("Combine sites?")
        combine_elements = get_user_choice("Combine elements?")
        combine_orbitals = get_user_choice("Combine orbitals?")
        combine_spin = get_user_choice("Combine spin?")
    
    # Load numerical data (skip header)
    data = np.genfromtxt(fname, delimiter=",", skip_header=1)

    desired_energy=None

    if default == True:
        # Additional energy selection: use all energies or pick the closest row.
        include_all_energies = True #get_user_choice("Include all energies?")
    else:
        include_all_energies = get_user_choice("Include all energies?")

    if not include_all_energies:
        while True:
            try:
                desired_energy = float(input("Enter desired energy value: "))
                break
            except ValueError:
                print("Invalid input! Please enter a valid numerical value.")
        energy_values = data[:, 0]
        idx = np.argmin(np.abs(energy_values - desired_energy))
        print(f"Closest energy found: {energy_values[idx]} eV")
        data = data[idx:idx+1, :]
    
    # Determine DOS columns to sum based on user choices.
    summed_dos = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: np.zeros(data.shape[0]))))

    for site, elements in column_map.items():
        site_key = "All Sites" if combine_sites else f"Site {site}"
        for element, orbitals in elements.items():
            element_key = "All Elements" if combine_elements else element
            for orbital, spins in orbitals.items():
                orbital_key = "All Orbitals" if combine_orbitals else orbital
                if combine_spin:
                    indices = [idx for spin_list in spins.values() for idx in spin_list]
                    if indices:
                        summed_dos[site_key][element_key][orbital_key] += np.sum(data[:, indices], axis=1)
                else:
                    for spin, indices in spins.items():
                        if indices:
                            key = orbital_key + ' ' + spin
                            summed_dos[site_key][element_key][key] += np.sum(data[:, indices], axis=1)
    return data, summed_dos, desired_energy, combine_sites, combine_elements, combine_orbitals, combine_spin, include_all_energies

# Transformation functions for header components
def transform_site(site):
    # Replace "Site X" with "AX" or "All Sites" with "all sites"
    if site.startswith("Site "):
        parts = site.split()
        if len(parts) == 2:
            return "A" + parts[1]
    elif site == "All Sites":
        return "all sites"
    return site

def transform_element(element):
    # Replace "All Elements" with "all elements"
    return "all elements" if element == "All Elements" else element

def transform_orbital(orbital):
    # Replace "All Orbitals" with "all orbitals"
    return "all orbitals" if orbital == "All Orbitals" else orbital

def build_datastruct(data, summed_dos, desired_energy, fname, combine_sites, combine_elements, combine_orbitals, combine_spin, include_all_energies):
    # Create output array and header.
    # Energy column header is replaced with "E"
    energy = data[:, 0]
    output_data = [energy]
    output_header = ["E"]

    # Build header for each DOS column from the nested dictionary.
    for site_key, elements in summed_dos.items():
        t_site = transform_site(site_key)
        for element_key, orbitals in elements.items():
            t_element = transform_element(element_key)
            for orbital_key, dos_values in orbitals.items():
                parts = orbital_key.split()
                if len(parts) == 2:
                    t_orbital = transform_orbital(parts[0])
                    t_spin = parts[1]  # spin remains unchanged
                    header_string = f"{t_site} {t_element} {t_orbital} {t_spin}"
                else:
                    t_orbital = transform_orbital(orbital_key)
                    header_string = f"{t_site} {t_element} {t_orbital}"
                output_header.append(header_string)
                output_data.append(dos_values)

    # Stack and save output; the header is comma-joined.
    output_array = np.column_stack(output_data)
    output_filename = fname + "_export_final.dat"
    np.savetxt(output_filename, output_array, delimiter=",", header=",".join(output_header), comments='')

    # Output summary
    print("\nSummary Mode:")
    print(f"  Sites: {'Yes' if combine_sites else 'No'}")
    print(f"  Elements: {'Yes' if combine_elements else 'No'}")
    print(f"  Orbitals: {'Yes' if combine_orbitals else 'No'}")
    print(f"  Spin: {'Yes' if combine_spin else 'No'}")
    print(f"  Energy selection: {'All energies' if include_all_energies else f'Closest to {desired_energy} eV'}")
    print(f"\nSummed DOS saved to '{output_filename}'")
    return output_filename

#SPRKKR_DOS_plotter.py
def convert_to_pandas(filepath, default=False):
    if default:
        df=pd.read_csv(filepath)
        return df
    else:
        print("Default settings are of. This may break the plotter...")
        root = tk.Tk()
        root.withdraw()
        fname = filedialog.askopenfilename(
        title="Select SPR-KKR Exported DAT file",
        filetypes=[("DAT files", "*.dat"), ("All files", "*.*")]
        )
        df=pd.read_csv(fname)
        if not fname:
            print("No file selected.")
            exit()
    return df

def analyze_headers(columns):
    """
    Scans the file headers to find what Sites (A1..), Elements (Fe..), 
    and Orbitals (s,p..) are present.
    """
    metadata = {
        "sites": set(),
        "elements": set(),
        "orbitals": set(),
        "spins": set()
    }
    
    # Regex to capture: "Site Element Orbital [Spin]"
    # Matches: "A1 Fe d up"  OR  "A1 Fe d"
    pattern = re.compile(r"(\S+)\s+(\S+)\s+(\S+)(?:\s+(up|dn))?")

    for col in columns:
        if col.strip() == "E": continue
        
        match = pattern.search(col)
        if match:
            site, elem, orb, spin = match.groups()
            metadata["sites"].add(site)
            metadata["elements"].add(elem)
            metadata["orbitals"].add(orb)
            if spin: metadata["spins"].add(spin)
            
    # Sort them so A1, A2, A3 appear in order
    meta = {k: sorted(list(v)) for k, v in metadata.items()}

    print("\n--- Detected in file ---")
    print(f" Sites:    {', '.join(meta['sites'])}")
    print(f" Elements: {', '.join(meta['elements'])}")
    print(f" Orbitals: {', '.join(meta['orbitals'])}")
    print("------------------------")

    return meta

def compute_group_dos(df, unique_items):
    """
    Sum columns based on a list of keywords. 
    Example: if unique_items=['A1', 'A2'], it creates 'PDOS A1' and 'PDOS A2'.
    """
    computed_df = pd.DataFrame()
    
    for item in unique_items:
        # Find all columns containing this keyword (e.g. "A1")
        # We split by space to ensure "s" doesn't match "spin"
        cols_to_sum = [c for c in df.columns if c != "E" and item in c.split()]
        
        if cols_to_sum:
            col_name = f"PDOS {item}"
            computed_df[col_name] = df[cols_to_sum].sum(axis=1)
            print(f"  > Created '{col_name}' (sum of {len(cols_to_sum)} columns)")
            
    return computed_df

def process_pandas(df_raw, meta):
    df_plot=pd.DataFrame()
    df_plot["E"]=df_raw["E"]

    data_cols = [c for c in df_raw.columns if c != "E"]
    df_plot["Total DOS"] = df_raw[data_cols].sum(axis=1)

    print("\nWhat do you want to compare?")
    print("1. Nothing (Total DOS only)")
    menu = {"1": "total"}
    idx = 2
    
    if len(meta["sites"]) > 1:
        print(f"{idx}. Sites ({', '.join(meta['sites'])})")
        menu[str(idx)] = ("sites", meta["sites"])
        idx += 1
    
    if len(meta["elements"]) > 1:
        print(f"{idx}. Elements ({', '.join(meta['elements'])})")
        menu[str(idx)] = ("elements", meta["elements"])
        idx += 1

    if len(meta["orbitals"]) > 1:
        print(f"{idx}. Orbitals ({', '.join(meta['orbitals'])})")
        menu[str(idx)] = ("orbitals", meta["orbitals"])
        idx += 1
        
    if len(meta["spins"]) > 1:
        print(f"{idx}. Spins ({', '.join(meta['spins'])})")
        menu[str(idx)] = ("spins", meta["spins"])
        idx += 1

    choice = input("\nSelect number: ").strip()
    selection = menu.get(choice, "total")

    if selection != "total":
        mode, items = selection
        print(f"\nCalculating {mode} resolved DOS...")
        
        # This function loops through ['A1', 'A2'...] and sums the columns for each
        df_groups = compute_group_dos(df_raw, items)
        
        # Add new columns to the plotting dataframe
        df_plot = pd.concat([df_plot, df_groups], axis=1)
        title = f"{mode.capitalize()}-Resolved DOS"
    
    return df_plot

def plot_dos(df, title="DOS Plot", filename="plot.png"):
    x_col = "E"
    y_cols = [c for c in df.columns if c != x_col]
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    
    for i, col in enumerate(y_cols):
        # Special style for Total DOS
        if "Total" in col:
            ax.plot(df[x_col], df[col], color='k', linewidth=1.5, label=col, zorder=10)
            ax.fill_between(df[x_col], df[col], color='k', alpha=0.05)
        else:
            c = COLORS[i % len(COLORS)]
            ax.plot(df[x_col], df[col], color=c, linewidth=1.2, label=col)

    # Labels and Limits
    ax.set_xlabel(r"$E-E_\text{F}$ (eV)", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(r"Density of States (st./eV/at.)", fontsize=LABEL_FONT_SIZE)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.set_xlim(XLIM)
    #ax.set_ylim(YLIM)
    
    # Reference lines
    ax.axvline(0, color='gray', linestyle=':', linewidth=1) 
    ax.axhline(0, color='gray', linewidth=0.5) 

    # Legend
    ax.legend(fontsize=9, frameon=False, loc='upper right')
    
    plt.tight_layout()
    print("Showing plot...")
    plt.show()

def main()->None:
    #SPRKKR_DOS_parser.py
    filepath = file_select()
    header, data_raw = load_data(filepath)
    HEADER, NE, NQ_eff, NT_eff, EFERMI, IREL, IT, TXT_T, CONC, NAT, IQAT, NLQ, IQ = header_parser(header)
    data = data_parser(data_raw,NE,IT, NLQ)
    data, EFERMI, opt_unit, opt_conc = unit_fixes(data, EFERMI, CONC, TXT_T, HEADER, IT)
    data_tot, HEADER_tot = data_processor(data, TXT_T, IQ, IQAT, NLQ, HEADER, CONC, EFERMI)
    fname_bands = save_parsed(data,filepath,HEADER, opt_conc,opt_unit)
    fname_tot = save_tot(data_tot,filepath,HEADER_tot, opt_conc,opt_unit)
    
    #SPRKKR_DOS_export.py
    data, summed_dos, desired_energy, combine_sites, combine_elements, combine_orbitals, combine_spin, include_all_energies = sort_dos_data_from_file(fname_bands,default=DEFAULT_COMB)
    fname_comb=build_datastruct(data, summed_dos, desired_energy, fname_bands, combine_sites, combine_elements, combine_orbitals, combine_spin, include_all_energies)

    #SPRKKR_DOS_plotter.py
    df_raw=convert_to_pandas(fname_comb, default=DEFAULT_COMB)
    meta = analyze_headers(df_raw.columns)
    df_plot=process_pandas(df_raw, meta)
    plot_dos(df_plot)

if __name__=="__main__":
    main()
