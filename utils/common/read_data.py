# =============================================================================
# Author   : 刘彤 (Tong Liu)
# Date     : 2026-03-04
# Version  : 1.0
# =============================================================================


import os
import re
import base64
import struct
import xml.etree.ElementTree as ET

def extract_mzml_data(binaryDataArray, index, byte_type):
    """
    Extract and decode binary data from an mzML file.
    
    Parameters:
    - binaryDataArray: The binaryDataArray element from the mzML file containing the binary data.
    - index: The index value indicating whether to extract Retention Time (RT) data (index=0) or m/z data (index=1).
    
    Returns:
    - Decoded data (RT data as double precision floats, m/z data as single precision floats).
    """
    if index == 0:  # Retention Time (RT) data
        rt_bytes = base64.b64decode(binaryDataArray.text.strip())
        num_data_points = len(rt_bytes) // 8  # Each RT data point takes 8 bytes
        rt_data = struct.unpack("<" + "d" * num_data_points, rt_bytes)
        return rt_data
    elif index == 1:  # m/z data
        if byte_type == 32:
            mz_bytes = base64.b64decode(binaryDataArray.text.strip())
            num_data_points = len(mz_bytes) // 4  # Each m/z data point takes 4 bytes
            mz_data = struct.unpack("<" + "f" * num_data_points, mz_bytes)
        if byte_type == 64:
            mz_bytes = base64.b64decode(binaryDataArray.text.strip())
            num_data_points = len(mz_bytes) // 8  # Each m/z data point takes 4 bytes
            mz_data = struct.unpack("<" + "d" * num_data_points, mz_bytes)
        return mz_data
    return None

def parse_mzml_file(chrom_file, indir):
    """
    Parse a single mzML file and extract relevant signal information.
    
    Parameters:
    - chrom_file: The filename of the mzML file to be parsed.
    - indir: The directory where the mzML files are stored.
    
    Returns:
    - signal_information: A list of extracted signal information from the mzML file.
      This includes sample ID, Q1, Q3 values, start/end values, RT data, and m/z data.
    """
    full_path = os.path.join(indir, "mzML", chrom_file)
    sampleID = full_path.split("/")[-1].split("-")[-1][:-5].split("(")[0].strip()  # Extract the sample ID
    tree = ET.parse(full_path)  # Parse the XML file
    root = tree.getroot()

    signal_information = []  # List to store extracted signal information
    tic_information = []
    
    # Traverse the XML structure and extract relevant data
    for child in root:
        if child.tag == "{http://psi.hupo.org/ms/mzml}mzML":
            for subchild in child:
                if subchild.tag == "{http://psi.hupo.org/ms/mzml}run":
                    for subsubchild in subchild:
                        for subsubsubchild in subsubchild:
                            id_attr = subsubsubchild.attrib.get('id', '')
                            if 'TIC' in id_attr:
                                for subsubsubsubchild in subsubsubchild:
                                    if subsubsubsubchild.tag == "{http://psi.hupo.org/ms/mzml}binaryDataArrayList":
                                        for index, binaryDataArrayList in enumerate(subsubsubsubchild):
                                            for binaryDataArray in binaryDataArrayList:
                                                if binaryDataArray.tag == "{http://psi.hupo.org/ms/mzml}cvParam":
                                                    if "float" in binaryDataArray.get('name'):
                                                        if "64" in binaryDataArray.get('name'):
                                                            byte_type = 64
                                                        if "32" in binaryDataArray.get('name'):
                                                            byte_type = 32
                                                            
                                                if binaryDataArray.tag == "{http://psi.hupo.org/ms/mzml}binary":
                                                    if index == 0:
                                                        rt_data = extract_mzml_data(binaryDataArray, index, byte_type)  # Decode Retention Time (RT) data
                                                    if index == 1:
                                                        mz_data = extract_mzml_data(binaryDataArray, index, byte_type)  # Decode m/z data
                                                        mz_data = [x + 50 for x in mz_data]
                                                    if index == 1:  # Only append when m/z data has been decoded
                                                        tic_information.append([sampleID, rt_data, mz_data])
                            if 'SRM SIC' in id_attr:  # Only process 'SRM SIC' type IDs
                                # Extract mw_ID, start, end, Q1, Q3 from the 'id' attribute
                                mw_ID = re.search(r'name=([^ ]+)', id_attr).group(1) if re.search(r'name=([^ ]+)', id_attr) else ''
                                start = re.search(r'start=([^ ]+)', id_attr).group(1) if re.search(r'start=([^ ]+)', id_attr) else ''
                                end = re.search(r'end=([^ ]+)', id_attr).group(1) if re.search(r'end=([^ ]+)', id_attr) else ''
                                Q1_value = re.search(r'Q1=([^ ]+)', id_attr).group(1) if re.search(r'Q1=([^ ]+)', id_attr) else ''
                                Q3_value = re.search(r'Q3=([^ ]+)', id_attr).group(1) if re.search(r'Q3=([^ ]+)', id_attr) else ''
                                
                                for subsubsubsubchild in subsubsubchild:
                                    if subsubsubsubchild.tag == "{http://psi.hupo.org/ms/mzml}binaryDataArrayList":
                                        for index, binaryDataArrayList in enumerate(subsubsubsubchild):
                                            for binaryDataArray in binaryDataArrayList:
                                                if binaryDataArray.tag == "{http://psi.hupo.org/ms/mzml}cvParam":
                                                    if "float" in binaryDataArray.get('name'):
                                                        if "64" in binaryDataArray.get('name'):
                                                            byte_type = 64
                                                        if "32" in binaryDataArray.get('name'):
                                                            byte_type = 32

                                                if binaryDataArray.tag == "{http://psi.hupo.org/ms/mzml}binary":
                                                    # Decode RT and m/z data
                                                    if index == 0:
                                                        if binaryDataArray.text is not None:
                                                            rt_data = extract_mzml_data(binaryDataArray, index, byte_type)  # Decode Retention Time (RT) data
                                                        else:
                                                            rt_data = []
                                                    if index == 1:
                                                        if binaryDataArray.text is not None:
                                                            mz_data = extract_mzml_data(binaryDataArray, index, byte_type)  # Decode m/z data
                                                            mz_data = [x + 50 for x in mz_data]
                                                        else:
                                                            mz_data = []
                                                    
                                                    # Only append to signal_information after both RT and m/z data are extracted
                                                    if index == 1 and len(rt_data) > 0:  # Only append when m/z data has been decoded
                                                        signal_information.append([sampleID, mw_ID, start, end, Q1_value, Q3_value, rt_data, mz_data])

    return signal_information, tic_information

def read_mzml_data(indir, chrom_files):
    """
    Process multiple mzML files and extract signal information from all of them.
    
    Parameters:
    - indir: The directory where mzML files are located.
    - chrom_files: A list of mzML file names to be processed.
    
    Returns:
    - A list of signal information from all mzML files, including data from all sample files.
    """
    all_signal_information = []  # List to store signal information from all files
    all_tic_information = []
    for chrom_file in chrom_files:
        signal_information, tic_information = parse_mzml_file(chrom_file, indir)  # Parse a single mzML file
        all_signal_information.extend(signal_information)  # Add the results to the list of all signal information
        all_tic_information.extend(tic_information)  # Add the results to the list of all tic information
    
    return all_signal_information, all_tic_information
