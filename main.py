# -*- coding: utf-8 -*-
"""
MS Data Analysis Suite - Main CLI Router

This script serves as the unified Command Line Interface (CLI) entry point
for both widely-targeted (MRM) and untargeted mass spectrometry data analysis workflows.
It utilizes argparse subparsers to route the user's commands and isolate arguments 
for the appropriate module, ensuring a clean and scalable project structure.
"""
import argparse
import multiprocessing

# Import the targeted pipeline execution module
from main_targeted import run_targeted_pipeline

# Import the untargeted pipeline execution module (uncomment when implemented)
from main_untargeted import run_untargeted_pipeline 

def main():
    """
    Parses command-line arguments and routes execution to the corresponding workflow.
    
    This function defines two main execution paths via subparsers:
    1. 'targeted': Triggers the widely-targeted (MRM) peak mapping, YOLO detection, 
       and integration pipeline.
    2. 'untargeted': Triggers the global feature extraction, alignment, and 
       annotation pipeline.
       
    The parsed arguments (args) are then passed directly to the respective 
    pipeline's main execution function.
    """
    # 1. Initialize the top-level CLI parser
    parser = argparse.ArgumentParser(description="MS Data Analysis Suite (Targeted & Untargeted)")
    
    # dest="workflow" stores the chosen sub-command ('targeted' or 'untargeted') into args.workflow
    # required=True enforces that the user must explicitly declare the workflow type
    subparsers = parser.add_subparsers(dest="workflow", required=True, help="Choose your analytical workflow")

    # ==========================================
    # 2. Register arguments exclusively for the Targeted (MRM) workflow
    # ==========================================
    parser_tgt = subparsers.add_parser("targeted", help="Run the widely-targeted (MRM) pipeline")
    
    parser_tgt.add_argument('--indir', type=str, required=True, 
                            help='Path to the input data directory containing .wiff/.mzML and metadata files. [required]')
    parser_tgt.add_argument('--threads', default=8, type=int, metavar='N', 
                            help='Number of CPU threads to allocate for parallel processing (default: 8)')
    parser_tgt.add_argument('--type', type=str, default="rp", 
                            help='Chromatography type identifier, e.g., Reverse Phase (rp) (default: rp)')

    # ==========================================
    # 3. Register arguments exclusively for the Untargeted workflow
    # ==========================================
    # ==========================================
    # 3. 注册非靶 (Untargeted) 的所有专属参数
    # ==========================================
    parser_untgt = subparsers.add_parser("untargeted", help="Run the untargeted pipeline")
    
    parser_untgt.add_argument('--indir', type=str, required=True, help='Data folder.')
    parser_untgt.add_argument('--threads', default=16, type=int, metavar='N', help='Number of data loading workers (default: 16)')
    parser_untgt.add_argument('--ppm', type=int, default=10, help='PPM tolerance for extraction peak')
    parser_untgt.add_argument('--all_ions', type=str, default="self", choices=["cal", "self"], help='Type of all ions processing')
    parser_untgt.add_argument('--polarity', default='positive', choices=["positive", "negative"], help='Ionization polarity')
    parser_untgt.add_argument('--minWidth', type=float, default=5.0, help='Minimum peak width')
    parser_untgt.add_argument('--maxWidth', type=float, default=50.0, help='Maximum peak width')
    parser_untgt.add_argument('--s2n', type=float, default=5.0, help='Signal-to-noise ratio threshold')
    parser_untgt.add_argument('--noise', type=float, default=100.0, help='Noise level threshold')
    parser_untgt.add_argument('--mzDiff', type=float, default=0.015, help='m/z difference for peak grouping')
    parser_untgt.add_argument('--prefilter', type=float, default=3.0, help='Pre-filtering intensity threshold')

    # Parse the arguments provided by the user in the terminal
    args = parser.parse_args()

    # ==========================================
    # 4. Route execution based on the user's workflow selection
    # ==========================================
    if args.workflow == "targeted":
        print("[*] Starting Targeted (MRM) Workflow...")
        # Delegate execution and pass the parsed namespace to the targeted logic
        run_targeted_pipeline(args)
        
    elif args.workflow == "untargeted":
        print("[*] Starting Untargeted Workflow...")
        # Delegate execution and pass the parsed namespace to the untargeted logic
        run_untargeted_pipeline(args)
        print("Notice: Untargeted pipeline is currently under construction.")

if __name__ == "__main__":
    # Freeze support is essential for safe multiprocessing on Windows operating systems
    # and maintaining cross-platform compatibility when packaging the script.
    multiprocessing.freeze_support()
    main()