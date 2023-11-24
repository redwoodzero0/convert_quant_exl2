#!/usr/bin/env python

import os
import subprocess
import shutil
import json
import argparse 

parser = argparse.ArgumentParser(description='Convert and Quantize fp16 models to Exllama2.')
parser.add_argument('model_path', type=str, help='Path to FP16 model directory')
args = parser.parse_args()

settings_path = os.path.join('/workspace/exllamav2/util', 'settings.json')
with open(settings_path, 'r') as f:
    config = json.load(f)

fp16_model_dir = args.model_path
cal_dataset = config["cal_dataset"]
bits_per_weight = config["bits_per_weight"]
head_bits = config["head_bits"]
gpu_rows = config["gpu_rows"]
token_length = config["token_length"]
measurement_length = config["measurement_length"]
rope_alpha = config["rope_alpha"]
quant_dir = os.path.join(fp16_model_dir, f"{os.path.basename(fp16_model_dir)}-{bits_per_weight}bpw-h{head_bits}-exl2")
util_dir = os.path.dirname(os.path.abspath(__file__))
exllama_dir = os.path.dirname(util_dir)
measurement_file = os.path.join(exllama_dir, f"measurement-{os.path.basename(fp16_model_dir)}.json")

def clean_up_leftovers(directory):
    print("Remove out_tensor/")
    shutil.rmtree(os.path.join(directory, 'out_tensor'), ignore_errors=True)
    files_to_remove = ['cal_data.safetensors', 'job.json', 'input_states.safetensors', 'output_states.safetensors']
    for file_name in files_to_remove:
        print(f"Removing {file_name}")
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)

def save_measurement_file():
    if not os.path.exists(measurement_file):
        print("Copying measurement file...")
    try:
        shutil.copy(os.path.join(quant_dir, 'measurement.json'), measurement_file)
    except FileNotFoundError:
        print(f"Failed to find 'measurement.json' in {quant_dir}.")

def copy_model_config():
    print("Copying config files to quant directory...")
    model_files = [f for f in os.listdir(fp16_model_dir) if f.endswith('.json') or f.startswith('tokenizer.')]
    for model_file in model_files:
        shutil.copy(os.path.join(fp16_model_dir, model_file), os.path.join(quant_dir, model_file))

if os.path.exists(quant_dir):
    print(f"Target output directory already exists: {quant_dir}")
else:
    print(f"Creating directory {quant_dir}")
    os.makedirs(quant_dir)

os.chdir(exllama_dir)

if os.path.exists(measurement_file):
    print(f"Using previous measurement.json file: {measurement_file}")
    measurement_arg = ['-m', measurement_file]
else:
    print(f"No previous measurement.json file found")
    measurement_arg = []

try:
    subprocess.run(['python', './convert.py', '-i', fp16_model_dir, '-o', quant_dir, '-c', f'./{cal_dataset}', '-b', bits_per_weight, '-hb', head_bits,'-gr', gpu_rows, '-l', token_length, '-ml', measurement_length, '-ra', rope_alpha]  + measurement_arg)
except subprocess.CalledProcessError as e:
    print("### ERROR ###")
    print(e)
else:
    copy_model_config()
    save_measurement_file()
    clean_up_leftovers(quant_dir)
