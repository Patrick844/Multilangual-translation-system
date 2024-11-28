"""
Module Name: converter.py

This module contains functions for converting
decompressing tmx files (decompres_gz)
and convert them to CSV (parse_tmx)
"""


import csv
import gzip
import shutil
from typing import Tuple
from lxml import etree


def parse_tmx(input_file: str,
              output_file: str,
              source: Tuple[str, str],
              target: Tuple[str, str],
              ) -> None:
    """Parsing document from tmx

    Args:
        input_file (str): Name of input_file
        output_file (str): Name of output_file
        source (Tuple[str, str]):
            - **source_lang (str)**: Source language
            - **source_code (str)**: Source code
        target (Tuple[str, str]):
            - **target_lang (str)**: Target language
            - **target_code (str)**: Target code

    Returns:
        None
    """

    source_code, source_lang = source
    target_lang, target_code = target

    # Get the body of the TMX file, where the translation units are located
    body = etree.parse(input_file).getroot().find('body')

    # Find all translation units (tu) in the body
    translation_units = body.findall("tu")

    namespaces = {
      'xml': 'http://www.w3.org/XML/1998/namespace'
        }

    # Store the translation pairs (English and Arabic text)
    data = []

    # Iterate through each translation unit (tu) in the TMX file
    for tu in translation_units:
        # Extract text using XPath, filtering 'source_code'
        sr_txt = tu.xpath(f"tuv[@xml:lang='{source_code}']/seg/text()", namespaces=namespaces)

        # Extract text using XPath, filtering by 'target_code'
        trgt_txt = tu.xpath(f"tuv[@xml:lang='{target_code}']/seg/text()", namespaces=namespaces)

        # Append the extracted English and Arabic text to the data list as a tuple
        # Note: Since en_text and fr_text are lists, we're taking the first element (if present)
        data.append((sr_txt[0] if sr_txt else '', trgt_txt[0] if trgt_txt else ''))

    # Open a CSV file to write the translation data
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row for the CSV file with 'English' and 'Arabic' columns
        writer.writerow([source_lang, target_lang])

        # Write all translation rows (English and Arabic pairs) to the CSV file
        writer.writerows(data)

    # Print a success message once the CSV file is created
    print("CSV file created successfully!")


def decompress_gz(input_file, output_file):

    """
    Decompress Large TMX files

    Args:
        input_file (str): Name of the input file
        output_file (str): Name of the output file

    Return:
        None
    """

    with gzip.open(input_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print("File decompressed successfully.")
