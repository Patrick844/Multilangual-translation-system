from lxml import etree
import csv
import gzip
import shutil # Used for file decompression

def parse_tmx(path,new_file,source_lang, target_lang, source_code, target_code):
  # Parse the TMX (Translation Memory eXchange) file and get the root element
  tree = etree.parse(path)  # Replace with the path to your TMX file
  root = tree.getroot()

  # Get the body of the TMX file, where the translation units are located
  body = root.find('body')  # The 'body' element contains all translation units

  # Find all translation units (tu) in the body
  translation_units = body.findall("tu")  # 'tu' elements contain translations for each segment

  # Define namespaces to correctly handle the xml:lang attribute in the TMX file
  namespaces = {
      'xml': 'http://www.w3.org/XML/1998/namespace'  # XML namespace for the xml:lang attribute
  }

  # Prepare an empty list to store the translation pairs (English and Arabic text)
  data = []

  # Iterate through each translation unit (tu) in the TMX file
  for tu in translation_units:
      # Extract English text using XPath, filtering by xml:lang='en' for the English language
      en_text = tu.xpath(f"tuv[@xml:lang='{source_code}']/seg/text()", namespaces=namespaces)

      # Extract Arabic text using XPath, filtering by xml:lang='fr' for the Arabic language
      ar_text = tu.xpath(f"tuv[@xml:lang='{target_code}']/seg/text()", namespaces=namespaces)

      # Append the extracted English and Arabic text to the data list as a tuple
      # Note: Since en_text and fr_text are lists, we're taking the first element (if present)
      data.append((en_text[0] if en_text else '', ar_text[0] if ar_text else ''))

  # Open a CSV file to write the translation data
  with open(new_file, 'w', newline='', encoding='utf-8') as csvfile:
      writer = csv.writer(csvfile)

      # Write the header row for the CSV file with 'English' and 'Arabic' columns
      writer.writerow([source_lang, target_lang])

      # Write all translation rows (English and Arabic pairs) to the CSV file
      writer.writerows(data)

  # Print a success message once the CSV file is created
  print("CSV file created successfully!")


    # Function to decompress the .gz file
def decompress_gz(file_path, output_file):
    with gzip.open(file_path, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print("File decompressed successfully.")

