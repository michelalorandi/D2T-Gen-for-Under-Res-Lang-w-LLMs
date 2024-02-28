import xml.etree.ElementTree as ET


# Load data file and put it in json format
def get_data(filepath, lang='ga'):
    tree = ET.parse(filepath)
    root = tree.getroot()

    data = []
    # Extract all entries
    for entry in root.findall('./entries/entry'):
        triples = []
        # Extract the triples associated to the current entry
        for triple in entry.find('modifiedtripleset').findall('mtriple'):
            triples.append(triple.text)

        sentences = []
        # Extract the verbalization associated to the current entry
        for lex in entry.findall('lex'):
            target = lex.text

            if lex.attrib['lang'] == lang:
                #print(lex.tag, lex.attrib)
                sentences.append(target)

        data.append({
            'triples_set': triples,  # List of triples (string)
            'sentences': sentences  # List of sentences
        })
    return data


def get_data_w_trcount(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()

    data = []
    count = {}
    # Extract all entries
    for entry in root.findall('./entries/entry'):
        triples = []
        # Extract the triples associated to the current entry
        for triple in entry.find('modifiedtripleset').findall('mtriple'):
            triples.append(triple.text)

        sentences = []
        # Extract the verbalization associated to the current entry
        for lex in entry.findall('lex'):
            target = lex.text

            sentences.append(target)

        data.append({
            'tripleset': '<br>'.join(triples),
            'output': sentences[0],  # List of sentences,
            'triple_count': entry.attrib['size'],
            'category': entry.attrib['category'],
            'eid': entry.attrib['eid'],
            'system': 'refs'
        })
    return data


# Send request to OpenAI APIs using the specified hyperparameters
def request_openai(experiment_params, prompt):
  while True:
    try:
       resp = openai.Completion.create(
          model=experiment_params['model'],
          prompt=prompt,
          max_tokens=experiment_params['maximum_length'],
          temperature=experiment_params['temperature'],
          top_p=experiment_params['top_p'],
          frequency_penalty=experiment_params['frequency_penalty'],
          presence_penalty=experiment_params['presence_penalty'])
       return resp, resp['choices'][0]['text']
    except Exception as e:
      print(str(e))
      print("Retrying...")
