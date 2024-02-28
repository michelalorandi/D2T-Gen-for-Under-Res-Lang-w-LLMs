import os
import argparse
from tqdm import tqdm
from os import environ

from google.cloud import translate


PROJECT_ID = environ.get("PROJECT_ID", "")
assert PROJECT_ID
PARENT = f"projects/{PROJECT_ID}"


target_languages = ["ga", "mt", "cy"]


def translate_text(text: str, target_language_code: str) -> translate.Translation:
    client = translate.TranslationServiceClient()

    response = client.translate_text(
        parent=PARENT,
        contents=[text],
        source_language_code="en",
        target_language_code=target_language_code,
        mime_type="text/plain"
    )

    return response.translations[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_folder_path', metavar='path',
                        default=os.path.join('.', 'results', 'processed'))
    parser.add_argument('--language', type=str, required=True,
                        choices=['ga', 'mt', 'cy', 'br', 'en'])
    parser.add_argument('--model', type=str, required=True, 
                        choices=['gpt35', 'llama270bchat', 'bloom176B', 'falcon180b'])
    parser.add_argument('--prompt', type=str, required=True, choices=['zero_shot', 'few_shot'])
    args = parser.parse_args()

    exp = f'{args.language}_{args.model}_prompt_{args.prompt}'
    print(exp)
    hyps = {
        'ga': [],
        'mt': [],
        'cy': []
    }
    with open(os.path.join(args.results_folder_path, exp+'_hypothesis_processed.txt'), 'r') as hyp_file:
        en_hyps = hyp_file.read().split('\n')

    for index in tqdm(range(len(hyps['ga']), len(en_hyps)), desc="Executing "+exp):
        text = en_hyps[index]
        for target_language in target_languages:
            translation = translate_text(text, target_language)
            source_language = translation.detected_language_code
            translated_text = translation.translated_text

            hyps[target_language].append(translated_text)

    for target_language in target_languages:
        with open(os.path.join(args.results_folder_path, exp.replace('en', target_language+'_tr_google') + '_hypothesis_processed.txt'), 'w') as hyp_file:
            hyp_file.write('\n'.join(hyps[target_language]))

