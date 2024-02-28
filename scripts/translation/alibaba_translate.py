import os
import json
import time
import argparse
from tqdm import tqdm
from aliyunsdkcore.client import AcsClient
from aliyunsdkalimt.request.v20181012 import TranslateGeneralRequest


target_languages = ['ga', 'mt', 'cy', 'br']


def translate_text(text, lang, client):
    # Create an API request and set the request parameters.
    request = TranslateGeneralRequest.TranslateGeneralRequest()
    request.set_SourceLanguage("en")
    request.set_SourceText(text)
    request.set_FormatType("text")
    request.set_TargetLanguage(lang)
    request.set_method("POST")
    # Initiate the API request and obtain the response.
    response = client.do_action_with_exception(request)
    return json.loads(response)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_folder_path', metavar='path',
                        default=os.path.join('.', 'results', 'processed'))
    parser.add_argument('--language', type=str, required=True,
                        choices=['ga', 'mt', 'cy', 'br', 'en'])
    parser.add_argument('--model', type=str, required=True, 
                        choices=['gpt35', 'llama270bchat', 'bloom176B', 'falcon180b'])
    parser.add_argument('--prompt', type=str, required=True, choices=['zero_shot', 'few_shot'])
    parser.add_argument('--token_path', type=str, required=False, default=os.path.join('.', 'token.json'))

    args = parser.parse_args()

    if os.path.exists(args.token_path):
        with open(args.token_path, 'r') as token_file:
            token = json.load(token_file)
        # Create an AcsClient instance.
        client = AcsClient(
            token['access_key_id'],  # The AccessKey ID of your Alibaba Cloud account.
            token['access_key'],  # AccessKey Secret
            token['region_id']  # The ID of the region.
        )

        exp = f'{args.language}_{args.model}_prompt_{args.prompt}'
        print(exp)
        hyps = {
            'ga': [],
            'mt': [],
            'cy': [],
            'br': []
        }
        with open(os.path.join(args.results_folder_path, exp+'_hypothesis_processed.txt'), 'r') as hyp_file:
            en_hyps = hyp_file.read().split('\n')
        for text in tqdm(en_hyps, desc="Executing "+exp):
            for target_language in target_languages:
                translation = None
                try:
                    translation = translate_text(text, target_language, client)
                    hyps[target_language].append(translation['Data']['Translated'])
                except Exception as e:
                    print(str(e))
                    print(translation)
                    print('Retrying in 20 seconds')
                    time.sleep(20)

        for target_language in target_languages:
            with open(os.path.join(args.results_folder_path, exp.replace('en', target_language+'_tr_alibaba') + '_hypothesis_processed.txt'), 'w') as hyp_file:
                hyp_file.write('\n'.join(hyps[target_language]))
    else:
        print('Token file not found')
