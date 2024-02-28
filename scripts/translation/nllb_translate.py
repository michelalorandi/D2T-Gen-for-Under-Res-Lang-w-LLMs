import os
import time
import argparse
from tqdm import tqdm


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-1.3B")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-1.3B").to("cuda")


target_languages = ['ga', 'mt', 'cy']
batch_size = 128


def translate_text(text, lang):
    if lang == "mt":
        lang = "mlt_Latn"
    elif lang == "cy":
        lang = "cym_Latn"
    elif lang == "ga":
        lang = "gle_Latn"
    inputs = tokenizer(text, return_tensors="pt", padding=True).to("cuda")
    translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[lang])
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)


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
    for target_language in target_languages:
        for index in tqdm(range(0, len(en_hyps), batch_size), desc=f"Executing {exp} {target_language}"):
            batch = en_hyps[index:index+batch_size]
            translation = None
            try:
                translation = translate_text(batch, target_language)
                hyps[target_language] += translation
            except Exception as e:
                print(str(e))
                print(translation)
                print('Retrying in 20 seconds')
                time.sleep(20)

        with open(os.path.join(args.results_folder_path, exp.replace('en', target_language+'_tr_nllb') + '_hypothesis_processed.txt'), 'w') as hyp_file:
            hyp_file.write('\n'.join(hyps[target_language]))
