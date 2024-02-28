import os
import argparse

eval_dir = os.path.join('.', 'evaluation_results')
res_dir = os.path.join('.', 'results', 'processed')
ref_dir = os.path.join('.', 'references')

ref_num = {
    'ga': '1',
    'mt': '1',
    'cy': '1',
    'br': '2',
    'en': '5'
}


def execute_eval(res_filename, lang):
    res_filepath = os.path.join(res_dir, res_filename)
    ref_filepath = os.path.join(ref_dir, f'{lang}_reference')
    command = f'python eval.py -hyp {res_filepath} -ref {ref_filepath} -nr {ref_num[lang]} -m bleu,meteor,chrf++,ter,bert -lng {lang}'
    os.system(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, required=True,
                        choices=['ga', 'mt', 'cy', 'br', 'en'])
    parser.add_argument('--model', type=str, required=True, 
                        choices=['gpt35', 'llama270bchat', 'bloom176B', 'falcon180b'])
    parser.add_argument('--prompt', type=str, required=True, choices=['zero_shot', 'few_shot'])
    parser.add_argument('--mt_system', type=str, required=False, default=None, choices=['google', 'alibaba', 'nllb'])

    args = parser.parse_args()

    if args.mt_system is not None:
        filename = f'{args.language}_tr_{args.mt_system}_{args.model}_prompt_{args.prompt}_hypothesis_processed.txt'
    else:
        filename = f'{args.language}_{args.model}_prompt_{args.prompt}_hypothesis_processed.txt'
    execute_eval(filename, args.language)


