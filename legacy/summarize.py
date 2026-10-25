import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import os
import torch
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

from ctxcomp.utils import batch_iterator, load_model
from tools.utils import load_hits_tsv

def truncate_and_concat(texts, tokenizer, max_length=512):
    tokenized = tokenizer.tokenize(texts)
    length = len(tokenizer.tokenize(texts))
    max_length = (max_length or tokenizer.max_lengt_single_sentence-1)
    if (length+6) < max_length:
        return texts
    else:
        return tokenizer.convert_tokens_to_string(tokenized[:(max_length-6)])

def load_rewrite(file):
    data_items = json.load(open(file))
    rewritten_queries = {}
    for item in data_items:
        id = str(item['requestid']) + item['colectionids'].replace('neuclir/1/', '')[:2]
        rewritten_queries[id] = item['rewrite']
    return rewritten_queries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--model_class", type=str, default=None, choices=["fid", "seq2seq", "causualLM"])
    parser.add_argument("--template", type=str, default="title: {T} content: {P}")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--eval_rewrite_file", type=str, default=None)
    parser.add_argument("--candidate_tsv", type=str, default=None)
    parser.add_argument("--output_key", type=str, default="recomp_summary")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--truncate", default=False, action='store_true')
    args = parser.parse_args()

    # load model
    model, tokenizer = load_model(args.model_name_or_path, model_class=args.model_class)
    model.eval()

    # load writer and evaluation data 
    eval_data = [json.loads(line.strip()) for line in open(args.eval_file).readlines()]
    candidates = load_hits_tsv(args.candidate_tsv)
    rewritten_queries = load_rewrite(args.eval_rewrite_file)
    writer = open(args.output_file, 'w')

    for eval_item in tqdm(eval_data, total=len(eval_data)):

        # multiple retrieved document here
        request_id = eval_item['request_id']
        lang = eval_item['collection_ids'][0].replace('neuclir/1/', '')[:2]
        # request = eval_item['problem_statement']
        request = rewritten_queries[request_id + lang].strip()
        candidate_docs = [d for d in candidates[request_id + lang]]

        # batch inference
        summaries = []
        for batch_docs in batch_iterator(candidate_docs, args.batch_size):

            batch_docs = [doc['translation'] for doc in batch_docs]

            if args.truncate:
                batch_docs = [truncate_and_concat(doc, tokenizer, max_length=args.max_length) for doc in batch_docs]

            input = list(
                args.template.replace("{Q}", request.strip()).replace("{P}", doc.strip()) \
                        for doc in batch_docs
            )
            tokenized_input = tokenizer(input, padding=True, truncation=True, max_length=args.max_length, return_tensors='pt').to(model.device)
            outputs = model.generate(
                **tokenized_input, min_new_tokens=32, max_new_tokens=512
            )
            outputs = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

            summaries.extend(outputs)

        logger.info(f"Recomp-summarizaiton: {outputs[0]}")
        for i, summary in enumerate(summaries):
            item = candidates[request_id + lang][i]
            item.update({args.output_key: summary})
            writer.write(json.dumps(item, ensure_ascii=False)+'\n')

    writer.close()

if __name__ == '__main__':
    with torch.no_grad():
        main()

