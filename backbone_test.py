import torch
from models.load_hnet import load_hnet_encoder
from transformers import AutoTokenizer
from configs.default import DEVICE, DTYPE, MODEL_ID

encoder, _ = load_hnet_encoder(
    'runs_gemma_v2/hnet_pretrain_pos_guided/best.pt',
    MODEL_ID, str(DEVICE), DTYPE, True, False, 1)
encoder.eval()

# Load Gemma tokenizer for subword stats
tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-4b-pt')

from scripts.eval_chunk_pos import read_conll

languages = {
    'Hindi':    'data/ud_hindi_treebank/hi_hdtb-ud-train.conllu',
    'Bhojpuri': 'data/ud_bhojpuri/bho_bhtb-ud-test.conllu',
    'Marathi':  'data/ud_marathi/mr_ufal-ud-train.conllu',
    'Magahi':   'data/ud_magahi/mag_mgtb-ud-test.conllu',
    'Sanskrit': 'data/ud_sanskrit/sa_ufal-ud-test.conllu',
    'Urdu':     'data/ud_urdu/ur_udtb-ud-test.conllu',
}

print('%-12s | %8s | %10s | %7s | %5s | %8s | %5s | %10s' % (
    'Language', 'Words', 'Subwords', '% Split', 'SW/W', 'Chunks', 'C/W', '1:1 Align%'))
print('-'*85)

for lang, path in languages.items():
    sentences = read_conll(path)
    total_words      = 0
    total_subwords   = 0
    total_split      = 0
    total_chunks     = 0
    aligned_1to1     = 0
    n_sents          = 0

    for words, tags in sentences:
        text = ' '.join(words)
        byte_ids = list(text.encode('utf-8')) + [1]
        input_ids = torch.tensor([byte_ids], dtype=torch.long).to(DEVICE)
        attention_mask = (input_ids != 0).long()

        with torch.no_grad():
            h_hat, p, aux = encoder(input_ids, attention_mask)
            chunk_ids = aux['chunk_out']['chunk_ids'][0]

        n_words  = len(words)
        n_chunks = max(1, chunk_ids.max().item())

        # Subword stats per word
        n_subwords = 0
        n_split    = 0
        for word in words:
            toks = tokenizer.tokenize(word)
            n_subwords += len(toks)
            if len(toks) > 1:
                n_split += 1

        total_words    += n_words
        total_subwords += n_subwords
        total_split    += n_split
        total_chunks   += n_chunks
        n_sents        += 1

        if abs(n_chunks - n_words) <= 1:
            aligned_1to1 += 1

    pct_split = total_split / max(total_words, 1) * 100
    sw_w      = total_subwords / max(total_words, 1)
    cw        = total_chunks / max(total_words, 1)
    align     = aligned_1to1 / max(n_sents, 1) * 100

    print('%-12s | %8s | %10s | %6.1f%% | %5.2f | %8s | %5.2f | %8.1f%%' % (
        lang,
        f'{total_words:,}',
        f'{total_subwords:,}',
        pct_split,
        sw_w,
        f'{total_chunks:,}',
        cw,
        align))