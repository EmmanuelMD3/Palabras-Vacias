import os
import re
import math
import spacy
import pandas as pd
from collections import Counter
from tqdm import tqdm
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================
# -------- Cambia aquí tus rutas --------
RESUMENES_DIR = r"C:\Users\chemo\OneDrive\Documentos\Servicio\SemEval2010_TEXTOS\resumenes_train"
KEYS_DIR      = r"C:\Users\chemo\OneDrive\Documentos\Servicio\SemEval2010_TEXTOS\frases_clave_train"

#RESUMENES_DIR = r"C:\Users\chemo\OneDrive\Documentos\Servicio\Servicio Social\resumenes_train"
#KEYS_DIR      = r"C:\Users\chemo\OneDrive\Documentos\Servicio\Servicio Social\frases_clave_train"

SPACY_MODEL = "en_core_web_sm"

# longitudes a evaluar
EVAL_LENGTHS = (1, 2, 3, 4, 5)

# greedy
PHRASE_GREEDY_A = 0.00005
MAX_PHRASE_CAND = 25000

# -------- control de longitudes --------
# Si quieres SOLO frases, pon:
# MIN_LEN = 2
# Si quieres permitir unigramas controlados, pon:
MIN_LEN = 1
MAX_LEN = 5

# -------- control de unigramas --------
ALLOW_UNIGRAMS = True
MIN_UNIGRAM_DOC_FREQ = 3
ALLOWED_UNIGRAM_STRUCTURES = {"NOUN", "PROPN"}

# conectores permitidos dentro de ciertas frases
LINKING_POS = {"ADP"}
ALLOWED_LINK_WORDS = {"of", "in", "for", "on", "with", "to", "by"}

# POS de contenido
CONTENT_POS = {"ADJ", "NOUN", "PROPN"}

# frecuencia mínima para considerar frase
MIN_GLOBAL_FREQ = 1
MIN_DOC_FREQ = 1

# score mínimo del patrón POS aprendido
MIN_PATTERN_SCORE = 0.005

# stemming opcional solo para matching
USE_STEMMING_FOR_MATCH = False

# regex para basura
TOKEN_RE = re.compile(r"^[a-z]+(?:-[a-z]+)*$")

nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])

# =========================================================
# STEMMER OPCIONAL
# =========================================================
if USE_STEMMING_FOR_MATCH:
    try:
        from nltk.stem import PorterStemmer
        STEMMER = PorterStemmer()
    except Exception:
        STEMMER = None
        USE_STEMMING_FOR_MATCH = False
else:
    STEMMER = None

# =========================================================
# IO
# =========================================================
def leer_carpeta_txt(ruta):
    archivos = {}
    for fname in sorted(os.listdir(ruta)):
        if fname.lower().endswith(".txt"):
            with open(os.path.join(ruta, fname), "r", encoding="utf-8", errors="ignore") as f:
                archivos[fname] = f.read().strip()
    return archivos

def leer_gold(keys_dir):
    gold = {}
    for fname, text in leer_carpeta_txt(keys_dir).items():
        gold[fname] = [line.strip() for line in text.splitlines() if line.strip()]
    return gold

# =========================================================
# PREPROCESAMIENTO
# =========================================================
def clean_text(text):
    text = text.lower()
    text = text.replace("\u00a0", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[/\\]+", " ", text)  # separar slash raro
    return text.strip()

def preprocess_docs(texts_dict):
    print("Preprocesando spaCy en batch...")
    cleaned = {k: clean_text(v) for k, v in texts_dict.items()}
    docs = list(nlp.pipe(cleaned.values(), batch_size=32, n_process=os.cpu_count()))
    return dict(zip(cleaned.keys(), docs))

def normalize_token(tok):
    return tok.text.lower().strip()

def valid_raw_word(word):
    if len(word) < 2:
        return False
    return TOKEN_RE.match(word) is not None

def is_valid_token(tok):
    return valid_raw_word(normalize_token(tok))

def normalize_phrase_text(text):
    """
    Normaliza una frase con lemma.
    """
    doc = nlp(clean_text(text))
    terms = []

    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue

        w = normalize_token(tok)
        if not valid_raw_word(w):
            continue

        lemma = tok.lemma_.lower().strip() if tok.lemma_ else w
        if not lemma or lemma == "-pron-":
            lemma = w
        if not valid_raw_word(lemma):
            lemma = w

        if USE_STEMMING_FOR_MATCH and STEMMER is not None:
            lemma = STEMMER.stem(lemma)

        terms.append(lemma)

    return " ".join(terms)

# =========================================================
# STOPLIST EXTENDIDA AUTOMÁTICA
# =========================================================
def build_extended_stoplist(pre_docs, top_n=120):
    """
    Stoplist extendida:
    - stopwords de spaCy
    - funcionales frecuentes
    - términos genéricos del corpus
    """
    stopset = set()

    # base spaCy
    for lex in nlp.vocab:
        if lex.is_stop:
            stopset.add(lex.text.lower())

    pos_counter = Counter()
    token_counter = Counter()

    for doc in pre_docs.values():
        for tok in doc:
            if tok.is_space or tok.is_punct:
                continue

            w = normalize_token(tok)
            if not valid_raw_word(w):
                continue

            token_counter[w] += 1

            if tok.pos_ in {"DET", "PRON", "AUX", "CCONJ", "SCONJ", "PART"}:
                pos_counter[w] += 1

    # funcionales frecuentes
    for w, _ in pos_counter.most_common(top_n):
        stopset.add(w)

    # genéricas frecuentes del dominio
    generic_candidates = {
        "paper", "approach", "method", "methods", "result", "results",
        "problem", "problems", "model", "models", "system", "systems",
        "information", "data", "analysis", "use", "used", "using",
        "user", "users", "time", "performance"
    }
    for w in generic_candidates:
        if token_counter[w] > 0:
            stopset.add(w)

    # NO quitar conectores útiles
    for w in ALLOWED_LINK_WORDS:
        stopset.discard(w)

    return stopset

# =========================================================
# MÉTRICAS
# =========================================================
def precision_recall_f_by_len(gold_list, sys_list, lengths=EVAL_LENGTHS, mode="micro"):
    gold_set = set(gold_list)
    sys_set = set(sys_list)

    def by_len(s, L):
        return {p for p in s if len(p.split()) == L}

    if mode == "micro":
        tp_total = fp_total = fn_total = 0

        for L in lengths:
            gL = by_len(gold_set, L)
            sL = by_len(sys_set, L)

            tp = len(gL & sL)
            fp = len(sL - gL)
            fn = len(gL - sL)

            tp_total += tp
            fp_total += fp
            fn_total += fn

        prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
        rec  = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
        f    = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return prec, rec, f

    # macro
    ps, rs, fs = [], [], []
    for L in lengths:
        gL = by_len(gold_set, L)
        sL = by_len(sys_set, L)

        tp = len(gL & sL)
        prec = tp / len(sL) if sL else 0.0
        rec  = tp / len(gL) if gL else 0.0
        f    = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        ps.append(prec)
        rs.append(rec)
        fs.append(f)

    return sum(ps) / len(ps), sum(rs) / len(rs), sum(fs) / len(fs)

# =========================================================
# GOLD NORMALIZADO
# =========================================================
def build_gold_normalized(gold_per_doc):
    gold_norm_per_doc = {}
    gold_all_norm = []

    for fname, phrases in gold_per_doc.items():
        norm_list = []
        for p in phrases:
            norm = normalize_phrase_text(p)
            if norm:
                norm_list.append(norm)
                gold_all_norm.append(norm)
        gold_norm_per_doc[fname] = norm_list

    return gold_norm_per_doc, gold_all_norm

# =========================================================
# TOKENS ÚTILES POR DOC
# =========================================================
def doc_to_units(doc, stopset):
    """
    Convierte doc a una lista:
    [(surface, normalized, pos), ...]
    Conserva conectores útiles ADP (of/in/for/...)
    """
    units = []

    for tok in doc:
        if tok.is_space or tok.is_punct:
            units.append(None)
            continue

        raw = normalize_token(tok)
        if not valid_raw_word(raw):
            units.append(None)
            continue

        pos = tok.pos_
        lemma = tok.lemma_.lower().strip() if tok.lemma_ else raw
        if not lemma or lemma == "-pron-":
            lemma = raw
        if not valid_raw_word(lemma):
            lemma = raw

        if USE_STEMMING_FOR_MATCH and STEMMER is not None:
            lemma = STEMMER.stem(lemma)

        # contenido
        if pos in CONTENT_POS:
            if raw in stopset:
                units.append(None)
            else:
                units.append((raw, lemma, pos))

        # conectores útiles
        elif pos in LINKING_POS and raw in ALLOWED_LINK_WORDS:
            units.append((raw, raw, pos))

        else:
            units.append(None)

    return units

# =========================================================
# PATRONES POS DESDE GOLD
# =========================================================
def learn_pos_patterns(pre_docs, gold_norm_per_doc, stopset):
    pattern_hits = Counter()
    pattern_total = Counter()

    for fname, doc in pre_docs.items():
        gold_set = set(gold_norm_per_doc.get(fname, []))
        units = doc_to_units(doc, stopset)

        start = 0
        while start < len(units):
            while start < len(units) and units[start] is None:
                start += 1
            if start >= len(units):
                break

            end = start
            while end < len(units) and units[end] is not None:
                end += 1

            seg = units[start:end]
            seg_len = len(seg)

            for L in range(MIN_LEN, min(MAX_LEN, seg_len) + 1):
                for i in range(seg_len - L + 1):
                    sub = seg[i:i + L]
                    norm = [x[1] for x in sub]
                    poss = [x[2] for x in sub]

                    # restricciones
                    if poss[-1] not in {"NOUN", "PROPN"}:
                        continue
                    if poss[0] == "ADP":
                        continue
                    if poss[-1] == "ADP":
                        continue
                    if poss.count("ADP") > 1:
                        continue

                    pattern = "+".join(poss)
                    phrase_norm = " ".join(norm)

                    pattern_total[pattern] += 1
                    if phrase_norm in gold_set:
                        pattern_hits[pattern] += 1

            start = end

    pattern_score = {}
    for pat, total in pattern_total.items():
        hits = pattern_hits.get(pat, 0)
        pattern_score[pat] = hits / total if total else 0.0

    return pattern_score, pattern_hits, pattern_total

# =========================================================
# CANDIDATAS
# =========================================================
def build_phrase_candidates(pre_docs, stopset, pattern_score):
    freq_global = Counter()
    doc_freq = Counter()
    phrase_to_struct = {}
    phrase_to_len = {}
    phrase_to_original = {}

    for fname, doc in pre_docs.items():
        units = doc_to_units(doc, stopset)
        seen_doc = set()

        start = 0
        while start < len(units):
            while start < len(units) and units[start] is None:
                start += 1
            if start >= len(units):
                break

            end = start
            while end < len(units) and units[end] is not None:
                end += 1

            seg = units[start:end]
            seg_len = len(seg)

            for L in range(MIN_LEN, min(MAX_LEN, seg_len) + 1):
                for i in range(seg_len - L + 1):
                    sub = seg[i:i + L]
                    original = [x[0] for x in sub]
                    normalized = [x[1] for x in sub]
                    poss = [x[2] for x in sub]

                    # restricciones
                    if poss[-1] not in {"NOUN", "PROPN"}:
                        continue
                    if poss[0] == "ADP":
                        continue
                    if poss[-1] == "ADP":
                        continue
                    if poss.count("ADP") > 1:
                        continue

                    structure = "+".join(poss)
                    pscore = pattern_score.get(structure, 0.0)
                    if pscore < MIN_PATTERN_SCORE:
                        continue

                    phrase_norm = " ".join(normalized)
                    phrase_orig = " ".join(original)

                    freq_global[phrase_norm] += 1
                    phrase_to_struct.setdefault(phrase_norm, structure)
                    phrase_to_len.setdefault(phrase_norm, L)
                    phrase_to_original.setdefault(phrase_norm, phrase_orig)

                    if phrase_norm not in seen_doc:
                        doc_freq[phrase_norm] += 1
                        seen_doc.add(phrase_norm)

            start = end

    # -------- filtrado mínimo + control de unigramas --------
    candidates = []
    for p in freq_global:
        if freq_global[p] < MIN_GLOBAL_FREQ:
            continue
        if doc_freq[p] < MIN_DOC_FREQ:
            continue

        L = phrase_to_len[p]
        structure = phrase_to_struct[p]

        # control especial para unigramas
        if L == 1:
            if not ALLOW_UNIGRAMS:
                continue

            # solo dejar unigramas si aparecen en varios documentos
            if doc_freq[p] < MIN_UNIGRAM_DOC_FREQ:
                continue

            # y además que su estructura sea nominal
            if structure not in ALLOWED_UNIGRAM_STRUCTURES:
                continue

        candidates.append(p)

    def rank_score(p):
        fg = freq_global[p]
        df = doc_freq[p]
        ps = pattern_score.get(phrase_to_struct[p], 0.0)
        L = phrase_to_len[p]

        score = (
            0.40 * math.log1p(fg) +
            0.30 * math.log1p(df) +
            0.30 * ps
        )

        # ligera preferencia por 2-3 grams
        if L == 2:
            score += 0.08
        elif L == 3:
            score += 0.06
        elif L == 1:
            score += 0.02

        return score

    candidates = sorted(candidates, key=rank_score, reverse=True)[:MAX_PHRASE_CAND]

    return candidates, freq_global, doc_freq, phrase_to_struct, phrase_to_len, phrase_to_original

# =========================================================
# GREEDY DE FRASES
# =========================================================
def phrase_greedy_selection(
    candidates,
    gold_all_norm,
    freq_global,
    doc_freq,
    phrase_to_struct,
    phrase_to_len,
    phrase_to_original,
    pattern_score
):
    sys_set = set()
    _, _, f_base = precision_recall_f_by_len(gold_all_norm, sys_set, lengths=EVAL_LENGTHS, mode="micro")
    print(f"[PHRASE] Baseline sys=∅ F={f_base:.4f}")

    accepted = []
    rejected = []
    gold_set = set(gold_all_norm)

    for phrase_norm in tqdm(candidates, desc="Phrase-greedy"):
        trial = sys_set | {phrase_norm}
        _, _, f_trial = precision_recall_f_by_len(gold_all_norm, trial, lengths=EVAL_LENGTHS, mode="micro")
        delta = f_trial - f_base

        structure = phrase_to_struct.get(phrase_norm, "")
        row = {
            "phrase_original": phrase_to_original.get(phrase_norm, phrase_norm),
            "phrase_normalized": phrase_norm,
            "structure": structure,
            "len": phrase_to_len.get(phrase_norm, len(phrase_norm.split())),
            "freq_global": freq_global.get(phrase_norm, 0),
            "doc_freq": doc_freq.get(phrase_norm, 0),
            "pattern_score": round(pattern_score.get(structure, 0.0), 6),
            "in_gold": phrase_norm in gold_set,
        }

        if delta > PHRASE_GREEDY_A:
            sys_set.add(phrase_norm)
            f_base = f_trial
            row.update({
                "status": "accepted",
                "delta_f": delta,
                "f_score_after": f_base
            })
            accepted.append(row)
        else:
            row.update({
                "status": "rejected",
                "delta_f": delta,
                "f_score_after": None
            })
            rejected.append(row)

    p, r, f = precision_recall_f_by_len(gold_all_norm, sys_set, lengths=EVAL_LENGTHS, mode="micro")
    print(f"\n[PHRASE] Final selected={len(sys_set)} P={p:.4f} R={r:.4f} F={f:.4f}")

    return sys_set, accepted, rejected, (p, r, f)

# =========================================================
# MAIN
# =========================================================
def main():
    resumenes = leer_carpeta_txt(RESUMENES_DIR)
    gold = leer_gold(KEYS_DIR)

    inter = sorted(set(resumenes) & set(gold))
    resumenes = {k: resumenes[k] for k in inter}
    gold = {k: gold[k] for k in inter}

    print(f"[INFO] Docs con intersección: {len(inter)}")

    pre_docs = preprocess_docs(resumenes)

    # stoplist extendida automática
    stopset = build_extended_stoplist(pre_docs, top_n=120)
    print(f"[INFO] Stoplist extendida: {len(stopset)}")

    # gold normalizado
    gold_norm_per_doc, gold_all_norm = build_gold_normalized(gold)
    print(f"[INFO] Gold normalizado único: {len(set(gold_all_norm))}")

    # aprender patrones POS
    pattern_score, pattern_hits, pattern_total = learn_pos_patterns(pre_docs, gold_norm_per_doc, stopset)
    print(f"[INFO] Patrones aprendidos: {len(pattern_score)}")

    # generar candidatas
    candidates, freq_global, doc_freq, phrase_to_struct, phrase_to_len, phrase_to_original = build_phrase_candidates(
        pre_docs, stopset, pattern_score
    )
    print(f"[INFO] Candidatas generadas: {len(candidates)}")

    # greedy
    sys_set, accepted, rejected, (p, r, f) = phrase_greedy_selection(
        candidates,
        gold_all_norm,
        freq_global,
        doc_freq,
        phrase_to_struct,
        phrase_to_len,
        phrase_to_original,
        pattern_score
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # log principal
    out_csv = f"phrase_decisions_preprocessed_{timestamp}.csv"
    df = pd.DataFrame(accepted + rejected, columns=[
        "phrase_original",
        "phrase_normalized",
        "structure",
        "len",
        "freq_global",
        "doc_freq",
        "pattern_score",
        "in_gold",
        "status",
        "delta_f",
        "f_score_after"
    ])
    df.to_csv(out_csv, index=False)
    print(f"[OK] Log guardado en: {out_csv}")

    # frases seleccionadas
    selected_csv = f"selected_phrases_preprocessed_{timestamp}.csv"
    pd.DataFrame({
        "phrase_normalized": sorted(sys_set),
        "phrase_original": [phrase_to_original.get(p, p) for p in sorted(sys_set)]
    }).to_csv(selected_csv, index=False)
    print(f"[OK] Frases seleccionadas guardadas en: {selected_csv}")

    # patrones aprendidos
    pat_rows = []
    for pat in sorted(pattern_score.keys(), key=lambda x: pattern_score[x], reverse=True):
        pat_rows.append({
            "pattern": pat,
            "score": pattern_score[pat],
            "hits_in_gold": pattern_hits.get(pat, 0),
            "total_seen": pattern_total.get(pat, 0),
        })
    pat_csv = f"learned_patterns_preprocessed_{timestamp}.csv"
    pd.DataFrame(pat_rows).to_csv(pat_csv, index=False)
    print(f"[OK] Patrones guardados en: {pat_csv}")

    # resumen final
    summary_csv = f"summary_preprocessed_{timestamp}.csv"
    pd.DataFrame([{
        "docs": len(inter),
        "stoplist_size": len(stopset),
        "gold_unique": len(set(gold_all_norm)),
        "candidates": len(candidates),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "precision": p,
        "recall": r,
        "f_score": f
    }]).to_csv(summary_csv, index=False)
    print(f"[OK] Resumen guardado en: {summary_csv}")

if __name__ == "__main__":
    main()