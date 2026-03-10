import os
import spacy
from collections import Counter
import pandas as pd
from itertools import product
from tqdm import tqdm
import re
from datetime import datetime

"""
Versión optimizada del método del artículo:
"Stop-words in Keyphrase Extraction Problem"

Cambios aplicados (según lo que pediste):
✅ El extractor genera FRASES (n-gramas) 2..N y NO tokens sueltos
✅ Se filtran símbolos y letras sueltas (%, -, a, f, etc.)
✅ La métrica (F-score) se calcula por longitud (2 con 2, 3 con 3, etc.)
✅ El log que se guarda en Excel/CSV incluye FRASES generadas (y si están en GOLD)
✅ El log de decisiones de stopwords usa timestamp para evitar "Permission denied"

Requiere:
pip install spacy pandas tqdm
python -m spacy download en_core_web_sm
"""


# --------- CONFIGURACIÓN ---------
RESUMENES_DIR = r"C:\Users\chemo\OneDrive\Documentos\Cosigos Python\Servicio Social\resumenes_train"
KEYS_DIR     = r"C:\Users\chemo\OneDrive\Documentos\Cosigos Python\Servicio Social\frases_clave_train"

SPACY_MODEL = "en_core_web_sm"

TOP_N_STOP_CANDIDATES = 200
FREQ_STEPS = [50, 100, 200]
NONSTOP_STEPS = [0, 50, 100]

TRAIN_METHOD_A = 0.0001
MAX_TRAIN_CAND = 1000

# Longitudes que quieres considerar en la evaluación (estructura)
EVAL_LENGTHS = (2, 3, 4, 5)

# n-gramas que el extractor genera
MIN_N = 2
MAX_N = 5

CSV_OUT = "results_advanced_method.csv"

nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])

# Solo letras (minúsculas) con guiones internos opcionales: state-of-the-art
TOKEN_RE = re.compile(r"^[a-z]+(?:-[a-z]+)*$")


# --------- UTILIDADES ---------
def leer_carpeta_txt(ruta):
    archivos = {}
    for fname in sorted(os.listdir(ruta)):
        if fname.lower().endswith(".txt"):
            with open(os.path.join(ruta, fname), "r", encoding="utf-8", errors="ignore") as f:
                archivos[fname] = f.read().strip()
    return archivos


def leer_gold_per_doc(keys_dir):
    gold = {}
    for fname, text in leer_carpeta_txt(keys_dir).items():
        gold[fname] = [line.strip().lower() for line in text.splitlines() if line.strip()]
    return gold


def normalize_token(tok):
    return tok.text.lower().strip()


def is_valid_token(tok):
    """
    Filtro anti-basura:
    - longitud mínima 2
    - solo letras (o con guiones internos)
    """
    w = normalize_token(tok)
    if len(w) < 2:
        return False
    return TOKEN_RE.match(w) is not None


# --------- PREPROCESAMIENTO spaCy ---------
def preprocess_docs(resumenes):
    print("Preprocesando spaCy en batch...")
    docs = list(nlp.pipe(
        resumenes.values(),
        batch_size=32,
        n_process=os.cpu_count()
    ))
    return dict(zip(resumenes.keys(), docs))


# --------- EXTRACCIÓN DE CANDIDATOS (FRASES) ---------
def extract_candidates_from_doc(doc, stopset, min_n=MIN_N, max_n=MAX_N):
    """
    Genera candidatos como n-gramas (min_n..max_n) basados en POS,
    permitiendo conectores (stopwords) dentro si NO están en stopset.
    Corta secuencia en:
      - puntuación
      - espacios
      - tokens inválidos (%, -, una letra, etc.)
      - tokens en stopset
    """

    CONTENT_POS = {"NOUN", "PROPN", "ADJ"}
    FUNCTION_POS = {"ADP", "DET", "PART", "CCONJ", "SCONJ", "PRON", "AUX"}

    usable = []
    for tok in doc:
        w = normalize_token(tok)

        if tok.is_space or tok.is_punct:
            usable.append(None)
            continue

        if w in stopset:
            usable.append(None)
            continue

        if not is_valid_token(tok):
            usable.append(None)
            continue

        if tok.pos_ in CONTENT_POS or tok.pos_ in FUNCTION_POS:
            usable.append((w, tok.pos_))
        else:
            usable.append(None)

    out = set()

    start = 0
    while start < len(usable):
        while start < len(usable) and usable[start] is None:
            start += 1
        if start >= len(usable):
            break

        end = start
        while end < len(usable) and usable[end] is not None:
            end += 1

        segment = usable[start:end]  # [(w,pos),...]
        seg_len = len(segment)

        for n in range(min_n, max_n + 1):
            if seg_len < n:
                continue
            for i in range(seg_len - n + 1):
                window = segment[i:i+n]
                words = [x[0] for x in window]
                poses = [x[1] for x in window]

                # Reglas de estructura:
                # 1) Terminar en NOUN/PROPN (típico de keyphrase)
                if poses[-1] not in {"NOUN", "PROPN"}:
                    continue

                # 2) Contener al menos un NOUN/PROPN
                if not any(p in {"NOUN", "PROPN"} for p in poses):
                    continue

                # 3) Evitar solo funcionales
                if all(p in FUNCTION_POS for p in poses):
                    continue

                out.add(" ".join(words))

        start = end

    return list(out)


def extract_all(pre_docs, stopset):
    all_sys = []
    for doc in pre_docs.values():
        all_sys.extend(extract_candidates_from_doc(doc, stopset))
    return all_sys


# --------- MÉTRICAS ---------
def precision_recall_f_by_len(gold_list, sys_list, lengths=EVAL_LENGTHS, mode="micro"):
    """
    Evalúa por estructura/longitud:
      - Compara 2-gram con 2-gram, 3 con 3, etc.
    mode:
      - "micro": suma TP/FP/FN en todas las longitudes (recomendado)
      - "macro": promedio simple de F por longitud
    """
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
        f = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        ps.append(prec); rs.append(rec); fs.append(f)
    return sum(ps)/len(ps), sum(rs)/len(rs), sum(fs)/len(fs)


def print_f_by_length(gold_list, sys_list, lengths=EVAL_LENGTHS):
    gold_set = set(gold_list)
    sys_set = set(sys_list)

    print("\n===== F-score por longitud =====")
    for L in lengths:
        gL = {p for p in gold_set if len(p.split()) == L}
        sL = {p for p in sys_set if len(p.split()) == L}
        tp = len(gL & sL)
        prec = tp / len(sL) if sL else 0.0
        rec  = tp / len(gL) if gL else 0.0
        f = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(f"{L}-gram → P={prec:.4f} R={rec:.4f} F={f:.4f} | gold={len(gL)} sys={len(sL)} tp={tp}")


# --------- FRECUENCIAS Y H-POINT ---------
def build_frequency_dicts(pre_docs):
    corpus_freq = Counter()
    candidate_token_freq = Counter()

    allowed_pos = {"NOUN", "ADJ", "PROPN", "ADP", "DET", "PART", "CCONJ", "SCONJ", "PRON", "AUX"}

    for doc in pre_docs.values():
        for tok in doc:
            if tok.pos_ in allowed_pos and is_valid_token(tok):
                corpus_freq[normalize_token(tok)] += 1

        cands = extract_candidates_from_doc(doc, set())
        for phrase in cands:
            for token in phrase.split():
                candidate_token_freq[token] += 1

    return corpus_freq, candidate_token_freq


def compute_h_point(freq_counter):
    freqs = sorted(freq_counter.values(), reverse=True)
    h = 0
    for i, f in enumerate(freqs, start=1):
        if f >= i:
            h = i
        else:
            break
    return h


# --------- MÉTODO 1: FRECUENCIA + COMPENSACIÓN ---------
def frequency_based_method(pre_docs, gold_all):
    corpus_freq, cand_token_freq = build_frequency_dicts(pre_docs)

    corpus_sorted = [w for w, _ in corpus_freq.most_common(TOP_N_STOP_CANDIDATES)]
    cand_sorted   = [w for w, _ in cand_token_freq.most_common(TOP_N_STOP_CANDIDATES)]

    print(f"[INFO] h-point: {compute_h_point(cand_token_freq)}")

    results = []

    for stop_k, non_k in product(FREQ_STEPS, NONSTOP_STEPS):
        stop_candidates    = set(corpus_sorted[:stop_k])
        nonstop_candidates = set(cand_sorted[:non_k])

        final_stoplist = stop_candidates - nonstop_candidates

        sys_phrases = extract_all(pre_docs, final_stoplist)
        prec, rec, f = precision_recall_f_by_len(gold_all, sys_phrases, lengths=EVAL_LENGTHS, mode="micro")

        print(f"[FREQ] stop_k={stop_k} non_k={non_k} F={f:.4f}")

        results.append({
            "method": "frequency_compensation",
            "stop_k": stop_k,
            "non_k": non_k,
            "stop_count": len(final_stoplist),
            "precision": prec,
            "recall": rec,
            "f_score": f
        })

    return results


# --------- MÉTODO 2: TRAIN-BASED (GREEDY STOPLIST) ---------
def train_based_method(pre_docs, gold_all):
    """
    OJO: este método aprende STOPWORDS (tokens a eliminar).
    Lo que cambia aquí es:
      - la métrica ahora es por longitud (estructura)
      - se guardan también las FRASES del sistema final en otro CSV
      - se evita PermissionError usando timestamp en el nombre del log
    """

    corpus_freq, _ = build_frequency_dicts(pre_docs)
    candidates = [w for w, _ in corpus_freq.most_common(MAX_TRAIN_CAND) if len(w) >= 2 and TOKEN_RE.match(w)]

    baseline_sys = extract_all(pre_docs, set())
    _, _, f_base = precision_recall_f_by_len(gold_all, baseline_sys, lengths=EVAL_LENGTHS, mode="micro")
    print(f"[TRAIN] Baseline F(struct)={f_base:.4f}")
    print_f_by_length(gold_all, baseline_sys, lengths=EVAL_LENGTHS)

    selected_stopwords = set()
    accepted_tokens = []
    rejected_tokens = []

    for token in tqdm(candidates, desc="Train-based"):
        trial_stop = selected_stopwords | {token}
        sys_phrases = extract_all(pre_docs, trial_stop)

        _, _, f_trial = precision_recall_f_by_len(gold_all, sys_phrases, lengths=EVAL_LENGTHS, mode="micro")
        improvement = f_trial - f_base

        if improvement > TRAIN_METHOD_A:
            selected_stopwords.add(token)
            f_base = f_trial
            accepted_tokens.append((token, improvement, f_base))
            print(f"  [+] {token} accepted → ΔF={improvement:.6f} → F={f_base:.4f}")
        else:
            rejected_tokens.append((token, improvement))

    sys_final = extract_all(pre_docs, selected_stopwords)
    prec, rec, f_final = precision_recall_f_by_len(gold_all, sys_final, lengths=EVAL_LENGTHS, mode="micro")

    print(f"\n[TRAIN] Final stopwords={len(selected_stopwords)} F(struct)={f_final:.4f}")
    print_f_by_length(gold_all, sys_final, lengths=EVAL_LENGTHS)

    # --------- LOG DE DECISIONES (STOPWORDS) ---------
    df_train_log = pd.DataFrame(
        [(t, "accepted", imp, f) for t, imp, f in accepted_tokens] +
        [(t, "rejected", imp, None) for t, imp in rejected_tokens],
        columns=["token", "status", "delta_f", "f_score_after"]
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_log_name = f"train_decisions_log_{timestamp}.csv"
    try:
        df_train_log.to_csv(train_log_name, index=False)
        print(f"Log stopwords guardado en: {train_log_name}")
    except PermissionError:
        print("No se pudo guardar el log de stopwords (¿abierto en Excel?).")

    # --------- LOG DE FRASES DEL SISTEMA FINAL ---------
    gold_set = set(gold_all)
    df_phrases = pd.DataFrame({"phrase": sorted(set(sys_final))})
    df_phrases["len"] = df_phrases["phrase"].str.split().str.len()
    df_phrases["in_gold"] = df_phrases["phrase"].isin(gold_set)

    phrases_log_name = f"sys_phrases_log_{timestamp}.csv"
    try:
        df_phrases.to_csv(phrases_log_name, index=False)
        print(f"Log de FRASES guardado en: {phrases_log_name}")
    except PermissionError:
        print("No se pudo guardar el log de frases (¿abierto en Excel?).")

    return {
        "method": "train_based",
        "stopwords": selected_stopwords,
        "precision": prec,
        "recall": rec,
        "f_score": f_final
    }


# --------- FUNCIÓN PRINCIPAL ---------
def main():
    resumenes = leer_carpeta_txt(RESUMENES_DIR)
    gold = leer_gold_per_doc(KEYS_DIR)

    inter = sorted(set(resumenes) & set(gold))
    resumenes = {k: resumenes[k] for k in inter}
    gold = {k: gold[k] for k in inter}

    gold_all = []
    for g in gold.values():
        gold_all.extend(g)

    pre_docs = preprocess_docs(resumenes)

    # Baseline sin stoplist
    baseline_sys = extract_all(pre_docs, set())
    p0, r0, f0 = precision_recall_f_by_len(gold_all, baseline_sys, lengths=EVAL_LENGTHS, mode="micro")
    print(f"Baseline(struct) → P={p0:.4f} R={r0:.4f} F={f0:.4f}")
    print_f_by_length(gold_all, baseline_sys, lengths=EVAL_LENGTHS)

    # Método frecuencia
    freq_results = frequency_based_method(pre_docs, gold_all)

    # Método train-based
    train_result = train_based_method(pre_docs, gold_all)

    # Guardar resultados
    rows = freq_results + [{
        "method": "train_based",
        "stop_k": None,
        "non_k": None,
        "stop_count": len(train_result["stopwords"]),
        "precision": train_result["precision"],
        "recall": train_result["recall"],
        "f_score": train_result["f_score"]
    }]

    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUT, index=False)

    print("\nTop resultados:")
    print(df.sort_values("f_score", ascending=False).head(10).to_string(index=False))
    print(f"\nResultados guardados en: {CSV_OUT}")


if __name__ == "__main__":
    main()